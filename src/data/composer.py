# src/data/composer.py
"""
BatchComposer: 배치 구성 비율 관리
- Normal 50% + Pseudo 30% + Hard 20%
- 언어 Ko:En = 1:1
- 스타일 Formal 50% + Casual 30% + None 20%
"""

import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .pool import DataPoolManager, Sample
from .buffer import PseudoBuffer, HardExampleBuffer, BTSample, HardSample

class SourceType(Enum):
    NORMAL = 'normal'
    PSEUDO = 'pseudo'
    HARD = 'hard'

@dataclass
class ComposedSample:
    """Composer가 출력하는 통합 샘플"""
    source_type: SourceType
    
    # L_auto용 (Normal, Hard)
    text: Optional[str] = None
    language: Optional[str] = None
    style_tag: Optional[str] = None
    
    # L_back용 (Pseudo)
    bt_source: Optional[str] = None
    bt_target: Optional[str] = None
    bt_direction: Optional[str] = None

class BatchComposer:
    """
    역할:
    - 배치 구성 비율 관리 (Normal/Pseudo/Hard)
    - 언어 균형 (Ko:En = 1:1)
    - 스타일 분포 관리
    """
    
    def __init__(
        self,
        pool: DataPoolManager,
        pseudo_buffer: PseudoBuffer,
        hard_buffer: HardExampleBuffer,
        config: Dict = None
    ):
        self.pool = pool
        self.pseudo_buffer = pseudo_buffer
        self.hard_buffer = hard_buffer
        
        config = config or {}
        
        # 배치 구성 비율
        self.normal_ratio = config.get('normal_ratio', 0.5)
        self.pseudo_ratio = config.get('pseudo_ratio', 0.3)
        self.hard_ratio = config.get('hard_ratio', 0.2)
        
        # 스타일 분포
        self.style_dist = config.get('style_distribution', {
            'formal': 0.5,
            'casual': 0.3,
            'none': 0.2
        })
        
        # L_back 활성화 상태
        self.lback_activated = False
    
    def set_lback_activated(self, activated: bool):
        """L_back 활성화 상태 설정"""
        self.lback_activated = activated
    
    def compose_batch(self, batch_size: int) -> List[ComposedSample]:
        """배치 구성"""
        batch = []
        
        for _ in range(batch_size):
            sample = self._compose_single()
            batch.append(sample)
        
        return batch
    
    def _compose_single(self) -> ComposedSample:
        """단일 샘플 구성"""
        source_type = self._select_source_type()
        
        if source_type == SourceType.NORMAL:
            return self._compose_normal()
        elif source_type == SourceType.PSEUDO:
            return self._compose_pseudo()
        else:  # HARD
            return self._compose_hard()
    
    def _select_source_type(self) -> SourceType:
        """소스 타입 선택 (비율에 따라)"""
        
        # 버퍼 상태에 따른 동적 비율 조정
        weights = [self.normal_ratio]
        types = [SourceType.NORMAL]
        
        # Pseudo: L_back 활성화 & 버퍼에 데이터 있을 때만
        if self.lback_activated and not self.pseudo_buffer.is_empty():
            weights.append(self.pseudo_ratio)
            types.append(SourceType.PSEUDO)
        
        # Hard: 버퍼에 데이터 있을 때만
        if not self.hard_buffer.is_empty():
            weights.append(self.hard_ratio)
            types.append(SourceType.HARD)
        
        # 정규화
        total = sum(weights)
        weights = [w / total for w in weights]
        
        return random.choices(types, weights=weights, k=1)[0]
    
    def _compose_normal(self) -> ComposedSample:
        """Normal 데이터 구성"""
        sample = self.pool.sample_balanced(self.style_dist)
        
        return ComposedSample(
            source_type=SourceType.NORMAL,
            text=sample.text,
            language=sample.language,
            style_tag=sample.style_tag
        )
    
    def _compose_pseudo(self) -> ComposedSample:
        """Pseudo (BT) 데이터 구성"""
        samples = self.pseudo_buffer.sample(1, direction='balanced')
        
        if not samples:
            # Fallback to normal
            return self._compose_normal()
        
        bt_sample = samples[0]
        
        return ComposedSample(
            source_type=SourceType.PSEUDO,
            bt_source=bt_sample.source,
            bt_target=bt_sample.target,
            bt_direction=bt_sample.direction
        )
    
    def _compose_hard(self) -> ComposedSample:
        """Hard Example 구성"""
        samples = self.hard_buffer.sample(1)
        
        if not samples:
            # Fallback to normal
            return self._compose_normal()
        
        hard_sample = samples[0]
        
        return ComposedSample(
            source_type=SourceType.HARD,
            text=hard_sample.text,
            language=hard_sample.language,
            style_tag=hard_sample.style_tag
        )
    
    def get_stats(self) -> Dict:
        """현재 상태 통계"""
        return {
            'pool_size': len(self.pool),
            'pseudo_buffer_size': len(self.pseudo_buffer),
            'hard_buffer_size': len(self.hard_buffer),
            'lback_activated': self.lback_activated
        }

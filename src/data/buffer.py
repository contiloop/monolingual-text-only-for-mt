# src/data/buffer.py
"""
PseudoBuffer: BT 생성 결과 저장 및 샘플링
HardExampleBuffer: Loss 높은 샘플 저장 및 재샘플링
"""

import json
import random
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import threading

@dataclass
class BTSample:
    """Back-Translation 샘플"""
    source: str       # 생성된 pseudo source
    target: str       # 원본 monolingual (real target)
    direction: str    # 'ko_to_en' or 'en_to_ko'
    score: float = 0.0  # 품질 점수 (필터링용)

@dataclass
class HardSample:
    """Hard Example 샘플"""
    text: str
    language: str
    style_tag: str
    loss: float
    step_added: int

class PseudoBuffer:
    """
    역할:
    - vLLM 생성 결과 (bt_cache/*.jsonl) 저장
    - 방향별 (ko→en, en→ko) 균형 샘플링
    - 5,000 step마다 리로드
    """
    
    def __init__(self, max_size: int = 50000):
        self.max_size = max_size
        self.ko_to_en: deque = deque(maxlen=max_size // 2)
        self.en_to_ko: deque = deque(maxlen=max_size // 2)
        self._lock = threading.Lock()
    
    def load_from_file(self, path: str):
        """BT 캐시 파일에서 로드"""
        with self._lock:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    sample = BTSample(
                        source=data.get('source', ''),
                        target=data.get('generated', ''),
                        direction=data.get('direction', 'unknown')
                    )
                    
                    if sample.direction == 'ko_to_en':
                        self.ko_to_en.append(sample)
                    else:
                        self.en_to_ko.append(sample)
        
        print(f"PseudoBuffer 로드: ko→en {len(self.ko_to_en)}, en→ko {len(self.en_to_ko)}")
    
    def add(self, sample: BTSample):
        """샘플 추가"""
        with self._lock:
            if sample.direction == 'ko_to_en':
                self.ko_to_en.append(sample)
            else:
                self.en_to_ko.append(sample)
    
    def sample(self, n: int = 1, direction: str = 'balanced') -> List[BTSample]:
        """
        샘플링
        direction: 'ko_to_en', 'en_to_ko', 'balanced'
        """
        with self._lock:
            if direction == 'ko_to_en':
                pool = list(self.ko_to_en)
            elif direction == 'en_to_ko':
                pool = list(self.en_to_ko)
            else:  # balanced
                pool = list(self.ko_to_en) + list(self.en_to_ko)
            
            if len(pool) < n:
                return pool
            return random.sample(pool, n)
    
    def clear(self):
        """버퍼 초기화"""
        with self._lock:
            self.ko_to_en.clear()
            self.en_to_ko.clear()
    
    def __len__(self):
        return len(self.ko_to_en) + len(self.en_to_ko)
    
    def is_empty(self):
        return len(self) == 0


class HardExampleBuffer:
    """
    역할:
    - Loss 높은 샘플 저장
    - 상위 20% 유지
    - 배치 구성 시 20% 비율로 샘플링
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        loss_percentile: float = 0.2
    ):
        self.max_size = max_size
        self.loss_percentile = loss_percentile
        self.buffer: List[HardSample] = []
        self._lock = threading.Lock()
        self.loss_threshold = float('inf')  # 동적으로 업데이트
    
    def add(self, text: str, language: str, style_tag: str, loss: float, step: int):
        """Loss 높은 샘플 추가"""
        with self._lock:
            sample = HardSample(
                text=text,
                language=language,
                style_tag=style_tag,
                loss=loss,
                step_added=step
            )
            self.buffer.append(sample)
            
            # 크기 제한
            if len(self.buffer) > self.max_size:
                # Loss 기준 정렬 후 상위만 유지
                self.buffer.sort(key=lambda x: x.loss, reverse=True)
                self.buffer = self.buffer[:self.max_size]
                
                # Threshold 업데이트
                self._update_threshold()
    
    def _update_threshold(self):
        """Loss threshold 업데이트 (상위 percentile 기준)"""
        if not self.buffer:
            self.loss_threshold = float('inf')
            return
        
        sorted_losses = sorted([s.loss for s in self.buffer], reverse=True)
        idx = int(len(sorted_losses) * self.loss_percentile)
        self.loss_threshold = sorted_losses[min(idx, len(sorted_losses) - 1)]
    
    def sample(self, n: int = 1) -> List[HardSample]:
        """Hard Example 샘플링"""
        with self._lock:
            if len(self.buffer) < n:
                return self.buffer.copy()
            return random.sample(self.buffer, n)
    
    def should_add(self, loss: float) -> bool:
        """이 loss가 추가할 만한 수준인지 체크"""
        return loss >= self.loss_threshold
    
    def clear(self):
        with self._lock:
            self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)
    
    def is_empty(self):
        return len(self.buffer) == 0

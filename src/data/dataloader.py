# src/data/dataloader.py
"""
메인 DataLoader 진입점
모든 컴포넌트를 조립하여 학습용 DataLoader 생성
"""

import json
from pathlib import Path
from typing import Dict, Iterator, Optional
from torch.utils.data import IterableDataset, DataLoader

from .pool import DataPoolManager
from .buffer import PseudoBuffer, HardExampleBuffer
from .composer import BatchComposer, ComposedSample
from .collator import TranslationCollator, CollatedBatch
from .noise import NoiseApplier, NoiseConfig

class FinancialTranslationDataset(IterableDataset):
    """
    금융 번역 모델 학습용 IterableDataset
    
    상태 관리:
    - set_step(): 커리큘럼 업데이트
    - set_lback_activated(): L_back 활성화
    - reload_bt_cache(): BT 데이터 리로드
    - add_hard_example(): Hard Example 추가
    """
    
    def __init__(
        self,
        pool: DataPoolManager,
        pseudo_buffer: PseudoBuffer,
        hard_buffer: HardExampleBuffer,
        composer: BatchComposer,
        collator: TranslationCollator,
        config: Dict
    ):
        self.pool = pool
        self.pseudo_buffer = pseudo_buffer
        self.hard_buffer = hard_buffer
        self.composer = composer
        self.collator = collator
        self.config = config
        
        # 상태
        self.current_step = 0
        self.lback_activated = False
    
    def __iter__(self) -> Iterator[CollatedBatch]:
        """무한 이터레이터"""
        # training.batch_size 또는 최상위 batch_size 사용
        training_config = self.config.get('training', {})
        batch_size = training_config.get('batch_size', self.config.get('batch_size', 4))

        while True:
            # 배치 구성
            composed = self.composer.compose_batch(batch_size)

            # 콜레이트 (토크나이징)
            collated = self.collator(composed)

            yield collated
    
    # === 상태 관리 인터페이스 ===
    
    def set_step(self, step: int):
        """현재 step 설정 (커리큘럼 업데이트)"""
        self.current_step = step
        
        # 노이즈 커리큘럼 업데이트
        total_steps = self.config.get('total_steps', 50000)
        progress = step / total_steps
        
        noise_curriculum = self.config.get('noise', {})
        self.collator.update_noise_config(progress, noise_curriculum)
    
    def set_lback_activated(self, activated: bool):
        """L_back 활성화 상태 설정"""
        self.lback_activated = activated
        self.composer.set_lback_activated(activated)
    
    def reload_bt_cache(self, path: str):
        """BT 캐시 리로드"""
        self.pseudo_buffer.clear()
        self.pseudo_buffer.load_from_file(path)
    
    def add_hard_example(self, text: str, language: str, style_tag: str, loss: float):
        """Hard Example 추가"""
        if self.hard_buffer.should_add(loss):
            self.hard_buffer.add(text, language, style_tag, loss, self.current_step)
    
    def get_stats(self) -> Dict:
        """현재 상태 통계"""
        return {
            'current_step': self.current_step,
            'lback_activated': self.lback_activated,
            **self.composer.get_stats()
        }


def create_dataloader(
    config: Dict,
    tokenizer,
    ko_path: str,
    en_path: str,
    glossary_path: Optional[str] = None
) -> DataLoader:
    """
    DataLoader 생성 팩토리
    
    Args:
        config: 전체 설정
        tokenizer: HuggingFace tokenizer
        ko_path: 한국어 processed 파일 경로
        en_path: 영어 processed 파일 경로
        glossary_path: 용어집 파일 경로
    
    Returns:
        DataLoader, FinancialTranslationDataset
    """
    
    # 용어집 로드
    glossary = {}
    if glossary_path and Path(glossary_path).exists():
        with open(glossary_path, 'r', encoding='utf-8') as f:
            glossary = json.load(f)
    
    # 1. DataPoolManager
    pool = DataPoolManager(
        ko_path=ko_path,
        en_path=en_path,
        val_ratio=config.get('val_ratio', 0.05)
    )
    
    # 2. Buffers
    buffer_config = config.get('buffer', {})
    pseudo_buffer = PseudoBuffer(
        max_size=buffer_config.get('pseudo_max_size', 50000)
    )
    hard_buffer = HardExampleBuffer(
        max_size=buffer_config.get('hard_max_size', 10000),
        loss_percentile=buffer_config.get('hard_percentile', 0.2)
    )
    
    # 3. Sampler (새로 추가)
    from .sampling import create_sampler
    sampler = create_sampler(config, pool)
    
    # 4. Composer
    composition_config = config.get('composition', {})
    composer = BatchComposer(
        pool=pool,
        pseudo_buffer=pseudo_buffer,
        hard_buffer=hard_buffer,
        sampler=sampler,  # 새로 추가
        config=composition_config
    )
    
    # 4. NoiseApplier
    noise_config = config.get('noise', {})
    noise_applier = NoiseApplier(NoiseConfig(
        total_ratio=noise_config.get('total_ratio', 0.15),
        deletion_prob=noise_config.get('deletion_prob', 0.15),
        filler_prob=noise_config.get('filler_prob', 0.15),
        infilling_prob=noise_config.get('infilling_prob', 0.20),
        repetition_prob=noise_config.get('repetition_prob', 0.10),
        spacing_prob=noise_config.get('spacing_prob', 0.15),
        punctuation_prob=noise_config.get('punctuation_prob', 0.10),
        newline_prob=noise_config.get('newline_prob', 0.05),
        typo_prob=noise_config.get('typo_prob', 0.10),
        asr_error_prob=noise_config.get('asr_error_prob', 0.05)
    ))
    
    # 5. Collator
    collator = TranslationCollator(
        tokenizer=tokenizer,
        noise_applier=noise_applier,
        max_length=config.get('max_length', 2048),
        glossary=glossary
    )
    
    # 6. Dataset
    dataset = FinancialTranslationDataset(
        pool=pool,
        pseudo_buffer=pseudo_buffer,
        hard_buffer=hard_buffer,
        composer=composer,
        collator=collator,
        config=config
    )
    
    # 7. DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=None,  # IterableDataset이므로 None (내부에서 배치 생성)
        num_workers=2,    # CPU 멀티코어 활용 (메모리 고려하여 2로 설정)
        pin_memory=True,  # GPU 전송 속도 향상
        prefetch_factor=2 # worker당 2개 배치 미리 준비
    )
    
    return dataloader, dataset, pseudo_buffer, hard_buffer

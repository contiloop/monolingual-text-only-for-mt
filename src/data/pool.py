# src/data/pool.py
"""
DataPoolManager: 전처리된 데이터 로드 + 문서 단위 Train/Val 분리
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class Sample:
    """단일 샘플"""
    text: str
    language: str  # 'ko' or 'en'
    style_tag: str  # '<|formal|>', '<|casual|>', ''
    source_type: str  # conference_call, article, report, bank_dict
    doc_id: str  # 문서 단위 분리용
    metadata: Optional[Dict] = None

class DataPoolManager:
    """
    역할:
    - 전처리된 JSONL 파일 로드
    - 문서 ID 기준 Train/Val 분리 (청킹 전 분리로 누수 방지)
    - 언어별/스타일별 인덱스 관리
    """
    
    def __init__(
        self,
        ko_path: str,
        en_path: str,
        val_ratio: float = 0.05,
        seed: int = 42
    ):
        self.val_ratio = val_ratio
        self.seed = seed
        random.seed(seed)
        
        # 데이터 로드
        self.ko_samples = self._load_jsonl(ko_path, 'ko')
        self.en_samples = self._load_jsonl(en_path, 'en')
        
        # Train/Val 분리 (문서 단위)
        self.train_pool, self.val_pool = self._split_by_document()
        
        # 빠른 샘플링을 위한 인덱스
        self._build_indices()
        
        print(f"DataPool 초기화 완료:")
        print(f"  - 한국어: {len(self.ko_samples):,} samples")
        print(f"  - 영어: {len(self.en_samples):,} samples")
        print(f"  - Train: {len(self.train_pool):,} / Val: {len(self.val_pool):,}")
    
    def _load_jsonl(self, path: str, language: str) -> List[Sample]:
        """JSONL 파일 로드"""
        samples = []
        path = Path(path)
        
        if not path.exists():
            print(f"Warning: {path} not found. Returning empty list.")
            return samples
        
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    data = json.loads(line)
                    
                    # 텍스트 추출
                    text = data.get('text', '')
                    if not text or len(text.strip()) < 50:
                        continue
                    
                    # 문서 ID 생성 (메타데이터에서 추출 또는 자동 생성)
                    metadata = data.get('metadata', {})
                    doc_id = metadata.get('source', '') + '_' + str(metadata.get('url', idx))
                    
                    sample = Sample(
                        text=text,
                        language=language,
                        style_tag=data.get('style_tag', ''),
                        source_type=metadata.get('source', 'unknown'),
                        doc_id=doc_id,
                        metadata=metadata
                    )
                    samples.append(sample)
                    
                except json.JSONDecodeError:
                    continue
        
        return samples
    
    def _split_by_document(self) -> Tuple[List[Sample], List[Sample]]:
        """문서 ID 기준으로 Train/Val 분리 (누수 방지)"""
        
        all_samples = self.ko_samples + self.en_samples
        
        # 문서 ID별로 그룹핑
        doc_groups = defaultdict(list)
        for sample in all_samples:
            doc_groups[sample.doc_id].append(sample)
        
        # 문서 ID 셔플 후 분리
        doc_ids = list(doc_groups.keys())
        random.shuffle(doc_ids)
        
        val_count = max(1, int(len(doc_ids) * self.val_ratio))
        val_doc_ids = set(doc_ids[:val_count])
        
        train_pool = []
        val_pool = []
        
        for doc_id, samples in doc_groups.items():
            if doc_id in val_doc_ids:
                val_pool.extend(samples)
            else:
                train_pool.extend(samples)
        
        return train_pool, val_pool
    
    def _build_indices(self):
        """빠른 샘플링을 위한 인덱스 구축"""
        
        # 언어별 인덱스
        self.lang_indices = {'ko': [], 'en': []}
        
        # 스타일별 인덱스
        self.style_indices = {'formal': [], 'casual': [], 'none': []}
        
        for idx, sample in enumerate(self.train_pool):
            # 언어별
            self.lang_indices[sample.language].append(idx)
            
            # 스타일별
            if '<|formal|>' in sample.style_tag:
                self.style_indices['formal'].append(idx)
            elif '<|casual|>' in sample.style_tag:
                self.style_indices['casual'].append(idx)
            else:
                self.style_indices['none'].append(idx)
    
    def sample_by_language(self, language: str) -> Sample:
        """특정 언어 샘플 랜덤 선택"""
        indices = self.lang_indices.get(language, [])
        if not indices:
            return random.choice(self.train_pool)
        idx = random.choice(indices)
        return self.train_pool[idx]
    
    def sample_by_style(self, style: str) -> Sample:
        """특정 스타일 샘플 랜덤 선택"""
        indices = self.style_indices.get(style, [])
        if not indices:
            return random.choice(self.train_pool)
        idx = random.choice(indices)
        return self.train_pool[idx]
    
    def sample_balanced(self, style_dist: Dict[str, float] = None) -> Sample:
        """
        언어 1:1, 스타일 분포에 따라 샘플링
        style_dist: {'formal': 0.5, 'casual': 0.3, 'none': 0.2}
        """
        style_dist = style_dist or {'formal': 0.5, 'casual': 0.3, 'none': 0.2}
        
        # 스타일 선택
        style = random.choices(
            list(style_dist.keys()),
            weights=list(style_dist.values()),
            k=1
        )[0]
        
        # 해당 스타일에서 샘플링
        return self.sample_by_style(style)
    
    def get_train_pool(self) -> List[Sample]:
        return self.train_pool
    
    def get_val_pool(self) -> List[Sample]:
        return self.val_pool
    
    def __len__(self):
        return len(self.train_pool)

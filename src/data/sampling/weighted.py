# src/data/sampling/weighted.py
"""
Weighted Sampler - Configurable weights per language/style
"""

import random
from typing import Dict, Any, List
from .base import BaseSampler


class WeightedSampler(BaseSampler):
    """
    Samples with configurable weights per language.
    Useful for upsampling underrepresented languages.
    
    Config:
        ko_weight: float (default 1.0)
        en_weight: float (default 1.0)
        style_weights: dict (optional)
    """
    
    def __init__(self, pool, config: Dict[str, Any]):
        super().__init__(pool, config)
        self.ko_weight = config.get('ko_weight', 1.0)
        self.en_weight = config.get('en_weight', 1.0)
        self.style_weights = config.get('style_weights', {})
        
        self._build_weighted_indices()
    
    def _build_weighted_indices(self):
        """Build weighted sampling indices."""
        self.weighted_indices: List[int] = []
        
        # Add Korean samples with weight
        ko_count = int(len(self.pool.lang_indices.get('ko', [])) * self.ko_weight)
        en_count = int(len(self.pool.lang_indices.get('en', [])) * self.en_weight)
        
        # Repeat indices based on weights
        self.weighted_ko = self.pool.lang_indices.get('ko', []) * max(1, int(self.ko_weight))
        self.weighted_en = self.pool.lang_indices.get('en', []) * max(1, int(self.en_weight))
        
        self.all_indices = self.weighted_ko + self.weighted_en
    
    def sample(self):
        """Sample with weighted probability."""
        if not self.all_indices:
            return random.choice(self.pool.train_pool)
        
        idx = random.choice(self.all_indices)
        return self.pool.train_pool[idx]

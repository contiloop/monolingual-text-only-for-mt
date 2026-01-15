# src/data/sampling/balanced.py
"""
Balanced Language Sampler - Equal sampling from each language
"""

import random
from typing import Dict, Any
from .base import BaseSampler


class BalancedLanguageSampler(BaseSampler):
    """
    Samples equally from Korean and English data pools.
    
    Config:
        ko_ratio: float (default 0.5)
        en_ratio: float (default 0.5)
    """
    
    def __init__(self, pool, config: Dict[str, Any]):
        super().__init__(pool, config)
        self.ko_ratio = config.get('ko_ratio', 0.5)
        self.en_ratio = config.get('en_ratio', 0.5)
        
        # Normalize ratios
        total = self.ko_ratio + self.en_ratio
        self.ko_ratio /= total
        self.en_ratio /= total
    
    def sample(self):
        """Sample with balanced language ratio."""
        # Choose language based on ratio
        if random.random() < self.ko_ratio:
            return self.pool.sample_by_language('ko')
        else:
            return self.pool.sample_by_language('en')

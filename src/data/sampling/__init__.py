# src/data/sampling/__init__.py
"""
Modular Sampling Strategies for Data Loading
"""

from .base import BaseSampler
from .balanced import BalancedLanguageSampler
from .weighted import WeightedSampler

def create_sampler(config: dict, pool) -> BaseSampler:
    """Factory function to create sampler from config."""
    sampling_config = config.get('sampling', {})
    strategy = sampling_config.get('strategy', 'balanced')
    
    if strategy == 'balanced':
        return BalancedLanguageSampler(pool, sampling_config.get('balanced', {}))
    elif strategy == 'weighted':
        return WeightedSampler(pool, sampling_config.get('weighted', {}))
    else:
        # Default to balanced
        return BalancedLanguageSampler(pool, sampling_config.get('balanced', {}))

__all__ = ['BaseSampler', 'BalancedLanguageSampler', 'WeightedSampler', 'create_sampler']

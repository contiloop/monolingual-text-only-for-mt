# src/data/sampling/base.py
"""
Base Sampler Interface
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseSampler(ABC):
    """
    Abstract base class for all sampling strategies.
    
    Samplers control how data is selected from the data pool.
    Different strategies can balance languages, apply curriculum learning, etc.
    """
    
    def __init__(self, pool, config: Dict[str, Any]):
        """
        Args:
            pool: DataPoolManager instance
            config: Strategy-specific configuration
        """
        self.pool = pool
        self.config = config
        self.current_step = 0
    
    @abstractmethod
    def sample(self):
        """
        Sample one item from the data pool.
        
        Returns:
            Sample: A data sample
        """
        pass
    
    def set_step(self, step: int):
        """Update current training step (for curriculum learning)."""
        self.current_step = step
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update configuration dynamically."""
        self.config.update(new_config)

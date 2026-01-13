# project/src/data/loader.py

import json
import random
import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import Dict, List, Optional
from collections import deque
from .noise import NoiseApplier, NoiseConfig

class FinancialDataset(IterableDataset):
    """
    금융 도메인 특화 데이터셋
    - Glossary Aware
    - Noise Injection (L_auto)
    - Buffer Mixing (Normal + Pseudo + Hard)
    """
    
    def __init__(
        self,
        ko_path: str,
        en_path: str,
        noise_config: dict,
        glossary_path: Optional[str] = None,
        config: dict = None,
    ):
        self.ko_data = self._load_data(ko_path)
        self.en_data = self._load_data(en_path)
        self.config = config or {}
        
        # Noise
        self.noise_applier = NoiseApplier(NoiseConfig(**noise_config))
        
        # Glossary
        self.glossary = self._load_json(glossary_path) if glossary_path else None
        
        # Buffers
        self.pseudo_buffer = deque(maxlen=config.get('buffer_size', 5000))
        self.hard_buffer = [] # (sample, loss) list
        
        # State
        self.lback_activated = False
        self.mode = 'train'
        
    def _load_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    def _load_json(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def set_lback_activated(self, active: bool):
        self.lback_activated = active

    def add_pseudo_data(self, data_list: List[dict]):
        self.pseudo_buffer.extend(data_list)
        
    def add_hard_examples(self, examples: List[dict]):
        # examples expected to have 'loss' key
        self.hard_buffer.extend(examples)
        # Sort and keep top
        self.hard_buffer.sort(key=lambda x: x['loss'], reverse=True)
        max_hard = self.config.get('hard_buffer_max', 10000)
        if len(self.hard_buffer) > max_hard:
            self.hard_buffer = self.hard_buffer[:max_hard]

    def __iter__(self):
        while True:
            yield self._generate_item()

    def _generate_item(self):
        # 1. Decide Source Type
        # Ratios from config: Normal 50, Pseudo 30, Hard 20
        ratios = [0.5, 0.3, 0.2]
        source_type = random.choices(['normal', 'pseudo', 'hard'], weights=ratios)[0]
        
        # Fallback if buffers empty
        if source_type == 'pseudo' and not self.pseudo_buffer: source_type = 'normal'
        if source_type == 'hard' and not self.hard_buffer: source_type = 'normal'
        
        item = {}
        
        if source_type == 'normal':
            # Monolingual Data for Denoising (L_auto)
            lang = random.choice(['ko', 'en'])
            data_source = self.ko_data if lang == 'ko' else self.en_data
            sample = random.choice(data_source)
            
            # Apply Noise
            noisy_text, n_type = self.noise_applier.apply(
                sample['text'], lang, 
                style_tag=sample.get('style_tag', '')
            )
            
            # Glossary Injection (for Denoising, we can inject term definitions into prompt to help recover?)
            # Actually Glossary is mostly for Translation. For L_auto, maybe less critical but good for consistency.
            glossary_prompt = self._get_glossary_prompt(sample['text'], lang)
            
            item = {
                'input': f"{glossary_prompt}{noisy_text}", # Input is Noisy
                'target': sample['text'],                  # Target is Clean
                'task': 'denoise',
                'lang': lang
            }
            
        elif source_type == 'pseudo':
            # Back-Translation Data (L_back)
            # Pseudo buffer contains: {source(generated), target(real), direction}
            sample = random.choice(self.pseudo_buffer)
            
            # Apply Glossary to Input
            glossary_prompt = self._get_glossary_prompt(sample['source'], sample['direction'].split('_')[0])
            
            item = {
                'input': f"{glossary_prompt}{sample['source']}",
                'target': sample['generated'], # Wait, pseudo buffer structure?
                # Usually: we generated Source from Target.
                # So Input = Generated Source, Target = Real Target (from monolingual)
                'task': 'translation',
                'lang': sample['direction']
            }
            
        # ... Hard example logic similarly ...
        
        return item

    def _get_glossary_prompt(self, text, lang):
        if not self.glossary: return ""
        # direction logic
        # if input is KO, we need KO->EN glossary? No, we need terms definitions.
        # Actually usually prompts are: "Use these terms: A=B".
        # If input is KO and task is translation to EN, we look up KO terms.
        
        lookup = self.glossary['ko_to_en'] if lang == 'ko' else self.glossary['en_to_ko']
        found = []
        for k, v in lookup.items():
            if k in text:
                found.append(f"{k}={v}")
        
        if found:
            # Limit to 5
            return f"[Terms: {', '.join(found[:5])}] "
        return ""

def create_dataloader(config, ko_path, en_path, glossary_path):
    dataset = FinancialDataset(
        ko_path, en_path, 
        config['noise'], glossary_path, config
    )
    return DataLoader(dataset, batch_size=config['training']['batch_size'])

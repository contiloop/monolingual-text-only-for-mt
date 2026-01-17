# src/data/collator.py
"""
TranslationCollator: 배치 토크나이징 + L_auto/L_back 분리
"""

import torch
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from .composer import ComposedSample, SourceType
from .noise import NoiseApplier, NoiseConfig
from .prompt_builder import PromptBuilder, extract_metadata_from_sample

@dataclass
class CollatedBatch:
    """Collator 출력"""
    # L_auto용
    auto_input_ids: Optional[torch.Tensor] = None
    auto_attention_mask: Optional[torch.Tensor] = None
    auto_labels: Optional[torch.Tensor] = None
    
    # L_back용
    back_input_ids: Optional[torch.Tensor] = None
    back_attention_mask: Optional[torch.Tensor] = None
    back_labels: Optional[torch.Tensor] = None
    
    # 메타데이터
    source_types: List[str] = None
    prompt_lengths: List[int] = None  # Loss Masking용: 프롬프트 부분 길이
    has_auto: bool = False
    has_back: bool = False

class TranslationCollator:
    """
    역할:
    - ComposedSample 리스트를 토크나이징된 배치로 변환
    - L_auto와 L_back 데이터 분리
    - 노이즈 적용 (L_auto용)
    """
    
    def __init__(
        self,
        tokenizer,
        noise_applier: NoiseApplier,
        max_length: int = 2048,
        glossary: Dict = None
    ):
        self.tokenizer = tokenizer
        self.noise_applier = noise_applier
        self.max_length = max_length
        self.glossary = glossary or {}
        
        # Padding 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __call__(self, batch: List[ComposedSample]) -> CollatedBatch:
        """배치 콜레이트"""
        
        # L_auto용과 L_back용 분리
        auto_samples = []  # (noisy_input, original_target)
        back_samples = []  # (source, target)
        source_types = []
        
        for sample in batch:
            source_types.append(sample.source_type.value)
            
            if sample.source_type in [SourceType.NORMAL, SourceType.HARD]:
                # L_auto: 노이즈 적용
                original = sample.text
                style_tag = sample.style_tag or ''
                language = sample.language or 'ko'
                
                # 노이즈 적용
                noisy, noise_type = self.noise_applier.apply(
                    original, language, style_tag
                )
                
                # Glossary 주입 (선택)
                noisy_with_glossary = self._inject_glossary(noisy, language)
                
                auto_samples.append((noisy_with_glossary, original))
                
            elif sample.source_type == SourceType.PSEUDO:
                # L_back: BT 데이터
                source = sample.bt_source
                target = sample.bt_target
                
                if source and target:
                    # Glossary 주입
                    direction = sample.bt_direction or 'ko_to_en'
                    lang = 'ko' if direction.startswith('ko') else 'en'
                    source_with_glossary = self._inject_glossary(source, lang)
                    
                    back_samples.append((source_with_glossary, target))
        
        # 토크나이징
        result = CollatedBatch(source_types=source_types)

        # 디버깅: 첫 번째 배치에서만 샘플 로깅
        if not hasattr(self, '_logged_sample'):
            self._logged_sample = False

        if auto_samples:
            result.has_auto = True
            auto_inputs, auto_targets = zip(*auto_samples)

            # Causal LM: "noisy_input → original_target" 형식으로 연결
            # 포맷: [noisy] [EOS] [target]
            batch_input_ids = []
            batch_labels = []

            for noisy_inp, clean_tgt in zip(auto_inputs, auto_targets):
                # Task tokens
                denoise_token_id = self.tokenizer.convert_tokens_to_ids('[DENOISE]')
                output_token_id = self.tokenizer.convert_tokens_to_ids('[OUTPUT]')

                # Noisy와 Clean 각각 토크나이징
                noisy_tokens = self.tokenizer(noisy_inp, add_special_tokens=False)['input_ids']
                clean_tokens = self.tokenizer(clean_tgt, add_special_tokens=False)['input_ids']

                # Input: [DENOISE] {noisy} [OUTPUT] {clean} <eos>
                input_ids = [denoise_token_id] + noisy_tokens + [output_token_id] + clean_tokens + [self.tokenizer.eos_token_id]

                # Labels: [-100...] (prefix 마스킹) + {clean} + <eos>
                prefix_length = len([denoise_token_id] + noisy_tokens + [output_token_id])
                labels = [-100] * prefix_length + clean_tokens + [self.tokenizer.eos_token_id]

                batch_input_ids.append(input_ids)
                batch_labels.append(labels)

            # Padding
            max_len = min(max(len(ids) for ids in batch_input_ids), self.max_length)
            padded_input_ids = []
            padded_labels = []
            padded_attention_mask = []

            for input_ids, labels in zip(batch_input_ids, batch_labels):
                # Truncate
                input_ids = input_ids[:max_len]
                labels = labels[:max_len]

                # Pad
                pad_len = max_len - len(input_ids)
                padded_input_ids.append(input_ids + [self.tokenizer.pad_token_id] * pad_len)
                padded_labels.append(labels + [-100] * pad_len)
                padded_attention_mask.append([1] * len(input_ids) + [0] * pad_len)

            result.auto_input_ids = torch.tensor(padded_input_ids)
            result.auto_attention_mask = torch.tensor(padded_attention_mask)
            result.auto_labels = torch.tensor(padded_labels)

            # 첫 번째 배치 샘플 로깅
            if not self._logged_sample and len(auto_inputs) > 0:
                print("\n" + "="*80)
                print("📊 [DENOISING SAMPLE - L_auto]")
                print("="*80)
                print(f"Noisy Input:  {auto_inputs[0][:200]}...")
                print(f"Clean Target: {auto_targets[0][:200]}...")
                print(f"\n🔢 Token lengths:")
                print(f"  Noisy tokens: {len(batch_input_ids[0][:len(noisy_tokens)])} tokens")
                print(f"  Clean tokens: {len(batch_labels[0]) - batch_labels[0].count(-100)} tokens")
                print(f"\n🎯 Labels masking:")
                print(f"  Input IDs:  {batch_input_ids[0][:50]}...")
                print(f"  Labels:     {batch_labels[0][:50]}...")
                print(f"  (-100 = masked, only clean text contributes to loss)")
                print("="*80 + "\n")
                self._logged_sample = True
        
        if back_samples:
            result.has_back = True
            back_inputs, back_targets = zip(*back_samples)

            # Back-translation도 동일하게 labels 마스킹
            batch_input_ids = []
            batch_labels = []

            for source_inp, target_tgt in zip(back_inputs, back_targets):
                # Task tokens (back-translation uses same format)
                denoise_token_id = self.tokenizer.convert_tokens_to_ids('[DENOISE]')
                output_token_id = self.tokenizer.convert_tokens_to_ids('[OUTPUT]')

                # Source와 Target 각각 토크나이징
                source_tokens = self.tokenizer(source_inp, add_special_tokens=False)['input_ids']
                target_tokens = self.tokenizer(target_tgt, add_special_tokens=False)['input_ids']

                # Input: [DENOISE] {source} [OUTPUT] {target} <eos>
                input_ids = [denoise_token_id] + source_tokens + [output_token_id] + target_tokens + [self.tokenizer.eos_token_id]

                # Labels: [-100...] (prefix 마스킹) + {target} + <eos>
                prefix_length = len([denoise_token_id] + source_tokens + [output_token_id])
                labels = [-100] * prefix_length + target_tokens + [self.tokenizer.eos_token_id]

                batch_input_ids.append(input_ids)
                batch_labels.append(labels)

            # Padding
            max_len = min(max(len(ids) for ids in batch_input_ids), self.max_length)
            padded_input_ids = []
            padded_labels = []
            padded_attention_mask = []

            for input_ids, labels in zip(batch_input_ids, batch_labels):
                input_ids = input_ids[:max_len]
                labels = labels[:max_len]

                pad_len = max_len - len(input_ids)
                padded_input_ids.append(input_ids + [self.tokenizer.pad_token_id] * pad_len)
                padded_labels.append(labels + [-100] * pad_len)
                padded_attention_mask.append([1] * len(input_ids) + [0] * pad_len)

            result.back_input_ids = torch.tensor(padded_input_ids)
            result.back_attention_mask = torch.tensor(padded_attention_mask)
            result.back_labels = torch.tensor(padded_labels)

            # 첫 번째 배치 샘플 로깅 (back-translation)
            if not self._logged_sample and len(back_inputs) > 0:
                print("\n" + "="*80)
                print("📊 [BACK-TRANSLATION SAMPLE - L_back]")
                print("="*80)
                print(f"Source Input: {back_inputs[0][:200]}...")
                print(f"Target:       {back_targets[0][:200]}...")
                print(f"\n🔢 Token lengths:")
                print(f"  Source tokens: {len(batch_input_ids[0][:len(source_tokens)])} tokens")
                print(f"  Target tokens: {len(batch_labels[0]) - batch_labels[0].count(-100)} tokens")
                print(f"\n🎯 Labels masking:")
                print(f"  Input IDs:  {batch_input_ids[0][:50]}...")
                print(f"  Labels:     {batch_labels[0][:50]}...")
                print(f"  (-100 = masked, only target text contributes to loss)")
                print("="*80 + "\n")
                self._logged_sample = True
        
        return result
    
    def _inject_glossary(self, text: str, language: str) -> str:
        """텍스트에 용어집 주입"""
        if not self.glossary:
            return text
        
        # 방향 결정
        lookup = self.glossary.get(
            'ko_to_en' if language == 'ko' else 'en_to_ko', 
            {}
        )
        
        found_terms = []
        for term, translation in lookup.items():
            if term.lower() in text.lower():
                found_terms.append(f"{term}={translation}")
        
        if found_terms:
            # 최대 5개 용어
            terms_str = ', '.join(found_terms[:5])
            return f"[Terms: {terms_str}] {text}"
        
        return text
    
    def update_noise_config(self, progress: float, curriculum: Dict):
        """노이즈 커리큘럼 업데이트"""
        self.noise_applier.update_config(progress, curriculum)

# project/src/data/noise.py

import random
import re
from typing import List, Set, Tuple
from dataclasses import dataclass

@dataclass
class NoiseConfig:
    """노이즈 설정"""
    total_ratio: float = 0.1
    deletion_prob: float = 0.30
    filler_prob: float = 0.25
    infilling_prob: float = 0.35
    shuffling_prob: float = 0.10
    asr_error_prob: float = 0.05  # ASR 노이즈 확률 추가

class NoiseApplier:
    """노이즈 적용기 (Text + ASR Noise)"""
    
    FILLERS_KO = ['어', '음', '그', '저', '뭐', '그러니까', '이제']
    FILLERS_EN = ['uh', 'um', 'well', 'you know', 'I mean', 'so']
    
    # ASR Error Maps (Phonetic similarity)
    ASR_ERRORS_EN = {
        'sell': ['cell'], 'buy': ['by', 'bye'], 'their': ['there', "they're"],
        'to': ['too', 'two'], 'than': ['then'], 'accept': ['except'],
        'effect': ['affect'], 'meet': ['meat'], 'won': ['one']
    }
    ASR_ERRORS_KO = {
        '매출': ['매축', '매쭐'], '성장': ['성짱', '썽장'],
        '투자': ['두자'], '영업': ['영업'], # 예시
    }
    
    MASK_TOKEN = '[MASK]'
    
    def __init__(self, config: NoiseConfig):
        self.config = config
    
    def apply(
        self,
        text: str,
        language: str, # "ko" or "en"
        style_tag: str = "",
        protected_indices: Set[int] = None
    ) -> Tuple[str, str]:
        """
        텍스트에 노이즈 적용
        Returns: (noisy_text, noise_type)
        """
        
        # 1. 스타일 태그 분리 (태그는 노이즈 안 줌)
        content = text
        if style_tag and text.startswith(style_tag):
            content = text[len(style_tag):].strip()
        
        tokens = content.split()
        
        # 2. 보호 인덱스 계산 (숫자, 부정어)
        if protected_indices is None:
            protected_indices = self._get_protected_indices(tokens, language)
        
        # 3. ASR 노이즈 적용 (선택적)
        if random.random() < self.config.asr_error_prob:
            tokens = self._apply_asr_noise(tokens, language, protected_indices)
        
        # 4. 일반 Noising (Deletion, Infilling, etc)
        candidates = [i for i in range(len(tokens)) if i not in protected_indices]
        num_noise = max(1, int(len(candidates) * self.config.total_ratio))
        
        noise_type = "none"
        if candidates:
            noise_type = self._select_noise_type()
            tokens = self._apply_general_noise(tokens, candidates, num_noise, noise_type, language)
            
        # 5. 재조립
        noisy_content = " ".join(tokens)
        final_text = f"{style_tag} {noisy_content}" if style_tag else noisy_content
        
        return final_text.strip(), noise_type
    
    def _get_protected_indices(self, tokens: List[str], language: str) -> Set[int]:
        """보호할 토큰 인덱스 찾기 (숫자 window 포함)"""
        protected = set()
        
        negations = ['않', '못', '안', '없', '아니', '절대'] if language == 'ko' else ["not", "n't", "never", "no", "none"]
        
        for i, token in enumerate(tokens):
            # 숫자 보호 (앞뒤 1토큰 포함)
            if re.search(r'\d', token):
                # Window 1
                for j in range(max(0, i-1), min(len(tokens), i+2)):
                    protected.add(j)
            
            # 부정어 보호
            if any(neg in token for neg in negations):
                protected.add(i)
                
        return protected

    def _apply_asr_noise(self, tokens: List[str], language: str, protected: Set[int]) -> List[str]:
        """ASR 에러 시뮬레이션"""
        errors = self.ASR_ERRORS_EN if language == 'en' else self.ASR_ERRORS_KO
        result = tokens.copy()
        
        for i, token in enumerate(tokens):
            if i in protected: continue
            
            token_lower = token.lower()
            if token_lower in errors:
                result[i] = random.choice(errors[token_lower])
                # 대소문자 복원 (간단히 첫글자만)
                if token[0].isupper():
                    result[i] = result[i].capitalize()
        return result

    def _select_noise_type(self) -> str:
        types = ["deletion", "filler", "infilling", "shuffling"]
        probs = [self.config.deletion_prob, self.config.filler_prob, self.config.infilling_prob, self.config.shuffling_prob]
        return random.choices(types, weights=probs, k=1)[0]
        
    def _apply_general_noise(self, tokens, candidates, num, n_type, lang):
        if n_type == "deletion":
            del_idc = set(random.sample(candidates, min(num, len(candidates))))
            return [t for i, t in enumerate(tokens) if i not in del_idc]
        elif n_type == "filler":
            fillers = self.FILLERS_KO if lang == 'ko' else self.FILLERS_EN
            pos_list = sorted(random.sample(candidates, min(num, len(candidates))), reverse=True)
            res = tokens.copy()
            for p in pos_list:
                res.insert(p, random.choice(fillers))
            return res
        elif n_type == "infilling":
            if len(candidates) < 2: return tokens
            start = random.choice(candidates[:-1])
            length = min(random.randint(1, 3), len(candidates)) # span 1~3
            res = tokens.copy()
            res[start:start+length] = [self.MASK_TOKEN]
            return res
        elif n_type == "shuffling":
            # 문장 셔플링은 전처리 단계에서 하는게 나을수도 있으나 여기서 간단히 구현
            # 문장 종결자 기준 스플릿은 복잡하므로 여기선 토큰 단위 셔플은 지양하고 패스 (혹은 n-gram 셔플)
             return tokens # Placeholder
        return tokens

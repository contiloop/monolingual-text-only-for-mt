# project/src/data/noise.py

import random
import re
from typing import List, Set, Tuple, Dict
from dataclasses import dataclass

@dataclass
class NoiseConfig:
    """노이즈 설정"""
    total_ratio: float = 0.15  # 노이즈 비율 증가

    # 기본 노이즈 타입 확률
    deletion_prob: float = 0.15
    filler_prob: float = 0.15
    infilling_prob: float = 0.20
    repetition_prob: float = 0.10  # 단어 반복
    spacing_prob: float = 0.15     # 띄어쓰기 오류
    punctuation_prob: float = 0.10 # 구두점 오류
    newline_prob: float = 0.05     # 줄바꿈 오류
    typo_prob: float = 0.10        # 오타

    # 추가 노이즈
    asr_error_prob: float = 0.05   # ASR 오류

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
        types = ["deletion", "filler", "infilling", "repetition", "spacing", "punctuation", "newline", "typo"]
        probs = [
            self.config.deletion_prob,
            self.config.filler_prob,
            self.config.infilling_prob,
            self.config.repetition_prob,
            self.config.spacing_prob,
            self.config.punctuation_prob,
            self.config.newline_prob,
            self.config.typo_prob
        ]
        return random.choices(types, weights=probs, k=1)[0]

    def _apply_general_noise(self, tokens, candidates, num, n_type, lang):
        if n_type == "deletion":
            # 단어 삭제
            del_idc = set(random.sample(candidates, min(num, len(candidates))))
            return [t for i, t in enumerate(tokens) if i not in del_idc]

        elif n_type == "filler":
            # Filler words 추가 (um, uh, 어, 음 등)
            fillers = self.FILLERS_KO if lang == 'ko' else self.FILLERS_EN
            pos_list = sorted(random.sample(candidates, min(num, len(candidates))), reverse=True)
            res = tokens.copy()
            for p in pos_list:
                res.insert(p, random.choice(fillers))
            return res

        elif n_type == "infilling":
            # [MASK] 토큰으로 대체
            if len(candidates) < 2: return tokens
            start = random.choice(candidates[:-1])
            length = min(random.randint(1, 3), len(candidates))
            res = tokens.copy()
            res[start:start+length] = [self.MASK_TOKEN]
            return res

        elif n_type == "repetition":
            # 단어 반복 (like like, 그 그 등)
            pos_list = random.sample(candidates, min(num, len(candidates)))
            res = tokens.copy()
            for p in sorted(pos_list, reverse=True):
                res.insert(p + 1, res[p])  # 다음 위치에 같은 단어 삽입
            return res

        elif n_type == "spacing":
            # 띄어쓰기 오류
            return self._apply_spacing_noise(tokens, candidates, num)

        elif n_type == "punctuation":
            # 구두점 오류 (추가/삭제/중복)
            return self._apply_punctuation_noise(tokens, num)

        elif n_type == "newline":
            # 줄바꿈 추가 (문장 중간에)
            if len(candidates) < 2: return tokens
            pos_list = random.sample(candidates, min(2, len(candidates)))
            res = tokens.copy()
            for p in sorted(pos_list, reverse=True):
                res.insert(p, "\n")
            return res

        elif n_type == "typo":
            # 오타 (키보드 인접 문자, 누락, 중복)
            return self._apply_typo_noise(tokens, candidates, num, lang)

        return tokens

    def _apply_spacing_noise(self, tokens, candidates, num):
        """띄어쓰기 오류"""
        if len(tokens) < 2: return tokens
        pos_list = random.sample(candidates, min(num, len(candidates)))
        res = []

        for i, token in enumerate(tokens):
            if i in pos_list and i + 1 < len(tokens):
                # 다음 단어와 붙이기
                res.append(token + tokens[i + 1])
                tokens[i + 1] = ""  # 다음 토큰 비우기
            elif token:  # 빈 문자열 제외
                res.append(token)

        return res

    def _apply_punctuation_noise(self, tokens, num):
        """구두점 오류"""
        punctuations = [',', '.', '!', '?', ';', ':', '...', ',,', '..']
        res = tokens.copy()

        for _ in range(min(num, len(tokens))):
            action = random.choice(['add', 'duplicate', 'wrong'])
            pos = random.randint(0, len(res) - 1)

            if action == 'add':
                # 랜덤 위치에 구두점 추가
                res.insert(pos, random.choice(punctuations))
            elif action == 'duplicate' and pos < len(res):
                # 마침표 중복 (.. 또는 ...)
                if res[pos] in ['.', ',', '!', '?']:
                    res[pos] = res[pos] * random.randint(2, 3)
            elif action == 'wrong' and pos < len(res):
                # 잘못된 구두점 사용
                if res[pos] in ['.', ',', '!', '?']:
                    res[pos] = random.choice(punctuations)

        return res

    def _apply_typo_noise(self, tokens, candidates, num, lang):
        """오타 생성"""
        # 키보드 인접 문자 맵 (QWERTY)
        keyboard_neighbors = {
            'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'erfcxs',
            'e': 'wrdsf', 'f': 'rtgvcd', 'g': 'tyhbvf', 'h': 'yujnbg',
            'i': 'uojkl', 'j': 'uikmnh', 'k': 'iolmj', 'l': 'opk',
            'm': 'njk', 'n': 'bhjm', 'o': 'iplk', 'p': 'ol',
            'q': 'wa', 'r': 'etfd', 's': 'wedxza', 't': 'ryfg',
            'u': 'yihj', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc',
            'y': 'tugh', 'z': 'asx'
        }

        res = tokens.copy()
        pos_list = random.sample(candidates, min(num, len(candidates)))

        for pos in pos_list:
            if pos >= len(res) or len(res[pos]) < 2:
                continue

            word = res[pos]
            typo_type = random.choice(['substitute', 'duplicate', 'omit'])

            if typo_type == 'substitute':
                # 인접 키로 대체
                char_pos = random.randint(0, len(word) - 1)
                char = word[char_pos].lower()
                if char in keyboard_neighbors:
                    new_char = random.choice(keyboard_neighbors[char])
                    word = word[:char_pos] + new_char + word[char_pos + 1:]

            elif typo_type == 'duplicate':
                # 문자 중복 (typo -> typpo)
                char_pos = random.randint(0, len(word) - 1)
                word = word[:char_pos] + word[char_pos] + word[char_pos:]

            elif typo_type == 'omit':
                # 문자 누락 (word -> wrd)
                if len(word) > 2:
                    char_pos = random.randint(0, len(word) - 1)
                    word = word[:char_pos] + word[char_pos + 1:]

            res[pos] = word

        return res
    
    def update_config(self, progress: float, curriculum: Dict = None):
        """커리큘럼에 따라 노이즈 설정 업데이트"""
        if curriculum is None:
            return
        
        # 진행률에 따라 노이즈 비율 조정
        # 예: 학습 초기에는 낮은 노이즈, 후반에는 높은 노이즈
        if 'total_ratio' in curriculum:
            self.config.total_ratio = curriculum['total_ratio']
        
        # 커리큘럼 스케줄 적용
        for schedule in curriculum.get('schedule', []):
            if progress >= schedule.get('progress', 0):
                if 'total_ratio' in schedule:
                    self.config.total_ratio = schedule['total_ratio']

# src/data/prompt_builder.py
"""
Hybrid Prompt Builder
- 스타일 태그: <|formal|>, <|casual|>
- 메타데이터 프롬프트: [Context: ...], [Terms: ...]
- 구분자: \n\n (프롬프트와 본문 분리)
- Prompt Dropout: 학습 시 랜덤하게 요소 생략
"""

import random
from typing import Dict, Optional, Tuple

# 도메인-스타일 충돌 방지 규칙
STRICT_FORMAL_DOMAINS = {'legal', 'contract', 'bank', 'regulation', 'official'}
STRICT_CASUAL_DOMAINS = {'chat', 'sns', 'comment'}

class PromptBuilder:
    """
    하이브리드 프롬프트 빌더
    
    출력 형식:
    [Context: Q1 2024 Earnings, Company: AAPL]
    [Terms: Revenue=매출, YoY=전년비]
    <|casual|>

    {본문}
    """
    
    def __init__(self, config: Dict = None):
        config = config or {}
        
        # Prompt Dropout 확률 (학습 시 robustness 위해)
        self.dropout = {
            'context': config.get('context_dropout', 0.20),     # 20% 확률로 Context 생략
            'terms': config.get('terms_dropout', 0.30),         # 30% 확률로 Terms 생략
            'style_tag': config.get('style_tag_dropout', 0.35), # 35% 확률로 태그 생략
        }
        
        self.max_terms = config.get('max_terms', 5)
        self.separator = "\n\n"  # 메타데이터와 본문 사이 구분자
    
    def build(
        self,
        text: str,
        style: str = 'auto',
        context: Dict = None,
        instruction: str = None,  # glossary_terms 대신 instruction
        training_mode: bool = True
    ) -> Tuple[str, Dict]:
        """
        프롬프트 빌드
        
        Args:
            text: 원본 텍스트
            style: 'formal', 'casual', 'auto'
            context: {'event': '...', 'company': '...', 'speaker': '...'}
            instruction: Soft instruction (e.g., "금융 용어는 표준 번역 사용...")
            training_mode: True면 Prompt Dropout 적용
        
        Returns:
            (final_prompt, metadata_for_logging)
        """
        parts = []
        meta_log = {'context_included': False, 'instruction_included': False, 'style_tag': None}
        
        # === 1. Context Block ===
        if context and not self._should_dropout('context', training_mode):
            ctx_parts = []
            
            if context.get('event'):
                ctx_parts.append(context['event'])
            if context.get('company'):
                ctx_parts.append(f"Company: {context['company']}")
            if context.get('speaker'):
                ctx_parts.append(f"Speaker: {context['speaker']}")
            if context.get('date'):
                ctx_parts.append(f"Date: {context['date']}")
            
            if ctx_parts:
                ctx_str = f"[Context: {', '.join(ctx_parts)}]"
                parts.append(ctx_str)
                meta_log['context_included'] = True
        
        # === 2. Soft Instruction ===
        if instruction and not self._should_dropout('terms', training_mode):
            parts.append(f"[Instruction: {instruction}]")
            meta_log['instruction_included'] = True
        
        # === 3. Style Tag ===
        resolved_style = self._resolve_style_conflict(style, context)
        
        if not self._should_dropout('style_tag', training_mode):
            if resolved_style == 'formal':
                parts.append('<|formal|>')
                meta_log['style_tag'] = 'formal'
            elif resolved_style == 'casual':
                parts.append('<|casual|>')
                meta_log['style_tag'] = 'casual'
        
        # === 4. 조립 ===
        header = " ".join(parts)
        
        if header:
            final_prompt = f"{header}{self.separator}{text}"
        else:
            final_prompt = text
        
        return final_prompt, meta_log
    
    def _should_dropout(self, element: str, training_mode: bool) -> bool:
        """Prompt Dropout 결정"""
        if not training_mode:
            return False
        return random.random() < self.dropout.get(element, 0)
    
    def _resolve_style_conflict(self, style: str, context: Dict) -> str:
        """
        도메인-스타일 충돌 해결
        예: Legal 도메인인데 casual 태그면 → formal로 강제
        """
        if not context:
            return style
        
        event = (context.get('event') or '').lower()
        source = (context.get('source') or '').lower()
        domain_hint = event + ' ' + source
        
        # Strict Formal 도메인 체크
        for keyword in STRICT_FORMAL_DOMAINS:
            if keyword in domain_hint:
                return 'formal'  # 강제 격식체
        
        # Strict Casual 도메인 체크
        for keyword in STRICT_CASUAL_DOMAINS:
            if keyword in domain_hint:
                return 'casual'  # 강제 비격식체
        
        return style


# === 유틸 함수 ===

def extract_metadata_from_sample(sample: Dict) -> Dict:
    """
    전처리된 샘플에서 메타데이터 추출
    """
    metadata = sample.get('metadata', {})
    source = metadata.get('source', '')
    
    context = {}
    
    # 소스별 메타데이터 매핑
    if source == 'sp500_earnings':
        context['event'] = f"{metadata.get('year', '')} {metadata.get('quarter', '')} Earnings Call"
        context['company'] = metadata.get('security') or metadata.get('symbol')
        context['sector'] = metadata.get('sector')
    
    elif source == 'earnings_qa':
        context['event'] = 'Earnings Q&A'
        context['company'] = metadata.get('ticker')
    
    elif source in ['hk', 'mk', 'naver']:
        context['event'] = f"{source.upper()} 기사"
        context['date'] = metadata.get('date')
    
    elif source == 'korea_bank':
        context['event'] = '한국은행 용어사전'
        categories = metadata.get('categories', [])
        if categories:
            context['category'] = ', '.join(categories)
    
    elif source == 'reuter':
        context['event'] = 'Reuters 기사'
        context['date'] = metadata.get('date')
    
    return context

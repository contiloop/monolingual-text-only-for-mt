# project/src/data/parsers.py
"""
다양한 데이터 소스 파서 - 풍부한 메타데이터 추출
"""

import json
from pathlib import Path
from typing import Iterator, Dict, Any

class BaseParser:
    def parse(self, path: str) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError

class StandardJsonlParser(BaseParser):
    """뉴스 JSONL (hk, mk, naver, reuter)"""
    
    def parse(self, path: str) -> Iterator[Dict[str, Any]]:
        source_name = Path(path).stem  # hk, mk, naver, reuter
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get('content') or data.get('text')
                    if not text or len(text.strip()) < 50:
                        continue
                    
                    yield {
                        'text': text.strip(),
                        'language': 'ko' if source_name in ['hk', 'mk', 'naver'] else 'en',
                        'style_tag': '<|formal|>',  # 뉴스는 격식체
                        'metadata': {
                            'source': source_name,
                            'event': f'{source_name.upper()} 기사',
                            'title': data.get('title'),
                            'date': data.get('date'),
                            'author': data.get('author'),
                            'category': data.get('category'),
                            'url': data.get('url')
                        }
                    }
                except json.JSONDecodeError:
                    continue

class BankDictionaryParser(BaseParser):
    """한국은행 용어사전"""
    
    def parse(self, path: str) -> Iterator[Dict[str, Any]]:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    term = data.get('term', '')
                    desc = data.get('text', '')
                    
                    if not desc or len(desc.strip()) < 10:
                        continue
                    
                    full_text = f"용어: {term}\n정의: {desc}"
                    categories = data.get('categories', [])
                    
                    yield {
                        'text': full_text,
                        'language': 'ko',
                        'style_tag': '<|formal|>',  # 공식 용어 → 격식체
                        'metadata': {
                            'source': 'korea_bank',
                            'event': '한국은행 용어사전',
                            'term': term,
                            'category': ', '.join(categories) if categories else None
                        }
                    }
                except json.JSONDecodeError:
                    continue

class SP500EarningsParser(BaseParser):
    """SP500 Earnings Calls - 분기별 transcript"""
    
    EARNINGS_COLUMNS = [
        '2022_earnings_q1', '2022_earnings_q2', '2022_earnings_q3', '2022_earnings_q4',
        '2023_earnings_q1', '2023_earnings_q2', '2023_earnings_q3', '2023_earnings_q4',
        '2024_earnings_q1', '2024_earnings_q2', '2024_earnings_q3', '2024_earnings_q4',
        '2025_earnings_q1'
    ]
    
    def parse(self, path: str) -> Iterator[Dict[str, Any]]:
        try:
            from datasets import load_from_disk
            
            ds = load_from_disk(path)
            if 'train' in ds:
                ds = ds['train']
            
            for row in ds:
                symbol = row.get('Symbol', 'UNKNOWN')
                security = row.get('Security', '')
                sector = row.get('GICS Sector', '')
                sub_industry = row.get('GICS Sub-Industry', '')
                
                for col in self.EARNINGS_COLUMNS:
                    text = row.get(col)
                    if not text or not isinstance(text, str) or len(text.strip()) < 100:
                        continue
                    
                    # 분기 정보 파싱: "2024_earnings_q1" → year=2024, quarter=Q1
                    parts = col.split('_')
                    year = parts[0]
                    quarter = parts[2].upper()
                    
                    yield {
                        'text': text.strip(),
                        'language': 'en',
                        'style_tag': '<|casual|>',  # 컨퍼런스 콜 → 구어체
                        'metadata': {
                            'source': 'sp500_earnings',
                            'event': f'{quarter} {year} Earnings Call',
                            'company': security or symbol,
                            'ticker': symbol,
                            'sector': sector,
                            'sub_industry': sub_industry,
                            'year': year,
                            'quarter': quarter
                        }
                    }
        except Exception as e:
            print(f"SP500 파싱 에러: {e}")

class EarningsQAParser(BaseParser):
    """Earnings Calls Q&A"""
    
    def parse(self, path: str) -> Iterator[Dict[str, Any]]:
        try:
            from datasets import load_from_disk
            
            ds = load_from_disk(path)
            if 'train' in ds:
                ds = ds['train']
            
            for row in ds:
                question = row.get('question', '')
                answer = row.get('answer', '')
                ticker = row.get('ticker', '')
                date = row.get('date', '')
                
                # Q&A 결합
                if question and answer:
                    text = f"Q: {question}\nA: {answer}"
                elif question:
                    text = question
                elif answer:
                    text = answer
                else:
                    continue
                
                if len(text.strip()) < 50:
                    continue
                
                yield {
                    'text': text.strip(),
                    'language': 'en',
                    'style_tag': '<|casual|>',  # Q&A → 구어체
                    'metadata': {
                        'source': 'earnings_qa',
                        'event': 'Earnings Q&A',
                        'company': ticker,
                        'ticker': ticker,
                        'date': date
                    }
                }
        except Exception as e:
            print(f"EarningsQA 파싱 에러: {e}")


def get_parser(path: str) -> BaseParser:
    """경로에 맞는 파서 반환"""
    name = Path(path).name
    
    registry = {
        'hk.jsonl': StandardJsonlParser(),
        'mk.jsonl': StandardJsonlParser(),
        'naver.jsonl': StandardJsonlParser(),
        'reuter.jsonl': StandardJsonlParser(),
        'korea-bank-700-cleaned.jsonl': BankDictionaryParser(),
    }
    
    if name in registry:
        return registry[name]
    
    if 'sp500' in path.lower():
        return SP500EarningsParser()
    if 'earnings' in path.lower() and 'qa' in path.lower():
        return EarningsQAParser()
    
    return StandardJsonlParser()

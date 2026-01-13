# scripts/preprocess_all.py
"""
전처리 스크립트: Raw 데이터 → Processed JSONL
"""

import sys
from pathlib import Path

# 프로젝트 루트 (scripts의 부모 = 루트)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
from tqdm import tqdm
from src.data.parsers import get_parser

# ============================================
# 설정
# ============================================

# Raw 데이터는 외부 경로 (환경에 맞게 수정)
RAW_DATA_DIR = Path.home() / "Desktop" / "raw_financial_data"  # 또는 환경변수

RAW_DATA_MAP = {
    "ko": [
        RAW_DATA_DIR / "hk.jsonl",
        RAW_DATA_DIR / "mk.jsonl",
        RAW_DATA_DIR / "naver.jsonl",
        RAW_DATA_DIR / "korea-bank-700-cleaned.jsonl",
    ],
    "en": [
        RAW_DATA_DIR / "reuter.jsonl",
        RAW_DATA_DIR / "sp500_earnings_calls_dataset",
        RAW_DATA_DIR / "earnings_calls_qa_dataset",
    ]
}

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# ============================================
# 메인 로직
# ============================================

def process_source(path: Path, output_file: Path) -> int:
    """단일 소스 처리 - 파서가 모든 필드를 채워줌"""
    if not path.exists():
        print(f"  ⚠️ 경로 없음: {path}")
        return 0
    
    parser = get_parser(str(path))
    print(f"  📂 {path.name} → {type(parser).__name__}")
    
    count = 0
    with open(output_file, 'a', encoding='utf-8') as f:
        for item in tqdm(parser.parse(str(path)), desc=f"    {path.name}", leave=False):
            # 파서가 이미 language, style_tag, metadata를 채워줌
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            count += 1
    
    return count

def main():
    print("=" * 60)
    print("📊 금융 번역 모델 데이터 전처리 (v2 - 풍부한 메타데이터)")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 한국어 처리
    print("\n🇰🇷 한국어 데이터 처리")
    ko_output = OUTPUT_DIR / "ko_processed.jsonl"
    if ko_output.exists():
        ko_output.unlink()
    
    ko_total = 0
    for path in RAW_DATA_MAP["ko"]:
        ko_total += process_source(path, ko_output)
    print(f"  ✅ 한국어 총 {ko_total:,} 샘플")
    
    # 영어 처리
    print("\n🇺🇸 영어 데이터 처리")
    en_output = OUTPUT_DIR / "en_processed.jsonl"
    if en_output.exists():
        en_output.unlink()
    
    en_total = 0
    for path in RAW_DATA_MAP["en"]:
        en_total += process_source(path, en_output)
    print(f"  ✅ 영어 총 {en_total:,} 샘플")
    
    # 샘플 출력
    print("\n" + "=" * 60)
    print("� 샘플 확인:")
    with open(ko_output, 'r') as f:
        sample = json.loads(f.readline())
        print(f"  KO: {json.dumps(sample, ensure_ascii=False, indent=2)[:500]}...")
    
    with open(en_output, 'r') as f:
        sample = json.loads(f.readline())
        print(f"  EN: {json.dumps(sample, ensure_ascii=False, indent=2)[:500]}...")
    
    print("\n" + "=" * 60)
    print(f"📁 출력: {OUTPUT_DIR}")
    print(f"📊 총 샘플: {ko_total + en_total:,}")
    print("=" * 60)

if __name__ == "__main__":
    main()

# 금융 번역 모델 (MidM Financial Translation)

## 빠른 시작

```bash
# 1. 클론 & 환경 설정
git clone https://github.com/contiloop/monolingual-text-only-for-mt.git
cd monolingual-text-only-for-mt
uv venv .venv && source .venv/bin/activate
uv pip install -e .

# 2. WandB 로그인
wandb login

# 3. 학습 실행
python src/train.py --config configs/base.yaml
```

---

## 데이터 준비

### 1. 학습 데이터 (Monolingual)
```
data/processed/
├── ko_processed.jsonl   # 한국어 (earnings calls, 금융 뉴스)
└── en_processed.jsonl   # 영어 (earnings calls, 금융 뉴스)
```

**로컬에서 압축:**
```bash
cd /path/to/local/data/processed
tar -czvf processed_data.tar.gz *.jsonl
```

**서버에서 압축 해제:**
```bash
cd /workspace/monolingual-text-only-for-mt/data/processed
tar -xzvf processed_data.tar.gz
```

### 2. 평가 데이터 (Parallel Corpus)
```
data/eval/
└── korean_english_parallel/   # Wiki 한영 병렬 코퍼스
    └── train/
        └── *.arrow            # HuggingFace datasets 포맷
```

**소스:** `lemon-mint/korean_english_parallel_wiki_augmented_v1`

**용도:**
- 번역 품질 평가 (BLEU)
- Cycle consistency 테스트 (ko→en→ko)
- 학습에는 **사용 안 함** (Unsupervised learning)

---

## GPU별 설정

```bash
# A100 40-44GB (기본값 - 4bit QLoRA)
python src/train.py --config configs/base.yaml

# A100 80GB (8bit)
python src/train.py --config configs/gpu_8bit.yaml

# H100 / Multi-GPU (Full bf16)
python src/train.py --config configs/gpu_full.yaml
```

---

## 프로젝트 구조
```
.
├── configs/              # 학습 설정
│   ├── base.yaml         # 기본 (4-bit QLoRA)
│   ├── gpu_8bit.yaml     # 8-bit
│   └── gpu_full.yaml     # Full precision
├── data/
│   ├── processed/        # 전처리된 학습 데이터
│   └── eval/             # 평가용 병렬 코퍼스
├── src/                  # 학습 코드
└── outputs/              # 체크포인트 저장
```

---

## 트러블슈팅

### CUDA OOM
```yaml
# configs/base.yaml에서 조정
training:
  batch_size: 1
model:
  max_seq_length: 1024
```

### ModuleNotFoundError
```bash
uv pip install -e .
```

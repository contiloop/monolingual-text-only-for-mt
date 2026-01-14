# 금융 번역 모델 (MidM Financial Translation)

## 환경 설정

### 1. 프로젝트 클론
```bash
git clone https://github.com/contiloop/monolingual-text-only-for-mt.git
cd monolingual-text-only-for-mt
```

### 2. 의존성 설치
```bash
# pyproject.toml 사용
pip install -e .

# 또는 개별 설치
pip install torch transformers peft accelerate huggingface_hub wandb datasets tqdm pyyaml
```

### 3. 데이터 준비
```bash
mkdir -p data/processed
# ko_processed.jsonl, en_processed.jsonl을 data/processed/에 복사
```

### 4. WandB 로그인
```bash
wandb login
# 프롬프트에 API 키 입력
```

또는 환경변수:
```bash
export WANDB_API_KEY="your-api-key"
```

## 학습 실행

```bash
# 단일 GPU
python src/train.py --config configs/base.yaml

# Multi-GPU
torchrun --nproc_per_node=4 src/train.py --config configs/base.yaml

# WandB 없이 테스트
WANDB_MODE=disabled python src/train.py --config configs/base.yaml
```

## 프로젝트 구조
```
.
├── configs/          # 학습 설정
├── data/processed/   # 전처리 데이터
├── src/              # 학습 코드
│   ├── data/         # DataLoader
│   └── train.py      # 메인 학습
└── pyproject.toml    # 의존성
```

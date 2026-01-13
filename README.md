# 금융 번역 모델 (MidM Financial Translation)

## 환경 설정 (uv 가상환경 필수)

### 1. uv 설치
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### 2. 프로젝트 설정
```bash
git clone https://github.com/contiloop/monolingual-text-only-for-mt.git
cd monolingual-text-only-for-mt

# 가상환경 생성 및 활성화
uv venv .venv
source .venv/bin/activate

# 의존성 설치
uv pip install torch transformers peft accelerate huggingface_hub wandb datasets tqdm pyyaml
```

### 3. 데이터 준비
```bash
# 전처리된 데이터 복사
mkdir -p data/processed
# ko_processed.jsonl, en_processed.jsonl을 data/processed/에 넣기
```

### 4. WandB 설정
```bash
export WANDB_API_KEY="your-api-key"
```

## 학습 실행 (항상 가상환경 안에서)

```bash
# 가상환경 활성화 확인
source .venv/bin/activate

# 단일 GPU
python src/train.py --config configs/base.yaml

# Multi-GPU (A100 x4)
torchrun --nproc_per_node=4 src/train.py --config configs/base.yaml

# WandB 없이 테스트
WANDB_MODE=disabled torchrun --nproc_per_node=4 src/train.py --config configs/base.yaml
```

## 트러블슈팅

### 가상환경 확인
```bash
which python
# /path/to/monolingual-text-only-for-mt/.venv/bin/python 이어야 함
```

### 시스템 패키지 충돌
반드시 `source .venv/bin/activate` 후 실행할 것!

### 버전 오류
```bash
uv pip install --upgrade transformers peft accelerate huggingface_hub
```

# 금융 번역 모델 (MidM Financial Translation)

## 환경 설정

### 1. uv 설치 (권장)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # 또는 터미널 재시작
```

### 2. 가상환경 생성 및 의존성 설치
```bash
# 프로젝트 클론
git clone https://github.com/contiloop/monolingual-text-only-for-mt.git
cd monolingual-text-only-for-mt

# 가상환경 생성
uv venv .venv
source .venv/bin/activate

# 의존성 설치
uv pip install -e .

# 또는 개별 설치 (트러블슈팅용)
uv pip install torch transformers peft accelerate huggingface_hub wandb datasets tqdm pyyaml
```

### 3. 데이터 준비
```bash
# 로컬에서 전처리된 데이터를 서버로 전송
mkdir -p data/processed
# rsync 또는 scp로 ko_processed.jsonl, en_processed.jsonl 복사
```

### 4. WandB 설정
```bash
export WANDB_API_KEY="your-api-key"
# 또는
wandb login
```

## 학습 실행

### 단일 GPU
```bash
python src/train.py --config configs/base.yaml
```

### Multi-GPU (A100 x4 등)
```bash
torchrun --nproc_per_node=4 src/train.py --config configs/base.yaml
```

### WandB 없이 테스트
```bash
WANDB_MODE=disabled torchrun --nproc_per_node=4 src/train.py --config configs/base.yaml
```

## 프로젝트 구조
```
.
├── configs/          # 학습 설정
├── data/processed/   # 전처리 데이터 (Git 제외)
├── scripts/          # 전처리 스크립트
├── src/              # 학습 코드
│   ├── data/         # DataLoader
│   ├── bt/           # Back-Translation
│   └── train.py      # 메인 학습
└── pyproject.toml    # 의존성
```

## 트러블슈팅

### `ModuleNotFoundError`
```bash
uv pip install --upgrade huggingface_hub peft accelerate
```

### `HFValidationError` 관련
```bash
uv pip install huggingface_hub>=0.20.0
```

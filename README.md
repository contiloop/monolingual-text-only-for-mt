# 금융 번역 모델 (MidM Financial Translation)

## Vast.AI 환경 설정

```bash
# 1. 프로젝트 클론
cd /workspace
git clone https://github.com/contiloop/monolingual-text-only-for-mt.git
cd monolingual-text-only-for-mt

# 2. 의존성 설치 (시스템 Python에)
pip install -e .

# 3. 데이터 복사
mkdir -p data/processed
# ko_processed.jsonl, en_processed.jsonl을 data/processed/에 업로드

# 4. WandB 로그인
wandb login

# 5. 학습 실행
python src/train.py --config configs/base.yaml
```

## Multi-GPU 실행
```bash
torchrun --nproc_per_node=4 src/train.py --config configs/base.yaml
```

## WandB 없이 테스트
```bash
WANDB_MODE=disabled python src/train.py --config configs/base.yaml
```

## 프로젝트 구조
```
.
├── configs/          # 학습 설정
├── data/processed/   # 전처리 데이터
├── src/              # 학습 코드
└── pyproject.toml    # 의존성
```

## 트러블슈팅

### ModuleNotFoundError
```bash
pip install -e .
```

### CUDA OOM
`configs/base.yaml`에서 배치 사이즈 줄이기:
```yaml
training:
  batch_size: 2
```

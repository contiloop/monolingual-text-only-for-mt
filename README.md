# 금융 번역 모델 (MidM Financial Translation)

## 환경 설정 (uv 사용)

```bash
# 1. uv 설치 (없으면)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 2. 프로젝트 클론
cd /workspace
git clone https://github.com/contiloop/monolingual-text-only-for-mt.git
cd monolingual-text-only-for-mt

# 3. 가상환경 생성 및 활성화
uv venv .venv
source .venv/bin/activate

# 4. 의존성 설치
uv pip install -e .

# 5. 데이터 복사
mkdir -p data/processed
# ko_processed.jsonl, en_processed.jsonl → data/processed/

# 6. WandB 로그인
wandb login

# 7. 학습 실행
python src/train.py --config configs/base.yaml
```

## Multi-GPU
```bash
torchrun --nproc_per_node=4 src/train.py --config configs/base.yaml
```

## WandB 없이 테스트
```bash
WANDB_MODE=disabled python src/train.py --config configs/base.yaml
```

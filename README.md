# 금융 번역 모델 학습

## 빠른 시작 (원격 서버)

### 1. 환경 설정
```bash
# 코드 클론
git clone <your-repo-url>
cd project

# 가상환경 생성 (uv 권장)
uv venv .venv && source .venv/bin/activate

# 의존성 설치
uv pip install torch transformers peft accelerate wandb datasets tqdm
```

### 2. 데이터 전송 (로컬 → 서버)
```bash
# 로컬에서 실행
rsync -avz --progress \
  data/processed/ \
  user@server:/path/to/project/data/processed/
```

### 3. WandB 로그인
```bash
wandb login
# API 키 입력
```

### 4. Accelerate 설정
```bash
accelerate config
# 대화형으로 GPU 수, 분산 방식 선택
```

### 5. 학습 실행
```bash
# 단일 GPU
python src/train.py --config configs/base.yaml

# Multi-GPU (자동 분산)
accelerate launch src/train.py --config configs/base.yaml

# DeepSpeed ZeRO-2
accelerate launch --use_deepspeed src/train.py --config configs/base.yaml
```

## 주요 설정 (configs/base.yaml)

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `model.name` | kt-ai/midm-12b | HuggingFace 모델 |
| `training.batch_size` | 4 | 배치 크기 |
| `training.steps` | 50000 | 총 스텝 |
| `model.lora.r` | 64 | LoRA rank |

## 모니터링

```bash
# WandB 대시보드
https://wandb.ai/<your-project>

# GPU 사용량
watch -n 1 nvidia-smi
```

## 체크포인트

자동 저장: `outputs/ckpt_1000/`, `ckpt_2000/`...
최종: `outputs/final/`

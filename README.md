# 금융 번역 모델 (MidM Financial Translation)

## 빠른 시작

```bash
# 1. 클론 & 환경 설정
git clone https://github.com/contiloop/monolingual-text-only-for-mt.git
cd monolingual-text-only-for-mt
uv venv .venv && source .venv/bin/activate

# 기본 설치 (SDPA 사용)
uv pip install -e .

# Flash Attention 2 포함 설치 (GPU 서버 권장, 더 빠름)
uv pip install -e ".[flash]" --no-build-isolation

# 2. WandB 로그인
wandb login

# 3. 학습 실행 (Hydra config 시스템)
python src/train.py                    # 기본 설정 (4x16GB GPU)
python src/train.py gpu=a100           # A100 프로파일
python src/train.py gpu=single_4bit    # 단일 GPU 4-bit
```

### WandB 로그인 안 될 때 (Jupyter/Lightning AI)

```bash
# 방법 1: 환경변수로 실행
WANDB_API_KEY=your-api-key python src/train.py

# 방법 2: Python에서 직접 설정
python -c "import os; os.environ['WANDB_API_KEY']='your-api-key'"

# 방법 3: .netrc 파일 생성
echo "machine api.wandb.ai login user password YOUR_API_KEY" > ~/.netrc
```

> API Key 확인: https://wandb.ai/authorize

---

## 데이터 준비

### 1. 학습 데이터 (Monolingual) 전송

**로컬에서 압축:**
```bash
cd /path/to/local/data/processed
tar -czvf processed_data.tar.gz *.jsonl
```

**서버로 전송:**
```bash
# SSH (Vast.AI 등)
scp -P [포트] processed_data.tar.gz root@[IP]:/workspace/monolingual-text-only-for-mt/data/processed/

# 백그라운드 전송 (터미널 닫아도 계속됨)
nohup scp -P [포트] data/processed/processed_data.tar.gz root@[IP]:/workspace/monolingual-text-only-for-mt/data/processed/ > scp_upload.log 2>&1 &
tail -f scp_upload.log  # 진행 확인

# 또는 Jupyter UI에서 Upload
```

**한 줄로 삭제 + 전송 + 압축해제:**
```bash
ssh -p [포트] root@[IP] "rm -rf /workspace/monolingual-text-only-for-mt/data/processed/*" && \
scp -P [포트] data/processed/processed_data.tar.gz root@[IP]:/workspace/monolingual-text-only-for-mt/data/processed/ && \
ssh -p [포트] root@[IP] "cd /workspace/monolingual-text-only-for-mt/data/processed && tar -xzvf processed_data.tar.gz && rm processed_data.tar.gz"
```

**서버에서 압축 해제 (수동):**
```bash
cd /workspace/monolingual-text-only-for-mt/data/processed
tar -xzvf processed_data.tar.gz
```

### 2. 평가용 병렬 코퍼스 다운로드

**서버에서 직접 다운로드 (권장):**
```bash
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('lemon-mint/korean_english_parallel_wiki_augmented_v1')
ds.save_to_disk('data/eval/korean_english_parallel')
print('✅ 다운로드 완료!')
"
```

**로컬에서 전송하려면:**
```bash
# 로컬
tar -czvf parallel_corpus.tar.gz korean_english_parallel_dataset/

# 서버
cd /workspace/monolingual-text-only-for-mt/data/eval
tar -xzvf parallel_corpus.tar.gz
mv korean_english_parallel_dataset korean_english_parallel
```

---

## GPU별 설정 (Hydra)

```bash
# 4× RTX 4080/5080 16GB (기본값)
torchrun --nproc_per_node=4 src/train.py

# A100 40-80GB
torchrun --nproc_per_node=4 src/train.py gpu=a100

# 단일 GPU 4-bit QLoRA
python src/train.py gpu=single_4bit

# CLI에서 값 override
python src/train.py training.batch_size=2 training.learning_rate=1e-4

# Sweep (여러 LR 테스트)
python src/train.py --multirun training.learning_rate=1e-4,2e-4,5e-4
```

---

## Config 구조 (Hydra)

```
configs/
├── config.yaml           # 메인 (defaults 정의)
├── model/
│   └── midm.yaml         # 모델 설정
├── training/
│   └── default.yaml      # 학습 기본 설정
├── data/
│   └── default.yaml      # 데이터 경로
└── gpu/
    ├── 4x16gb.yaml       # 4× RTX 4080/5080 (기본값)
    ├── a100.yaml         # A100 40GB+
    └── single_4bit.yaml  # 단일 GPU
```

**장점:**
- GPU별 config에는 override만 작성 (중복 제거)
- CLI에서 `training.batch_size=2` 형태로 즉시 변경
- 실험 결과 자동 저장: `outputs/프로젝트명/날짜/시간/`

---

## 프로젝트 구조
```
.
├── configs/              # Hydra config (계층적)
├── data/
│   ├── processed/        # 학습 데이터 (ko/en_processed.jsonl)
│   └── eval/             # 평가용 병렬 코퍼스
├── src/                  # 학습 코드
└── outputs/              # 체크포인트 (자동 생성)
```

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

# 또는 Jupyter UI에서 Upload
```

**서버에서 압축 해제:**
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

## GPU별 설정

```bash
# RTX 4080/4090 16GB (4-bit QLoRA)
python src/train.py --config configs/base.yaml

# A100 40-80GB (8-bit)
python src/train.py --config configs/gpu_8bit.yaml

# Multi-GPU
torchrun --nproc_per_node=2 src/train.py --config configs/base.yaml
```

---

## 프로젝트 구조
```
.
├── configs/              # 학습 설정
├── data/
│   ├── processed/        # 학습 데이터 (ko/en_processed.jsonl)
│   └── eval/             # 평가용 병렬 코퍼스
├── src/                  # 학습 코드
└── outputs/              # 체크포인트
```

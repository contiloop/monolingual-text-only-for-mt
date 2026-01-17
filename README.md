# 금융 번역 모델 (MidM Financial Translation)

## 빠른 시작

```bash
# 1. 클론 & 환경 설정
git clone https://github.com/contiloop/monolingual-text-only-for-mt.git
cd monolingual-text-only-for-mt
uv venv .venv && source .venv/bin/activate

# 기본 설치 (SDPA 사용)
uv pip install -e .

# Flash Attention 2 설치 (선택사항, GPU 서버 권장)
# - 요구사항: CUDA 11.6+, GPU Compute Capability 8.0+ (Ampere 이상: A100, RTX 30xx/40xx/50xx)
# - 효과: 학습 속도 20-30% 향상
# - 주의: CUDA 버전에 맞춰 소스에서 빌드 (10-20분 소요)

# CUDA 버전 확인
nvcc --version

# Flash Attention 설치 (현재 CUDA 버전에 맞춰 자동 빌드)
pip install ninja packaging wheel
MAX_JOBS=4 pip install flash-attn --no-build-isolation

# 2. WandB 로그인
wandb login

# 3. 학습 실행 (Hydra + Experiment)
# 4x16GB GPU 기준
torchrun --nproc_per_node=4 src/train.py experiment=4x16gb
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

**로컬 Mac에서 공개키 확인**
```
cat ~/.ssh/id_rsa.pub
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

## 학습 실행 (Experiment 패턴)

이 프로젝트는 Hydra의 **Experiment** 패턴을 사용하여 하드웨어 설정을 관리합니다.

### 주요 학습 명령어

```bash
# 단일 A100 80GB (Full precision bf16, 200K steps)
python src/train.py +experiment=96gb training.early_stopping.enabled=false

# 4× RTX 4080/5080 16GB (4-bit QLoRA)
torchrun --nproc_per_node=4 src/train.py +experiment=4x16gb

# A100 40-80GB 멀티 GPU (8-bit)
torchrun --nproc_per_node=4 src/train.py +experiment=a100

# 디버깅 모드 (1 step 실행, 저장 안 함)
python src/train.py +experiment=debug

# CLI에서 값 override (실험 config 위에 덮어쓰기)
python src/train.py +experiment=96gb training.batch_size=8 training.steps=100000
```

### Iterative Back-Translation Workflow

이 프로젝트는 **iterative BT (back-translation)** 방식을 사용합니다:

1. **Phase 1 (0~5000 steps)**: L_auto만으로 denoising 학습
2. **Step 5000**: 학습 자동 중단 → BT 데이터 생성 필요
3. **Phase 2 (5000~10000)**: L_auto + L_back 동시 학습
4. **Step 10000**: 다시 중단 → BT 재생성 (더 좋은 품질)
5. 반복...

#### BT Generation Mode 설정

```yaml
# configs/training/default.yaml
bt_generation_mode: 'pause'  # 기본값: 학습 중단 후 수동 생성
# 'online': 학습 중 자동 생성 (GPU 메모리 충분할 때만)
# 'manual': BT 스킵하고 계속 학습
```

#### 일반적인 워크플로우

**1단계: 초기 학습 시작**
```bash
python src/train.py
# Step 5000에 자동 중단되며 BT generation 명령어 출력됨
```

**2단계: BT 데이터 생성** (학습 중단된 상태에서)

학습이 자동 중단되면 두 가지 옵션이 출력됩니다:

**Option A: vLLM (빠름, 10K 샘플 ~10-20분)**
```bash
# vLLM 설치 필요 (한 번만)
pip install vllm

# BT 생성
python src/bt/vllm_generator.py \
    --base_model K-intelligence/Midm-2.0-Base-Instruct \
    --adapter ./outputs/ckpt_5000 \
    --input_file ./data/processed/ko_processed.jsonl \
    --output_file ./data/bt_cache/bt_5000.jsonl \
    --direction ko_to_en \
    --max_samples 10000
```

**Option B: Transformers (느림 ~1-2시간, 추가 설치 불필요)**
```bash
# vLLM 없이 작동
python src/bt/transformers_generator.py \
    --base_model K-intelligence/Midm-2.0-Base-Instruct \
    --adapter ./outputs/ckpt_5000 \
    --input_file ./data/processed/ko_processed.jsonl \
    --output_file ./data/bt_cache/bt_5000.jsonl \
    --direction ko_to_en \
    --max_samples 10000 \
    --load_in_4bit  # 메모리 절약
```

**3단계: 학습 재개**
```bash
python src/train.py training.resume_from_checkpoint="./outputs/ckpt_5000"
# Step 10000에 다시 중단 → BT 재생성 → 반복
```

### WandB 모니터링

학습 중 다음을 실시간으로 확인할 수 있습니다:
- **Loss 그래프**: Training/Validation loss 추이
- **Denoising Samples**: Noisy → Generated → Original 비교 (Table)
- **BLEU Score**: 번역 품질 (병렬 코퍼스 사용 시)

WandB 프로젝트: `https://wandb.ai/YOUR_USERNAME/midm-12b-financial-translation`

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
└── experiment/           # 하드웨어/실험 프로필 (설정 덮어쓰기)
    ├── 4x16gb.yaml       # 4-bit, Gradient Acc 16
    ├── a100.yaml         # 8-bit, Gradient Acc 4
    └── debug.yaml        # 디버깅용
```

**장점:**
- GPU별 config에는 override만 작성 (중복 제거)
- CLI에서 `training.batch_size=2` 형태로 즉시 변경
- 실험 결과 자동 저장: `outputs/프로젝트명/날짜/시간/`

---

## Denoising 학습 전략

이 프로젝트는 **Monolingual Denoising**을 통해 번역 모델을 학습합니다. 노이즈가 섞인 텍스트를 원본으로 복원하는 과정에서 언어의 구조와 표현력을 학습합니다.

### 목표
- **L_auto (Denoising)**: 노이즈 → 원본 복원
- **L_back (Back-Translation)**: 번역 품질 향상 (Phase 2)

### 핵심 전략

#### 1. Infilling 위주 노이즈 (60%)
```yaml
# configs/data/default.yaml
noise:
  infilling_prob: 0.60  # [MASK] 토큰으로 대체 - 복사 불가능
```
- **Why**: 단순 typo는 복사해도 85% 정확 → trivial copy 학습
- **Solution**: `[MASK]`로 마스킹하면 문맥 추론 강제

#### 2. Dynamic Noise Scheduling (0-40%)
```yaml
noise:
  dynamic_noise: true
  dynamic_noise_min: 0.0
  dynamic_noise_max: 0.40
```
- **Why**: 고정 비율 → 모델이 "N% 채우면 됨" 패턴 암기
- **Solution**: 매번 랜덤 비율 → 문맥 기반 판단 학습

#### 3. Clean Data Mix (15%)
```yaml
noise:
  clean_ratio: 0.15  # 15% 확률로 노이즈 없는 데이터
```
- **Why**: 노이즈만 보면 "모든 문장은 고장남" 편향
- **Solution**: 멀쩡한 문장도 섞어서 Identity Mapping 학습

#### 4. Diff-Weighted Loss (10x)
```yaml
training_loss:
  diff_weight: 10.0  # 노이즈 복원 위치에 10배 가중치
```
- **Why**: 복사 성공(85%) vs 복원 실패(15%) 동등 취급
- **Solution**: 틀린 위치에 10배 페널티 → 복원에 집중

#### 5. Instruction Prompt
```
Fix the errors in the following text: {noisy}

Corrected version: {clean}
```
- **Why**: `[DENOISE] ... [OUTPUT]` 토큰은 task 의미 불분명
- **Solution**: 자연어 지시문으로 task 명확화

### Config 전체 예시
```yaml
# configs/data/default.yaml
noise:
  total_ratio: 0.30
  clean_ratio: 0.15
  dynamic_noise: true
  dynamic_noise_min: 0.0
  dynamic_noise_max: 0.40
  infilling_prob: 0.60

training_loss:
  diff_weight: 10.0
```

### 테스트 방법
```bash
# Noise 적용 확인
python -c "
from src.data.noise import NoiseApplier, NoiseConfig
applier = NoiseApplier(NoiseConfig())
for i in range(5):
    text = 'This is a test sentence for denoising.'
    noisy, noise_type = applier.apply(text, 'en', '')
    print(f'[{noise_type}] {noisy}')
"
```

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

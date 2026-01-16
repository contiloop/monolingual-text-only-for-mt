#!/bin/bash
# 4-GPU DDP 실행 스크립트 (메모리 최적화)

# PyTorch 메모리 최적화
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 4-GPU DDP 실행
accelerate launch \
  --num_processes=4 \
  --mixed_precision=bf16 \
  src/train.py +experiment=4x16gb

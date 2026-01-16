#!/bin/bash
# 디버그 실행 스크립트 (메모리 최적화)

# PyTorch 메모리 최적화
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 단일 GPU로 실행
CUDA_VISIBLE_DEVICES=0 python src/train.py +experiment=debug

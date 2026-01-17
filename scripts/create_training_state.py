#!/usr/bin/env python3
"""
기존 checkpoint에 training_state.pt를 수동으로 생성하는 스크립트

Usage:
    python scripts/create_training_state.py \
        --checkpoint ./outputs/ckpt_5000 \
        --global_step 5000 \
        --lback_active true
"""

import argparse
import torch
from pathlib import Path


def create_training_state(
    checkpoint_path: str,
    global_step: int,
    lback_active: bool = False
):
    """
    기존 checkpoint에 training_state.pt 생성

    주의: optimizer와 scheduler state는 None으로 설정됩니다.
    이 경우 resume 시 optimizer/scheduler는 재초기화되지만,
    global_step과 lback_active는 복원됩니다.
    """
    ckpt_path = Path(checkpoint_path)

    if not ckpt_path.exists():
        raise ValueError(f"Checkpoint not found: {ckpt_path}")

    # Training state 생성 (optimizer/scheduler는 None)
    # resume 시 이 값들이 None이면 재초기화될 수 있도록 _load_checkpoint 수정 필요
    training_state = {
        'global_step': global_step,
        'lback_active': lback_active,
        'optimizer_state_dict': None,  # 재초기화됨
        'scheduler_state_dict': None,  # 재초기화됨
    }

    output_path = ckpt_path / "training_state.pt"
    torch.save(training_state, str(output_path))

    print(f"✅ Created training_state.pt at {output_path}")
    print(f"   - global_step: {global_step}")
    print(f"   - lback_active: {lback_active}")
    print(f"   - optimizer_state_dict: None (will be re-initialized)")
    print(f"   - scheduler_state_dict: None (will be re-initialized)")
    print()
    print("⚠️  Note: Optimizer and Scheduler will be re-initialized from config.")
    print("   This means learning rate will restart from warmup.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--global_step", type=int, required=True, help="Global step number")
    parser.add_argument("--lback_active", type=lambda x: x.lower() == 'true', default=False, help="Whether L_back is active")

    args = parser.parse_args()

    create_training_state(
        checkpoint_path=args.checkpoint,
        global_step=args.global_step,
        lback_active=args.lback_active
    )

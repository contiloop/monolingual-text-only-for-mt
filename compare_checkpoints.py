#!/usr/bin/env python3
"""
학습한 체크포인트들 비교 스크립트
LoRA adapter 로드
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path
import sys
import json

OUTPUT_DIR = "./outputs"
LOG_FILE = "checkpoint_comparison.txt"

TEST_CASES = [
    "<|casual|> um Q: What was I mean the company's return on equity for the 2021-Q2 period",
    "<|formal|> The revenue [MASK] Q2 2021 increased by approximately 15%",
    "Q: What is the percentage of capex spending that is projects A: The percentage of base capex",
    "<|casual|> Q: What was the percentage increase in net sales for Amazon in Q2 2021 compared to Q2 2020",
]


class TeeOutput:
    """콘솔과 파일에 동시 출력"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def find_checkpoints():
    """체크포인트 찾기"""
    checkpoints = []
    if Path(OUTPUT_DIR).exists():
        for item in sorted(Path(OUTPUT_DIR).iterdir()):
            if item.is_dir() and (item.name.startswith('ckpt_') or item.name == 'final'):
                checkpoints.append(str(item))
    return checkpoints


def load_model(model_path):
    """LoRA 모델 로드 (base model + adapter)"""
    print(f"  Loading {model_path}...")

    BASE_MODEL = "K-intelligence/Midm-2.0-Base-Instruct"

    # 1. Tokenizer 로드 (체크포인트에서)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # 2. Base model 로드
    print(f"    Loading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # 3. Special tokens 추가 (학습 시와 동일하게)
    special_tokens = ['<|formal|>', '<|casual|>', '<|sep|>']
    num_added = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

    if num_added > 0:
        print(f"    Added {num_added} special tokens")
        base_model.resize_token_embeddings(len(tokenizer))

    # 4. LoRA adapter 로드
    print(f"    Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, model_path)

    # 5. Merge adapter (inference 속도 향상)
    print(f"    Merging adapter...")
    model = model.merge_and_unload()

    model.eval()
    print(f"  ✓ Loaded successfully! (vocab_size={len(tokenizer)})")
    return model, tokenizer


def denoise(model, tokenizer, noisy_text):
    """Denoising"""
    prompt = f"{noisy_text} {tokenizer.eos_token}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][inputs['input_ids'].shape[1]:]
    clean_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return clean_text


def main():
    # 출력을 콘솔과 파일에 동시 저장
    tee = TeeOutput(LOG_FILE)
    sys.stdout = tee

    try:
        checkpoints = find_checkpoints()

        print("=" * 100)
        print("🔍 Checkpoint Comparison")
        print(f"   Base Model: K-intelligence/Midm-2.0-Base-Instruct")
        print("=" * 100)
        print("Found checkpoints:")
        for ckpt in checkpoints:
            print(f"  - {ckpt}")
        print("=" * 100)
        print()

        # 각 테스트 케이스별로 비교
        for test_idx, noisy_text in enumerate(TEST_CASES, 1):
            print(f"\n{'=' * 100}")
            print(f"📝 TEST CASE {test_idx}")
            print(f"{'=' * 100}")
            print(f"Noisy Input:")
            print(f"  {noisy_text}")
            print()

            # 각 체크포인트 테스트
            for ckpt_path in checkpoints:
                ckpt_name = Path(ckpt_path).name
                print(f"┌─ [{ckpt_name}] " + "─" * 80)
                try:
                    model, tokenizer = load_model(ckpt_path)
                    output = denoise(model, tokenizer, noisy_text)
                    print(f"│ Clean Output:")
                    print(f"│   {output}")
                    del model
                    torch.cuda.empty_cache()
                except Exception as e:
                    import traceback
                    print(f"│ ❌ Error: {str(e)[:300]}")
                print(f"└" + "─" * 90)
                print()

        print("=" * 100)
        print(f"✅ Comparison complete! Results saved to: {LOG_FILE}")
        print("=" * 100)

    finally:
        sys.stdout = tee.terminal
        tee.close()


if __name__ == "__main__":
    main()

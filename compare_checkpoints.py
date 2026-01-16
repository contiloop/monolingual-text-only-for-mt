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
    # 뉴스/금융 문서 스타일 (학습 데이터와 유사)
    "<|formal|> The company's revenue [MASK] Q2 2021 increased by approximately 15%",
    "<|casual|> um The total revenue for you know the second quarter was like $5.2 billion",
    "The company reported strong performance with net income of approximately uh $1.2 billion",
    "<|formal|> Market analysts expect the stock price to uh you know continue rising in the next quarter",
    "The Federal Reserve announced that interest rates would uh remain unchanged at like 5.25 percent",
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


# Base model 캐싱 (한 번만 로드)
_base_model_cache = None
_tokenizer_cache = None

def get_base_model():
    """Base model 캐싱"""
    global _base_model_cache, _tokenizer_cache

    if _base_model_cache is None:
        BASE_MODEL = "K-intelligence/Midm-2.0-Base-Instruct"
        print(f"  Loading base model (once): {BASE_MODEL}")

        _tokenizer_cache = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True
        )

        # Special tokens 추가 (학습 시와 동일)
        special_tokens = ['<|formal|>', '<|casual|>', '<|sep|>']
        _tokenizer_cache.add_special_tokens({'additional_special_tokens': special_tokens})

        _base_model_cache = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        # Embedding resize
        _base_model_cache.resize_token_embeddings(len(_tokenizer_cache))
        print(f"  ✓ Base model loaded (vocab_size={len(_tokenizer_cache)})")

    return _base_model_cache, _tokenizer_cache


def load_model(adapter_path):
    """LoRA adapter 로드"""
    print(f"  Loading adapter: {adapter_path}")

    # Base model을 새로 복사 (adapter 간섭 방지)
    BASE_MODEL = "K-intelligence/Midm-2.0-Base-Instruct"
    _, tokenizer = get_base_model()  # tokenizer만 캐시에서 가져오기

    # Base model 새로 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    base_model.resize_token_embeddings(len(tokenizer))

    # LoRA adapter 로드
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Merge (추론 속도 향상)
    print(f"  Merging adapter...")
    model = model.merge_and_unload()
    model.eval()

    print(f"  ✓ Adapter loaded and merged!")
    return model, tokenizer


def denoise(model, tokenizer, noisy_text):
    """Denoising"""
    prompt = f"{noisy_text} {tokenizer.eos_token}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

    # token_type_ids 제거 (LLaMA 계열은 사용 안함)
    inputs = {k: v.to(model.device) for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}

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

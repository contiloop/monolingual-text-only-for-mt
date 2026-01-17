#!/usr/bin/env python3
"""
간단한 모델 테스트 스크립트
Usage:
    python test_model.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 설정
CHECKPOINT = "./outputs/final"  # 체크포인트 경로
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"📦 Loading model from {CHECKPOINT}...")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()
print(f"✅ Model loaded on {DEVICE}\n")

# 테스트 케이스
test_cases = [
    "<|casual|> um Q: What was I mean the company's return on equity for the 2021-Q2 period",
    "<|formal|> The revenue [MASK] Q2 2021 increased by approximately 15%",
    "Q: What is the percentage of capex spending that is projects A: The percentage of base capex",
]

print("🧪 Testing denoising...\n")

for i, noisy_text in enumerate(test_cases, 1):
    print(f"[Test {i}]")
    print(f"📝 Noisy:  {noisy_text}")

    # 입력: <noisy> <eos> → 모델이 clean text 생성
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

    # 생성된 부분만 추출
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    clean_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    print(f"✨ Clean:  {clean_text}\n")

print("✅ Done!")

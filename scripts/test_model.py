#!/usr/bin/env python3
"""
모델 성능 간단 테스트 스크립트

Usage:
    python scripts/test_model.py --checkpoint ./outputs/ckpt_5000
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(base_model: str, adapter_path: str = None):
    """모델 로드"""
    print(f"Loading tokenizer from {adapter_path or base_model}...")
    tokenizer_path = adapter_path if adapter_path else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = 'left'

    print(f"Loading model from {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Resize embeddings
    if len(tokenizer) != model.config.vocab_size:
        print(f"Resizing embeddings: {model.config.vocab_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    # Load adapter
    if adapter_path:
        print(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    print("Model ready!\n")
    return model, tokenizer


def test_translation(model, tokenizer, text: str, direction: str = "ko_to_en"):
    """번역 테스트"""
    if direction == "ko_to_en":
        prompt = f"다음 한국어를 영어로 번역하세요:\n{text}\n번역:"
    else:
        prompt = f"Translate the following English to Korean:\n{text}\nTranslation:"

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items() if k != 'token_type_ids'}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,  # Greedy decoding for consistency
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt
    if direction == "ko_to_en" and "번역:" in generated:
        result = generated.split("번역:", 1)[1].strip()
    elif direction == "en_to_ko" and "Translation:" in generated:
        result = generated.split("Translation:", 1)[1].strip()
    else:
        result = generated.replace(prompt, '').strip()

    return result


def test_denoising(model, tokenizer, text: str):
    """Denoising 테스트"""
    # Simple noise: add repetition
    words = text.split()
    if len(words) > 3:
        words[2] = words[2] + " " + words[2]  # Duplicate a word
    noisy = " ".join(words)

    prompt = f"[DENOISE] {noisy} [OUTPUT]"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items() if k != 'token_type_ids'}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)

    if '[OUTPUT]' in generated:
        result = generated.split('[OUTPUT]', 1)[1].strip()
    else:
        result = generated.replace(prompt, '').strip()

    result = result.replace(tokenizer.eos_token, '').strip()

    return noisy, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="K-intelligence/Midm-2.0-Base-Instruct")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (e.g., ./outputs/ckpt_5000)")
    args = parser.parse_args()

    model, tokenizer = load_model(args.base_model, args.checkpoint)

    print("="*80)
    print("MODEL PERFORMANCE TEST")
    print("="*80)

    # Test 1: Korean to English
    print("\n📝 Test 1: Korean → English Translation")
    print("-"*80)
    ko_text = "삼성전자가 올해 3분기 영업이익이 전년 대비 274% 증가한 7조 7천억원을 기록했다고 발표했다."
    print(f"Input (KO):  {ko_text}")
    en_result = test_translation(model, tokenizer, ko_text, "ko_to_en")
    print(f"Output (EN): {en_result}")

    # Test 2: English to Korean
    print("\n📝 Test 2: English → Korean Translation")
    print("-"*80)
    en_text = "The company reported revenue of $50 billion for the second quarter, representing a 15% increase year-over-year."
    print(f"Input (EN):  {en_text}")
    ko_result = test_translation(model, tokenizer, en_text, "en_to_ko")
    print(f"Output (KO): {ko_result}")

    # Test 3: Denoising
    print("\n📝 Test 3: Denoising")
    print("-"*80)
    clean_text = "애플이 신제품 출시 이후 주가가 5% 상승했다."
    noisy, denoised = test_denoising(model, tokenizer, clean_text)
    print(f"Noisy:    {noisy}")
    print(f"Denoised: {denoised}")
    print(f"Original: {clean_text}")

    print("\n" + "="*80)
    print("✅ Test completed!")
    print("="*80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
간단한 번역 테스트 스크립트

Usage:
    python scripts/test_translation.py --checkpoint ./outputs/ckpt_5000

    # Custom text
    python scripts/test_translation.py --checkpoint ./outputs/ckpt_5000 --text "Your text here" --direction en_to_ko
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(base_model: str, checkpoint_path: str):
    """모델 로드"""
    print('Loading model...')

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'  # Truncate from left, keep recent context
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True
    )

    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model = model.merge_and_unload()
    model.eval()

    print(f'Model ready! Max position embeddings: {model.config.max_position_embeddings}')
    print()
    return model, tokenizer


def translate(model, tokenizer, text: str, direction: str = "en_to_ko", max_new_tokens: int = 1024):
    """번역 수행"""
    # Better prompt format: text first, then instruction
    if direction == "ko_to_en":
        prompt = f"{text}\n\n위 한국어 텍스트를 영어로 번역하세요:"
        split_key = "영어로 번역하세요:"
    else:  # en_to_ko
        prompt = f"{text}\n\n위 영어 텍스트를 한국어로 번역하세요:"
        split_key = "한국어로 번역하세요:"

    inputs = tokenizer(prompt, return_tensors='pt', max_length=4096, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items() if k != 'token_type_ids'}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt
    if split_key in result:
        result = result.split(split_key, 1)[1].strip()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="K-intelligence/Midm-2.0-Base-Instruct")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--text", type=str, default=None, help="Text to translate")
    parser.add_argument("--direction", type=str, default="en_to_ko", choices=["en_to_ko", "ko_to_en"])
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    model, tokenizer = load_model(args.base_model, args.checkpoint)

    # Default test text if not provided
    if args.text is None:
        args.text = """The pause was the culmination of nearly a week of escalating tensions between Washington and Tehran, during which the U.S.'s regional allies warned Trump that a bombing campaign might lead to a wider conflict and senior U.S. military officials prepared for a strike order Wednesday that never came.

The prospect of an attack, less than two weeks after U.S. forces captured Venezuelan leader Nicolás Maduro, rattled leaders in capitals across the world, who feared that Trump's penchant for quick aerial strikes could spark another protracted conflict in the Middle East while failing to dislodge the Iranian regime.

The U.S. is sending an aircraft-carrier strike group, additional jet fighters and missile defenses to the region, in a sign that bombs could still fall shortly after their arrival. But asked by reporters Friday whether American help for protesters was still on the way as promised, Trump said he alone decided not to issue an attack order.

"Nobody convinced me. I convinced myself," he said. "They didn't hang anyone. They canceled the hangings. That had a big impact." """

    print('='*80)
    print(f'TRANSLATION TEST ({args.direction.upper()})')
    print('='*80)
    print(f'\nINPUT:')
    print(args.text)
    print('\n' + '='*80)

    result = translate(model, tokenizer, args.text, args.direction, args.max_new_tokens)

    print(f'\nOUTPUT:')
    print(result)
    print('='*80)


if __name__ == "__main__":
    main()

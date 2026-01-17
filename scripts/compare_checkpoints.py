#!/usr/bin/env python
"""
Denoising 체크포인트 비교 스크립트
학습 시 사용한 프롬프트 형식: "Fix the errors in the following text: ... Corrected version:"
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(ckpt_path: str, base_model: str = "K-intelligence/Midm-2.0-Base-Instruct"):
    """체크포인트에서 모델 로드"""
    print(f"\n📦 Loading checkpoint: {ckpt_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    tokenizer.padding_side = 'left'
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True
    )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, ckpt_path)
    model = model.merge_and_unload()
    model.eval()
    
    return model, tokenizer

def denoise(model, tokenizer, noisy_text: str, max_new_tokens: int = 256):
    """Denoising 수행 - 학습 시 사용한 프롬프트 형식"""
    # 학습 시 사용한 프롬프트 형식
    prompt = f"Fix the errors in the following text: {noisy_text}\n\nCorrected version: "
    
    inputs = tokenizer(prompt, return_tensors='pt')
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
    
    # "Corrected version: " 이후 텍스트만 추출
    if "Corrected version: " in result:
        result = result.split("Corrected version: ", 1)[1].strip()
    else:
        result = result.replace(prompt, '').strip()
    
    return result

def main():
    # 테스트 케이스
    test_cases = [
        {
            "noisy": "삼성전자가 삼성전자가 올해 3분기 영업이익이 영업이익이 전년 대비 274% 증가한 증가한 7조 7천억원을 기록했다고 발표했다.",
            "expected": "삼성전자가 올해 3분기 영업이익이 전년 대비 274% 증가한 7조 7천억원을 기록했다고 발표했다."
        },
        {
            "noisy": "Q: What was [MASK] total revenue for the company in Q4 2023? A: The total revenue was $2.5 billion.",
            "expected": "Q: What was the total revenue for the company in Q4 2023? A: The total revenue was $2.5 billion."
        },
        {
            "noisy": "ECB가 ECB가 금리를 um 0.25% 인상했다고 um 발표했다.",
            "expected": "ECB가 금리를 0.25% 인상했다고 발표했다."
        }
    ]
    
    # 비교할 체크포인트
    checkpoints = [
        "/workspace/monolingual-text-only-for-mt/outputs/ckpt_1000",
        "/workspace/monolingual-text-only-for-mt/outputs/ckpt_5000"
    ]
    
    print("=" * 80)
    print("🔍 DENOISING CHECKPOINT COMPARISON")
    print("=" * 80)
    
    for ckpt_path in checkpoints:
        try:
            model, tokenizer = load_model(ckpt_path)
            ckpt_name = ckpt_path.split('/')[-1]
            
            print(f"\n{'=' * 80}")
            print(f"📍 {ckpt_name}")
            print("=" * 80)
            
            for i, case in enumerate(test_cases, 1):
                print(f"\n[Test {i}]")
                print(f"Noisy:    {case['noisy'][:80]}...")
                
                result = denoise(model, tokenizer, case['noisy'])
                print(f"Output:   {result[:80]}...")
                print(f"Expected: {case['expected'][:80]}...")
                
                # 간단한 유사도 체크
                match = "✅" if case['expected'][:50] in result else "⚠️"
                print(f"Match: {match}")
            
            # 메모리 정리
            del model, tokenizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Error loading {ckpt_path}: {e}")
    
    print("\n" + "=" * 80)
    print("✅ Comparison Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()

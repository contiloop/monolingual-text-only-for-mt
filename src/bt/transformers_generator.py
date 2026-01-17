#!/usr/bin/env python3
"""
Transformers 기반 Back-Translation 생성기 (vLLM 대안)

장점:
- vLLM 불필요 (이미 설치된 transformers 사용)
- 학습 모델과 별도 GPU 불필요
- 간단한 설정

단점:
- vLLM보다 느림 (~5-10배)
- 메모리 효율 낮음

Usage:
    python src/bt/transformers_generator.py \
        --base_model K-intelligence/Midm-2.0-Base-Instruct \
        --adapter ./outputs/ckpt_5000 \
        --input_file ./data/processed/ko_processed.jsonl \
        --output_file ./data/bt_cache/bt_5000.jsonl \
        --direction ko_to_en \
        --max_samples 10000
"""

import json
import argparse
from typing import List, Optional
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class TransformersBTGenerator:
    """Transformers 기반 BT 생성기"""

    def __init__(
        self,
        base_model_path: str,
        lora_adapter_path: Optional[str] = None,
        device: str = "cuda",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):
        self.device = device

        print(f"Loading tokenizer from {base_model_path}...")

        # Load tokenizer from adapter if available (contains added special tokens)
        tokenizer_path = lora_adapter_path if lora_adapter_path else base_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model loading
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
        }

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            print("Loading model in 4-bit...")
        elif load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            print("Loading model in 8-bit...")
        else:
            print("Loading model in bf16...")

        print(f"Loading base model from {base_model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path, **model_kwargs
        )

        # Resize embeddings to match tokenizer vocab size
        if len(self.tokenizer) != self.model.config.vocab_size:
            print(f"Resizing embeddings: {self.model.config.vocab_size} -> {len(self.tokenizer)}")
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Load LoRA adapter if provided
        if lora_adapter_path:
            print(f"Loading LoRA adapter from {lora_adapter_path}...")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)
            self.model = self.model.merge_and_unload()  # Merge for faster inference

        self.model.eval()
        print("Model ready!")

    def generate_bt(
        self,
        source_texts: List[str],
        direction: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
    ) -> List[str]:
        """BT 생성"""

        prompts = [self._build_prompt(text, direction) for text in source_texts]

        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated_texts = []
        for i, output in enumerate(outputs):
            # Remove prompt from output
            prompt_length = inputs.input_ids[i].shape[0]
            generated = self.tokenizer.decode(
                output[prompt_length:], skip_special_tokens=True
            )
            generated_texts.append(generated.strip())

        return generated_texts

    def _build_prompt(self, text: str, direction: str) -> str:
        """프롬프트 구성"""
        if direction == "ko_to_en":
            return f"다음 한국어를 영어로 번역하세요:\n{text}\n번역:"
        else:
            return f"Translate the following English to Korean:\n{text}\nTranslation:"

    def generate_and_save(
        self,
        input_path: str,
        output_path: str,
        direction: str,
        batch_size: int = 8,
        limit: Optional[int] = None,
    ):
        """파일에서 읽어 생성 후 저장"""

        # Load input
        texts = []
        print(f"Loading input from {input_path}...")
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                texts.append(data["text"])
                if limit and len(texts) >= limit:
                    break

        print(f"Generating BT for {len(texts)} samples (Direction: {direction})...")

        # Create output directory if needed
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Process in batches
        output_file = open(output_path, "w", encoding="utf-8")

        for i in tqdm(range(0, len(texts), batch_size), desc="Generating BT"):
            batch_texts = texts[i : i + batch_size]
            generated = self.generate_bt(batch_texts, direction)

            # Save results
            for source, gen in zip(batch_texts, generated):
                result = {
                    "source": source,
                    "generated": gen,
                    "direction": direction,
                }
                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")

            # Free memory
            if i % (batch_size * 10) == 0:
                output_file.flush()
                torch.cuda.empty_cache()

        output_file.close()
        print(f"Done! Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument(
        "--direction", type=str, choices=["ko_to_en", "en_to_ko"], required=True
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    args = parser.parse_args()

    generator = TransformersBTGenerator(
        base_model_path=args.base_model,
        lora_adapter_path=args.adapter,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    generator.generate_and_save(
        input_path=args.input_file,
        output_path=args.output_file,
        direction=args.direction,
        batch_size=args.batch_size,
        limit=args.max_samples,
    )


if __name__ == "__main__":
    main()

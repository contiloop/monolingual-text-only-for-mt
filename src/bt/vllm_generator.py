# project/src/bt/vllm_generator.py

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from pathlib import Path
import json
from typing import List, Optional
from tqdm import tqdm
import argparse

class VLLMBTGenerator:
    """vLLM 기반 고속 Back-Translation 생성기"""
    
    def __init__(
        self,
        base_model_path: str,
        lora_adapter_path: Optional[str] = None,
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
        dtype: str = "bfloat16"
    ):
        self.base_model_path = base_model_path
        self.lora_adapter_path = lora_adapter_path
        
        # vLLM 엔진 초기화
        print(f"Initializing vLLM with model: {base_model_path}")
        self.llm = LLM(
            model=base_model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            enable_lora=True if lora_adapter_path else False,
            max_lora_rank=64,
            dtype=dtype,
        )
        
        # LoRA 요청 준비
        self.lora_request = None
        if lora_adapter_path:
            print(f"Loading LoRA adapter from: {lora_adapter_path}")
            self.lora_request = LoRARequest(
                lora_name="translation_adapter",
                lora_int_id=1,
                lora_local_path=lora_adapter_path,
            )
        
        # 샘플링 설정 (Diversity를 위해 temperature 약간 높임)
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
            stop=["</s>", "\n\n", "<|endoftext|>"],
        )
    
    def generate_bt(
        self,
        source_texts: List[str],
        direction: str,  # "ko_to_en" or "en_to_ko"
        prompt_template: Optional[str] = None,
        glossary: Optional[dict] = None,
    ) -> List[dict]:
        """BT 생성"""
        
        # 프롬프트 구성
        prompts = []
        for text in source_texts:
            prompt = self._build_prompt(text, direction, prompt_template, glossary)
            prompts.append(prompt)
        
        # vLLM 배치 생성
        outputs = self.llm.generate(
            prompts,
            self.sampling_params,
            lora_request=self.lora_request,
            use_tqdm=False
        )
        
        # 결과 정리
        results = []
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            results.append({
                'source': source_texts[i],
                'generated': generated_text,
                'direction': direction,
            })
        
        return results
    
    def _build_prompt(
        self,
        text: str,
        direction: str,
        template: Optional[str] = None,
        glossary: Optional[dict] = None,
    ) -> str:
        """프롬프트 구성"""
        
        # 용어집 처리 (Glossary term injection)
        glossary_str = ""
        if glossary:
            terms = glossary.get(direction, {})
            found = []
            for term, trans in terms.items():
                if term.lower() in text.lower():
                    found.append(f"{term}={trans}")
            if found:
                # 상위 5개만 사용하여 프롬프트 오염 방지
                glossary_str = f"[Terms: {', '.join(found[:5])}] "
        
        # 방향에 따른 Task Prompt
        if direction == "ko_to_en":
            task = "다음 한국어를 영어로 번역하세요:"
        else:
            task = "Translate the following English to Korean:"
        
        # 템플릿 적용
        if template:
            return template.format(
                task=task,
                glossary=glossary_str,
                text=text,
            )
        
        # 기본 템플릿
        # 예: Translate... \n [Terms: A=B] Source \n Translation:
        return f"{task}\n{glossary_str}{text}\n번역:"
    
    def generate_and_save(
        self,
        input_path: str,
        output_path: str,
        direction: str,
        batch_size: int = 1000,
        glossary: Optional[dict] = None,
        limit: Optional[int] = None
    ):
        """파일에서 읽어 생성 후 저장"""
        
        # 입력 로드
        texts = []
        print(f"Loading input from {input_path}...")
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    texts.append(data['text'])
                    if limit and len(texts) >= limit:
                        break
        except FileNotFoundError:
            print(f"Error: Input file {input_path} not found.")
            return

        print(f"Generating BT for {len(texts)} samples (Direction: {direction})...")
        
        # 배치 처리
        all_results = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            results = self.generate_bt(batch_texts, direction, glossary=glossary)
            all_results.extend(results)
            
            # 중간 저장 (안정성)
            if i % (batch_size * 5) == 0:
                self._save_results(output_path, all_results, mode='w' if i==0 else 'a')
                all_results = [] # 메모리 비우기
        
        # 남은 것 저장
        if all_results:
            self._save_results(output_path, all_results, mode='a')
        
        print(f"Done! Saved to {output_path}")

    def _save_results(self, path: str, results: List[dict], mode: str = 'w'):
        with open(path, mode, encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--direction", type=str, choices=["ko_to_en", "en_to_ko"], required=True)
    parser.add_argument("--glossary_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=100)
    args = parser.parse_args()

    glossary = None
    if args.glossary_file:
        with open(args.glossary_file, 'r') as f:
            glossary = json.load(f)

    generator = VLLMBTGenerator(
        base_model_path=args.base_model,
        lora_adapter_path=args.adapter
    )

    generator.generate_and_save(
        input_path=args.input_file,
        output_path=args.output_file,
        direction=args.direction,
        batch_size=args.batch_size,
        glossary=glossary
    )

if __name__ == "__main__":
    main()

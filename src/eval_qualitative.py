# project/src/eval_qualitative.py

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import json
import argparse
from pathlib import Path
import pandas as pd

def run_eval(args):
    # Load Prompts
    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        prompt_data = json.load(f)
    
    prompts = [item['prompt'] for item in prompt_data]
    
    # Init vLLM
    print(f"Loading model: {args.base_model} with adapter {args.adapter}")
    llm = LLM(
        model=args.base_model,
        enable_lora=True if args.adapter else False,
        max_lora_rank=64,
        dtype="bfloat16"
    )
    
    sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
    
    lora_request = None
    if args.adapter:
        lora_request = LoRARequest("eval_adapter", 1, args.adapter)
        
    # Generate
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    
    # Collect Results
    results = []
    for i, output in enumerate(outputs):
        generated = output.outputs[0].text.strip()
        results.append({
            "category": prompt_data[i]['category'],
            "prompt": prompt_data[i]['prompt'],
            "generated": generated
        })
        
    # Save
    df = pd.DataFrame(results)
    print(df)
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_file = output_path / f"eval_results_{args.step}.csv"
    df.to_csv(save_file, index=False, encoding='utf-8-sig')
    print(f"Saved results to {save_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--prompt_file", default="data/eval/qualitative_prompts.json")
    parser.add_argument("--output_dir", default="outputs/evals")
    parser.add_argument("--step", default="0")
    args = parser.parse_args()
    
    run_eval(args)

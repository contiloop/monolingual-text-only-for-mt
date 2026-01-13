# project/src/train.py
"""
금융 번역 모델 학습 스크립트 (Accelerate 기반)
- Distributed: DDP / FSDP / DeepSpeed 자동 지원
- Phase 1: Denoising (L_auto)
- Phase 2: Back-Translation (L_back)
"""

import os
import sys
import json
import torch
import yaml
import argparse
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
from accelerate.utils import set_seed
import wandb
from tqdm import tqdm

# 프로젝트 루트 (src의 부모 = 루트)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import create_dataloader
from src.data.prompt_builder import PromptBuilder

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

class Trainer:
    def __init__(self, config):
        self.config = config
        
        # ===== Accelerator 초기화 =====
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config['training']['gradient_accumulation'],
            mixed_precision="bf16",
            log_with="wandb",  # 메인 프로세스만 실제로 로깅함
        )
        
        if self.accelerator.is_main_process:
            print(f"🌐 Accelerator: {self.accelerator.distributed_type}")
            print(f"   Device: {self.accelerator.device}")
            print(f"   Num processes: {self.accelerator.num_processes}")
        
        set_seed(42)
        
        # ===== Model & Tokenizer =====
        if self.is_main:
            print("📦 Loading Model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
        
        # Special Tokens 추가
        special_tokens = ['<|formal|>', '<|casual|>', '<|sep|>']
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # device_map="auto" 제거 - Accelerate가 처리
        self.model = AutoModelForCausalLM.from_pretrained(
            config['model']['name'],
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # Embedding 리사이즈
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # ===== LoRA =====
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config['model']['lora']['r'],
            lora_alpha=config['model']['lora']['alpha'],
            target_modules=config['model']['lora']['target_modules'],
            lora_dropout=config['model']['lora']['dropout'],
        )
        self.model = get_peft_model(self.model, lora_config)
        
        if self.is_main:
            self.model.print_trainable_parameters()
        
        # ===== Optimizer & Scheduler =====
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=float(config['training']['learning_rate'])
        )
        total_steps = config['training']['steps']
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)
        
        # ===== DataLoader =====
        if self.is_main:
            print("📊 Creating DataLoader...")
        
        ko_path = PROJECT_ROOT / config['data']['ko_processed_path']
        en_path = PROJECT_ROOT / config['data']['en_processed_path']
        glossary_path = PROJECT_ROOT / config['glossary']['path']
        
        self.dataloader, self.dataset, self.pseudo_buffer, self.hard_buffer = create_dataloader(
            config=config,
            tokenizer=self.tokenizer,
            ko_path=str(ko_path),
            en_path=str(en_path),
            glossary_path=str(glossary_path)
        )
        
        # ===== Accelerator로 준비 =====
        self.model, self.optimizer, self.dataloader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.dataloader, self.scheduler
        )
        
        # ===== PromptBuilder =====
        prompt_config = config.get('prompt', {})
        self.prompt_builder = PromptBuilder(prompt_config)
        
        # ===== State =====
        self.global_step = 0
        self.lback_active = False
    
    @property
    def is_main(self):
        """메인 프로세스인지 확인 (로깅용)"""
        return self.accelerator.is_main_process
    
    @property
    def device(self):
        return self.accelerator.device
        
    def compute_loss(self, batch):
        """L_auto와 L_back Loss 계산"""
        total_loss = 0.0
        loss_dict = {}
        
        # ===== L_auto (Denoising) =====
        if batch.has_auto and batch.auto_input_ids is not None:
            outputs = self.model(
                input_ids=batch.auto_input_ids,
                attention_mask=batch.auto_attention_mask,
                labels=batch.auto_labels
            )
            l_auto = outputs.loss
            total_loss += l_auto
            loss_dict['l_auto'] = l_auto.item()
        
        # ===== L_back (Translation) =====
        if batch.has_back and batch.back_input_ids is not None and self.lback_active:
            outputs = self.model(
                input_ids=batch.back_input_ids,
                attention_mask=batch.back_attention_mask,
                labels=batch.back_labels
            )
            l_back = outputs.loss
            
            beta = self.config['training']['loss'].get('beta', 0.5)
            total_loss += beta * l_back
            loss_dict['l_back'] = l_back.item()
        
        return total_loss, loss_dict
    
    def train(self):
        total_steps = self.config['training']['steps']
        
        # WandB 초기화 (메인 프로세스만)
        if self.is_main:
            wandb.init(
                project=self.config['project']['name'],
                config=self.config
            )
        
        # Progress bar (메인 프로세스만)
        pbar = tqdm(total=total_steps, desc="Training", disable=not self.is_main)
        data_iter = iter(self.dataloader)
        
        self.model.train()
        
        while self.global_step < total_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)
            
            # ===== Step 업데이트 =====
            self.dataset.set_step(self.global_step)
            
            # ===== L_back 활성화 체크 =====
            if not self.lback_active and self._should_activate_lback():
                self.lback_active = True
                self.dataset.set_lback_activated(True)
                if self.is_main:
                    print(f"\n🚀 L_back Activated at step {self.global_step}!")
                self._run_offline_bt()
            
            # ===== Forward & Backward (with gradient accumulation) =====
            with self.accelerator.accumulate(self.model):
                loss, loss_dict = self.compute_loss(batch)
                
                # Accelerate backward
                self.accelerator.backward(loss)
                
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # ===== Hard Example Mining =====
            if 'l_auto' in loss_dict and loss_dict['l_auto'] > self._get_hard_threshold():
                self.hard_buffer.add(
                    text="[placeholder]",
                    language="ko",
                    style_tag="",
                    loss=loss_dict['l_auto'],
                    step=self.global_step
                )
            
            # ===== Logging (메인만) =====
            if self.is_main:
                log_dict = {
                    "loss": loss.item(),
                    "lr": self.scheduler.get_last_lr()[0],
                    "step": self.global_step,
                    **loss_dict
                }
                wandb.log(log_dict)
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            pbar.update(1)
            self.global_step += 1
            
            # ===== Periodic BT Generation =====
            if self.lback_active and self.global_step % 5000 == 0:
                self._run_offline_bt()
            
            # ===== Checkpoint =====
            if self.global_step % 1000 == 0:
                self._save_checkpoint()
        
        pbar.close()
        self._save_checkpoint(final=True)
        
        if self.is_main:
            wandb.finish()
    
    def _should_activate_lback(self):
        """L_back 활성화 조건"""
        conds = self.config['training']['lback_activation']
        if self.global_step < conds.get('min_warmup_steps', 5000):
            return False
        return True
    
    def _get_hard_threshold(self):
        """Hard Example 임계값"""
        return self.hard_buffer.loss_threshold if len(self.hard_buffer) > 100 else 5.0
    
    def _run_offline_bt(self):
        """Offline BT (메인 프로세스만)"""
        if not self.is_main:
            return
            
        print(f"\n⏸️ BT Generation at step {self.global_step}...")
        ckpt_path = self._save_checkpoint()
        
        import subprocess
        bt_output = PROJECT_ROOT / self.config['data']['bt_cache_dir'] / f"bt_{self.global_step}.jsonl"
        
        cmd = [
            "python", str(PROJECT_ROOT / "src/bt/vllm_generator.py"),
            "--base_model", self.config['model']['name'],
            "--adapter", ckpt_path,
            "--input_file", str(PROJECT_ROOT / self.config['data']['ko_processed_path']),
            "--output_file", str(bt_output),
            "--direction", "ko_to_en",
            "--max_samples", "10000"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            self.dataset.reload_bt_cache(str(bt_output))
            print("▶️ Resuming training...")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ BT Generation failed: {e}")
    
    def _save_checkpoint(self, final=False):
        """체크포인트 저장 (unwrap 필요)"""
        self.accelerator.wait_for_everyone()
        
        if not self.is_main:
            return None
            
        output_dir = PROJECT_ROOT / self.config['project']['output_dir']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ckpt_path = output_dir / ("final" if final else f"ckpt_{self.global_step}")
        
        # unwrap_model로 원본 모델 추출
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(str(ckpt_path))
        self.tokenizer.save_pretrained(str(ckpt_path))
        
        print(f"💾 Saved: {ckpt_path}")
        return str(ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/base.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    trainer = Trainer(config)
    trainer.train()

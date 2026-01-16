# project/src/train.py
"""
금융 번역 모델 학습 스크립트 (Accelerate + Hydra 기반)
- Distributed: DDP / FSDP / DeepSpeed 자동 지원
- Phase 1: Denoising (L_auto)
- Phase 2: Back-Translation (L_back)
- Config: Hydra hierarchical config system
"""

import os
import sys
import json
import torch
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from accelerate import Accelerator
from accelerate.utils import set_seed
import wandb
from tqdm import tqdm

# Hydra
import hydra
from omegaconf import DictConfig, OmegaConf

# 프로젝트 루트 (src의 부모 = 루트)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import create_dataloader
from src.data.prompt_builder import PromptBuilder


def resolve_path(cfg_path: str) -> Path:
    """상대 경로를 프로젝트 루트 기준 절대 경로로 변환"""
    if os.path.isabs(cfg_path):
        return Path(cfg_path)
    return PROJECT_ROOT / cfg_path

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
        
        # Quantization 설정 (config에서 읽기)
        quant_config = config.get('model', {}).get('quantization', {})
        bnb_config = None
        
        if quant_config.get('load_in_4bit', False):
            # 4-bit QLoRA
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_config.get('bnb_4bit_quant_type', 'nf4'),
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
            if self.is_main:
                print("🔧 Using 4-bit quantization (QLoRA)")
        elif quant_config.get('load_in_8bit', False):
            # 8-bit
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            if self.is_main:
                print("🔧 Using 8-bit quantization")
        else:
            # Full precision
            if self.is_main:
                print("🔧 Using full precision (bf16)")
        
        # 모델 로드
        # DDP에서는 device_map='auto' 사용 불가 (각 프로세스가 자신의 GPU에만 로드)
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        is_distributed = self.accelerator.num_processes > 1
        
        if is_distributed:
            # DDP: 각 프로세스가 자신의 GPU에 로드
            model_kwargs = {
                'trust_remote_code': True,
                'device_map': {'': local_rank},  # 현재 프로세스의 GPU에만 로드
            }
        else:
            # 단일 GPU 또는 FSDP: auto 사용 가능
            model_kwargs = {
                'trust_remote_code': True,
                'device_map': 'auto',
            }
        
        # Attention 구현 선택 (Flash Attention 2 > SDPA > eager)
        def get_attn_implementation():
            # 1. Flash Attention 2 시도
            try:
                import flash_attn
                # GPU가 sm_80 이상(Ampere: A100, RTX 30xx, 40xx)인지 확인
                if torch.cuda.is_available():
                    capability = torch.cuda.get_device_capability()
                    if capability[0] >= 8:  # sm_80 이상
                        return 'flash_attention_2', "Flash Attention 2"
                    else:
                        return 'sdpa', f"SDPA (GPU sm_{capability[0]}{capability[1]} < sm_80)"
            except ImportError:
                pass
            
            # 2. SDPA fallback (PyTorch 2.0+)
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                return 'sdpa', "SDPA (Flash Attention not installed)"
            
            # 3. Eager fallback
            return 'eager', "Eager attention (legacy)"
        
        attn_impl, attn_msg = get_attn_implementation()
        model_kwargs['attn_implementation'] = attn_impl
        if self.is_main:
            print(f"⚡ Using {attn_msg}")
        
        if bnb_config:
            model_kwargs['quantization_config'] = bnb_config
        else:
            model_kwargs['torch_dtype'] = torch.bfloat16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config['model']['name'],
            **model_kwargs
        )
        
        # Embedding 리사이즈
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # gradient checkpointing과 호환을 위해 use_cache 비활성화
        self.model.config.use_cache = False
        
        # ===== LoRA =====
        # 4-bit/8-bit 모델은 gradient 준비 필요
        quant_config = config.get('model', {}).get('quantization', {})
        if quant_config.get('load_in_4bit') or quant_config.get('load_in_8bit'):
            self.model = prepare_model_for_kbit_training(
                self.model, 
                use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        
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
        
        # ===== Optimizer & Scheduler (with warmup) =====
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=float(config['training']['learning_rate'])
        )
        total_steps = config['training']['steps']
        warmup_ratio = config['training'].get('warmup_ratio', 0.05)
        warmup_steps = int(total_steps * warmup_ratio)
        
        # Warmup + Cosine Decay
        from transformers import get_cosine_schedule_with_warmup
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        if self.is_main:
            print(f"📈 LR Schedule: warmup {warmup_steps} steps → cosine decay")
        
        # ===== DataLoader =====
        if self.is_main:
            print("📊 Creating DataLoader...")
        
        ko_path = resolve_path(config['data']['ko_processed_path'])
        en_path = resolve_path(config['data']['en_processed_path'])
        glossary_path = resolve_path(config['data']['glossary']['path'])
        
        self.dataloader, self.dataset, self.pseudo_buffer, self.hard_buffer = create_dataloader(
            config=config,
            tokenizer=self.tokenizer,
            ko_path=str(ko_path),
            en_path=str(en_path),
            glossary_path=str(glossary_path)
        )
        
        # ===== Accelerator로 준비 =====
        # 양자화 모델은 device_map="auto"로 이미 GPU에 있으므로 model 제외
        self.optimizer, self.scheduler = self.accelerator.prepare(
            self.optimizer, self.scheduler
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
        device = self.accelerator.device
        
        # ===== L_auto (Denoising) =====
        if batch.has_auto and batch.auto_input_ids is not None:
            outputs = self.model(
                input_ids=batch.auto_input_ids.to(device),
                attention_mask=batch.auto_attention_mask.to(device),
                labels=batch.auto_labels.to(device)
            )
            l_auto = outputs.loss
            total_loss += l_auto
            loss_dict['l_auto'] = l_auto.item()
        
        # ===== L_back (Translation) =====
        if batch.has_back and batch.back_input_ids is not None and self.lback_active:
            outputs = self.model(
                input_ids=batch.back_input_ids.to(device),
                attention_mask=batch.back_attention_mask.to(device),
                labels=batch.back_labels.to(device)
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
        accumulation_step = 0
        
        # Early stopping 상태
        early_stop_config = self.config['training'].get('early_stopping', {})
        early_stop_enabled = early_stop_config.get('enabled', False)
        patience = early_stop_config.get('patience', 5)
        min_delta = early_stop_config.get('min_delta', 0.01)
        best_val_loss = float('inf')
        patience_counter = 0
        
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
            
            # ===== Forward & Backward (manual gradient accumulation) =====
            loss, loss_dict = self.compute_loss(batch)
            loss = loss / self.config['training']['gradient_accumulation']
            
            loss.backward()
            
            accumulation_step += 1
            
            # Gradient accumulation 완료 시 optimizer step
            if accumulation_step >= self.config['training']['gradient_accumulation']:
                # Gradient norm 계산 (clip 전)
                grad_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                accumulation_step = 0
                should_log = True
            else:
                should_log = False
                grad_norm = 0.0
            
            # ===== Hard Example Mining =====
            if 'l_auto' in loss_dict and loss_dict['l_auto'] > self._get_hard_threshold():
                self.hard_buffer.add(
                    text="[placeholder]",
                    language="ko",
                    style_tag="",
                    loss=loss_dict['l_auto'],
                    step=self.global_step
                )
            
            # ===== Logging & Step Update (only on actual optimizer step) =====
            if should_log:
                self.global_step += 1
                pbar.update(1)
                
                if self.is_main:
                    log_dict = {
                        "loss": loss.item() * self.config['training']['gradient_accumulation'],  # 원래 loss 복원
                        "lr": self.scheduler.get_last_lr()[0],
                        "step": self.global_step,
                        "gradient_norm": grad_norm,
                        **loss_dict
                    }
                    wandb.log(log_dict)
                    pbar.set_postfix(loss=f"{log_dict['loss']:.4f}", grad=f"{grad_norm:.2f}")
            
            # ===== Periodic Evaluation =====
            eval_interval = self.config['training'].get('eval_interval', 500)
            if self.global_step % eval_interval == 0 and self.global_step > 0:
                val_loss = self._evaluate(num_samples=100)
                
                # Early Stopping 체크
                if early_stop_enabled and val_loss is not None:
                    if val_loss < best_val_loss - min_delta:
                        best_val_loss = val_loss
                        patience_counter = 0
                        if self.is_main:
                            print(f"✅ Val loss improved to {val_loss:.4f}")
                    else:
                        patience_counter += 1
                        if self.is_main:
                            print(f"⚠️ No improvement ({patience_counter}/{patience})")
                        if patience_counter >= patience:
                            if self.is_main:
                                print(f"🛑 Early stopping at step {self.global_step}!")
                            break
            
            # ===== Periodic BT Generation =====
            if self.lback_active and self.global_step % 5000 == 0:
                self._run_offline_bt()
            
            # ===== Checkpoint =====
            if self.global_step % 1000 == 0 and self.global_step > 0:
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
        bt_output = resolve_path(self.config['data']['bt_cache_dir']) / f"bt_{self.global_step}.jsonl"
        
        cmd = [
            "python", str(PROJECT_ROOT / "src/bt/vllm_generator.py"),
            "--base_model", self.config['model']['name'],
            "--adapter", ckpt_path,
            "--input_file", str(resolve_path(self.config['data']['ko_processed_path'])),
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
            
        output_dir = resolve_path(self.config['project']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ckpt_path = output_dir / ("final" if final else f"ckpt_{self.global_step}")
        
        # unwrap_model로 원본 모델 추출
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(str(ckpt_path))
        self.tokenizer.save_pretrained(str(ckpt_path))
        
        print(f"💾 Saved: {ckpt_path}")
        return str(ckpt_path)
    
    def _evaluate(self, num_samples: int = 100):
        """Validation loss 계산 및 qualitative evaluation"""
        self.model.eval()
        val_losses = []
        
        # Validation pool에서 샘플링
        val_pool = self.dataset.pool.val_pool
        if not val_pool:
            self.model.train()
            return
        
        # 샘플 수 제한
        import random
        samples = random.sample(val_pool, min(num_samples, len(val_pool)))
        
        device = self.accelerator.device
        
        with torch.no_grad():
            for sample in samples:
                # 노이즈 적용
                noisy_text, _ = self.dataset.collator.noise_applier.apply(
                    sample.text, sample.language, sample.style_tag
                )
                
                # 토크나이징
                combined = f"{noisy_text} {self.tokenizer.eos_token} {sample.text}"
                enc = self.tokenizer(
                    combined,
                    truncation=True,
                    max_length=self.config['model']['max_seq_length'],
                    return_tensors='pt'
                )
                
                input_ids = enc.input_ids.to(device)
                attention_mask = enc.attention_mask.to(device)
                labels = input_ids.clone()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                val_losses.append(outputs.loss.item())
        
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
        
        if self.is_main:
            print(f"\n📊 Validation Loss: {avg_val_loss:.4f}")
            
            eval_log = {"val_loss": avg_val_loss, "step": self.global_step}
            
            # BLEU 평가 (병렬 코퍼스 있으면)
            bleu_score = self._evaluate_translation_bleu(num_samples=20)
            if bleu_score is not None:
                eval_log["bleu_ko_to_en"] = bleu_score
                print(f"🌐 Translation BLEU (ko→en): {bleu_score:.2f}")
            
            wandb.log(eval_log)
            
            # Qualitative: 3개 샘플 denoising 결과 출력
            print("📝 Qualitative Samples:")
            for i, sample in enumerate(samples[:3]):
                noisy, _ = self.dataset.collator.noise_applier.apply(
                    sample.text[:200], sample.language, sample.style_tag
                )
                print(f"  [{i+1}] Noisy: {noisy[:100]}...")
                print(f"      Original: {sample.text[:100]}...")
        
        self.model.train()
        return avg_val_loss
    
    def _evaluate_translation_bleu(self, num_samples: int = 20) -> float:
        """병렬 코퍼스로 번역 BLEU 계산 (ko→en)"""
        try:
            from datasets import load_from_disk
            import sacrebleu
        except ImportError:
            return None
        
        # 병렬 코퍼스 로드
        parallel_path = PROJECT_ROOT / "data/eval/korean_english_parallel"
        if not parallel_path.exists():
            return None
        
        try:
            dataset = load_from_disk(str(parallel_path))
            if hasattr(dataset, 'shuffle'):
                samples = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
            else:
                samples = dataset['train'].shuffle(seed=42).select(range(min(num_samples, len(dataset['train']))))
        except Exception:
            return None
        
        device = self.accelerator.device
        predictions = []
        references = []
        
        for item in samples:
            ko_text = item.get('korean', '')
            en_ref = item.get('english', '')
            
            if not ko_text or not en_ref:
                continue
            
            # 번역 (간단한 생성)
            prompt = f"Translate to English: {ko_text[:500]}"
            inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 프롬프트 제거
            if "Translate to English:" in generated:
                generated = generated.split("Translate to English:")[-1].strip()
            
            predictions.append(generated[:500])
            references.append(en_ref[:500])
        
        if not predictions:
            return None
        
        # BLEU 계산
        bleu = sacrebleu.corpus_bleu(predictions, [references])
        return bleu.score


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Hydra entry point
    
    Usage:
        python src/train.py                          # 기본 설정
        python src/train.py gpu=a100                 # GPU 프로파일 변경
        python src/train.py training.batch_size=2   # CLI override
        torchrun --nproc_per_node=4 src/train.py    # Multi-GPU
    """
    # DictConfig를 dict로 변환 (기존 코드 호환성)
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # 학습 시작
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

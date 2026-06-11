# sft.py
# Supervised Fine-Tuning (SFT) of TranslateGemma-4B-IT for Spanish-Valencian
# Paula Guerrero Castelló, May 2026
# ---------------------------------------------------------------------------

import os
import gc
import sacrebleu
import torch
import matplotlib.pyplot as plt
import transformers
import peft
import trl

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from huggingface_hub import login



# ------- Config -----------------------------------------------------------
hf_token = os.getenv("HF_TOKEN")
model_id = "google/translategemma-4b-it"
max_seq_length = 256
sft_train_samples = 50_000
output_root = "./outputs"
sft_run_dir = os.path.join(output_root, "sft")
sft_output_dir = os.path.join(sft_run_dir, "checkpoints")
sft_best_model_dir = os.path.join(sft_run_dir, "best_model")
sft_val_split = 0.02
sft_val_samples = 200
sft_val_every_steps = 100

source_lang_code = "es"
target_lang_code = "ca"
source_col = "ES"
target_col = "VA"

device = "cuda" if torch.cuda.is_available() else "cpu"
use_bf16 = torch.cuda.is_bf16_supported()


if hf_token:
    login(token=hf_token)

os.makedirs(sft_output_dir, exist_ok=True)
os.makedirs(sft_best_model_dir, exist_ok=True)



# --------- Model & tokenizer ---------------------------------------

print(f"PyTorch      : {torch.__version__}")
print(f"transformers : {transformers.__version__}")
print(f"peft         : {peft.__version__}")
print(f"trl          : {trl.__version__}")
print(f"CUDA         : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(f"VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.bfloat16 if use_bf16 else torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config = bnb_config,
    device_map          = "auto",
    token               = hf_token,
    dtype               = torch.bfloat16 if use_bf16 else torch.float16,
    trust_remote_code   = True,
)

print(f"Model  : {model_id}")
print(f"Params : {sum(p.numel() for p in base_model.parameters()) / 1e6:.0f}M")

base_model = prepare_model_for_kbit_training(base_model)

lora_config = LoraConfig(
    task_type      = TaskType.CAUSAL_LM,
    r              = 16,
    lora_alpha     = 32,
    lora_dropout   = 0.05,
    bias           = "none",
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()


# ------ Prompt template ---------------------------------------------------
def _make_messages(source_text: str) -> list:
    """Build the user-turn message list for the TranslateGemma template."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type"             : "text",
                    "source_lang_code" : source_lang_code,
                    "target_lang_code" : target_lang_code,
                    "text"             : source_text,
                }
            ],
        }
    ]


def format_for_sft(source_text: str, target_text: str) -> str:
    """Full prompt + reference answer; used to build the SFT dataset."""
    prompt = tokenizer.apply_chat_template(
        _make_messages(source_text), tokenize=False, add_generation_prompt=True
    )
    return prompt + target_text + tokenizer.eos_token


def make_inference_prompt(source_text: str) -> str:
    """Prompt only (no answer); used for inference and GRPO."""
    return tokenizer.apply_chat_template(
        _make_messages(source_text), tokenize=False, add_generation_prompt=True
    )



# ------- Dataset --------------------------------------------------------
raw_dataset = load_dataset("gplsi/amic_parallel")
dataset_split = raw_dataset["train"].train_test_split(test_size=sft_val_split, seed=42)
train_raw = dataset_split["train"]
val_raw = dataset_split["test"]
print(dataset_split)
print("\nExample:", train_raw[0])


def formatting_prompts_func(examples):
    return {"text": [
        format_for_sft(src, tgt)
        for src, tgt in zip(examples[source_col], examples[target_col])
    ]}


sft_train_dataset = train_raw.map(
    formatting_prompts_func,
    batched        = True,
    remove_columns = train_raw.column_names,
)
val_limit = min(sft_val_samples, len(val_raw))
val_samples = val_raw.select(range(val_limit))



# ------- Callbacks ------------------------------------------------

class LossPlotCallback(TrainerCallback):
    def __init__(self, save_path="sft_loss_curve.png"):
        self.save_path = save_path
        self.steps  = []
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.steps.append(state.global_step)
            self.losses.append(logs["loss"])

            plt.figure(figsize=(10, 4))
            plt.plot(self.steps, self.losses, linewidth=1.5, color="#2B5797")
            plt.title("SFT Training Loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.save_path, dpi=150, bbox_inches="tight")
            plt.close()


class BleuEvalSaveCallback(TrainerCallback):
    def __init__(self, tokenizer, model, eval_samples, save_dir, every_n_steps=100):
        self.tokenizer = tokenizer
        self.model = model
        self.eval_samples = eval_samples
        self.save_dir = save_dir
        self.every_n_steps = every_n_steps
        self.best_bleu = float("-inf")

    def _run_bleu_eval(self):
        hyps, refs = [], []
        self.model.eval()
        for sample in self.eval_samples:
            prompt = make_inference_prompt(sample[source_col])
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
            ).to(self.model.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            hyps.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
            refs.append(sample[target_col])
        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
        self.model.train()
        return bleu

    def _maybe_save_best(self, step):
        bleu = self._run_bleu_eval()
        print(f"[val] step={step:4d} | BLEU={bleu:.4f} | best={self.best_bleu:.4f}")
        if bleu > self.best_bleu:
            self.best_bleu = bleu
            self.model.save_pretrained(self.save_dir)
            self.tokenizer.save_pretrained(self.save_dir)
            print(f"[val] New best model saved to {self.save_dir}")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.every_n_steps != 0:
            return
        self._maybe_save_best(state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if self.best_bleu == float("-inf"):
            self._maybe_save_best(state.global_step)


class Gemma3DataCollator:
    """Wraps the standard causal-LM collator and injects token_type_ids
    (all zeros) required by Gemma 3's attention-mask function."""

    def __init__(self, tokenizer):
        self.base_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )

    def __call__(self, features):
        batch = self.base_collator(features)
        batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
        return batch



# ------ SFT training ---------------------------------------------------

model.train()

sft_trainer = SFTTrainer(
    model            = model,
    processing_class = tokenizer,
    train_dataset    = sft_train_dataset.shuffle(seed=42).select(
        range(min(sft_train_samples, len(sft_train_dataset)))
    ),
    data_collator    = Gemma3DataCollator(tokenizer),
    callbacks        = [
        LossPlotCallback("sft_loss_curve.png"),
        BleuEvalSaveCallback(
            tokenizer=tokenizer,
            model=model,
            eval_samples=val_samples,
            save_dir=sft_best_model_dir,
            every_n_steps=sft_val_every_steps,
        ),
    ],
    args = SFTConfig(
        packing                     = False,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 32,
        warmup_steps                = 25,
        max_steps                   = 2000,
        learning_rate               = 2e-4,
        logging_steps               = 25,
        optim                       = "paged_adamw_8bit",
        weight_decay                = 0.001,
        lr_scheduler_type           = "cosine",
        seed                        = 3407,
        output_dir                  = sft_output_dir,
        save_steps                  = 25,
        report_to                   = "none",
        fp16                        = not use_bf16,
        bf16                        = use_bf16,
        gradient_checkpointing      = True,
        dataloader_num_workers      = 2,
    ),
)

gpu_start = round(torch.cuda.max_memory_reserved() / 1e9, 2)
print(f"VRAM before training: {gpu_start} GB")

torch.cuda.empty_cache()
gc.collect()

sft_stats = sft_trainer.train()

print(f"\nSFT training complete")
print(f"VRAM max: {round(torch.cuda.max_memory_reserved()/1e9,2)} GB")
print(f"Time     : {sft_stats.metrics['train_runtime']:.1f}s")
print(f"Final Loss : {sft_stats.metrics.get('train_loss', 'N/A'):.4f}")

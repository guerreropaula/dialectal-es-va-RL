# 03_grpo_v1.py
# GRPO v1: chrF + HT/MT naturalness classifier
# Paula Guerrero Castelló, May 2026
# --------------------------------------------------------------------------

import os
import gc
import importlib
import sacrebleu
import torch
import torch.nn.functional as F
import transformers
import peft
import trl

from typing import List
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import PeftModel, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
from huggingface_hub import login

import transformers.models.gemma3.modeling_gemma3 as gemma3_module


# ---- Config -----------------------------------------------------------

hf_token = ""
base_model_id = "google/translategemma-4b-it"
sft_model_id = "guerreropaula/translategemma4b-sft-es-va"
clf_repo_id = "guerreropaula/ht_mt_classifier_best"
output_root = "./outputs"
grpo_run_dir = os.path.join(output_root, "grpov1")
grpo_output_dir = os.path.join(grpo_run_dir, "checkpoints")
grpo_best_model_dir = os.path.join(grpo_run_dir, "best_model")

source_lang_code = "es"
target_lang_code = "ca"
source_col = "ES"
target_col = "VA"

device = "cuda" if torch.cuda.is_available() else "cpu"
use_bf16 = torch.cuda.is_bf16_supported()

ht_label_idx = 1
clf_warmup_steps = 50
clf_weight_max = 0.3
total_steps = 100
grpo_train_samples = 5_000
grpo_val_split = 0.02
grpo_val_samples = 200
grpo_val_every_steps = 20

_reward_step_counter = {"step": 0}

if hf_token:
    login(token=hf_token)

os.makedirs(grpo_output_dir, exist_ok=True)
os.makedirs(grpo_best_model_dir, exist_ok=True)



# --- Model & tokenizer ------------------------------------------------------

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

tokenizer = AutoTokenizer.from_pretrained(sft_model_id, token=hf_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config = bnb_config,
    device_map          = "auto",
    token               = hf_token,
    torch_dtype         = torch.bfloat16 if use_bf16 else torch.float16,
    trust_remote_code   = True,
)
base_model = prepare_model_for_kbit_training(base_model)

model = PeftModel.from_pretrained(base_model, sft_model_id, is_trainable=True, token=hf_token)
model.print_trainable_parameters()



# --- Prompt Template ----------------------------------------------------------

def _make_messages(source_text: str) -> list:
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


def make_inference_prompt(source_text: str) -> str:
    return tokenizer.apply_chat_template(
        _make_messages(source_text), tokenize=False, add_generation_prompt=True
    )


# --- GRPO Dataset ------------------------------------------------------------

raw_dataset = load_dataset("gplsi/amic_parallel")
dataset_split = raw_dataset["train"].train_test_split(test_size=grpo_val_split, seed=42)
train_raw = dataset_split["train"]
val_raw = dataset_split["test"]


def make_grpo_example(examples):
    return {
        "prompt"   : [make_inference_prompt(src) for src in examples[source_col]],
        "reference": list(examples[target_col]),
    }


grpo_raw     = train_raw.shuffle(seed=123).select(range(min(grpo_train_samples, len(train_raw))))
grpo_dataset = grpo_raw.map(
    make_grpo_example,
    batched        = True,
    remove_columns = grpo_raw.column_names,
)
grpo_val_raw = val_raw.select(range(min(grpo_val_samples, len(val_raw))))
grpo_val_dataset = grpo_val_raw.map(
    make_grpo_example,
    batched=True,
    remove_columns=grpo_val_raw.column_names,
)


# --- Reward Functions ----------------------------------------------------------------

_clf_tok   = AutoTokenizer.from_pretrained(clf_repo_id)
_clf_model = AutoModelForSequenceClassification.from_pretrained(clf_repo_id)
_clf_model.eval().to(device)
print(f"Classifier : {clf_repo_id}")
print(f"Labels     : {_clf_model.config.id2label}")


def _clf_alpha() -> float:
    step = _reward_step_counter["step"]
    progress = min(1.0, step / max(1, clf_warmup_steps))
    return clf_weight_max * progress


@torch.no_grad()
def translationese_reward(texts: List[str], batch_size: int = 16) -> List[float]:
    rewards = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc   = _clf_tok(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(device)
        probs = F.softmax(_clf_model(**enc).logits, dim=-1)
        rewards.extend(probs[:, ht_label_idx].cpu().tolist())
    return rewards


def content_reward(hypotheses: List[str], references: List[str]) -> List[float]:
    rewards = []
    for hyp, ref in zip(hypotheses, references):
        if not hyp.strip() or not ref.strip():
            rewards.append(0.0)
        else:
            rewards.append(sacrebleu.sentence_chrf(hyp, [ref]).score / 100.0)
    return rewards


def composite_reward(hypotheses: List[str], references: List[str]) -> List[float]:
    alpha    = _clf_alpha()
    r_c_list = content_reward(hypotheses, references)
    if alpha == 0.0:
        return r_c_list
    r_t_list = translationese_reward(hypotheses)
    return [
        (1.0 - alpha) * r_c + alpha * r_t
        for r_c, r_t in zip(r_c_list, r_t_list)
    ]



# --- Reward wrapper ------------------------------------------------------- 

model.train()


def grpo_reward_fn(prompts, completions, reference=None, **kwargs):
    clean = [c.split("model\n")[-1].strip() for c in completions]
    _reward_step_counter["step"] += 1
    alpha = _clf_alpha()

    rewards = (
        content_reward(clean, [""] * len(clean))
        if reference is None
        else composite_reward(clean, list(reference))
    )

    mean_r = sum(rewards) / len(rewards)
    print(f"[reward] step={_reward_step_counter['step']:3d}  "
          f"alpha={alpha:.3f}  mean_reward={mean_r:.4f}")
    return rewards


class SampleLoggerCallback(TrainerCallback):
    def __init__(self, tokenizer, model, dataset, every_n_steps: int = 50, n_examples: int = 3):
        self.tokenizer  = tokenizer
        self.model      = model
        self.dataset    = dataset
        self.every_n    = every_n_steps
        self.n_examples = n_examples

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every_n != 0 or state.global_step == 0:
            return

        self.model.eval()
        indices = torch.randint(len(self.dataset), (self.n_examples,)).tolist()

        print(f"\n{'='*70}")
        print(f"  SAMPLES AT STEP {state.global_step}")
        print(f"{'='*70}")

        for idx in indices:
            sample = self.dataset[idx]
            inputs = self.tokenizer(
                sample["prompt"], return_tensors="pt", truncation=True, max_length=256
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens = 100,
                    do_sample      = False,
                    pad_token_id   = self.tokenizer.eos_token_id,
                )

            input_len = inputs["attention_mask"][0].sum().item()
            generated = self.tokenizer.decode(
                output_ids[0][input_len:], skip_special_tokens=True
            ).strip()

            src_text = sample["prompt"].split("\n\n")[-1].replace("<end_of_turn>", "").strip()
            chrf_val = sacrebleu.sentence_chrf(generated, [sample["reference"]]).score
            print(f"\n  SOURCE    : {src_text[:120]}")
            print(f"  REFERENCE : {sample['reference'][:120]}")
            print(f"  PREDICTION: {generated[:120]}")
            print(f"  chrF      : {chrf_val:.1f}")

        print(f"{'='*70}\n")
        self.model.train()


class RewardEvalSaveCallback(TrainerCallback):
    def __init__(self, tokenizer, model, eval_dataset, save_dir, every_n_steps=20):
        self.tokenizer = tokenizer
        self.model = model
        self.eval_dataset = eval_dataset
        self.save_dir = save_dir
        self.every_n_steps = every_n_steps
        self.best_reward = float("-inf")

    def _run_reward_eval(self):
        hyps, refs = [], []
        self.model.eval()
        for sample in self.eval_dataset:
            inputs = self.tokenizer(
                sample["prompt"], return_tensors="pt", truncation=True, max_length=256
            ).to(self.model.device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            input_len = inputs["attention_mask"][0].sum().item()
            hyp = self.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
            hyps.append(hyp)
            refs.append(sample["reference"])
        reward_values = []
        for hyp, ref in zip(hyps, refs):
            chrf_reward = 0.0 if not hyp.strip() or not ref.strip() else sacrebleu.sentence_chrf(hyp, [ref]).score / 100.0
            ht_reward = translationese_reward([hyp])[0] if hyp.strip() else 0.0
            reward_values.append((1.0 - clf_weight_max) * chrf_reward + clf_weight_max * ht_reward)
        mean_reward = sum(reward_values) / len(reward_values) if reward_values else 0.0
        self.model.train()
        return mean_reward

    def _maybe_save_best(self, step):
        reward = self._run_reward_eval()
        print(f"[val] step={step:4d} | reward={reward:.4f} | best={self.best_reward:.4f}")
        if reward > self.best_reward:
            self.best_reward = reward
            self.model.save_pretrained(self.save_dir)
            self.tokenizer.save_pretrained(self.save_dir)
            print(f"[val] New best model saved to {self.save_dir}")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.every_n_steps != 0:
            return
        self._maybe_save_best(state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if self.best_reward == float("-inf"):
            self._maybe_save_best(state.global_step)



# --- GRPO training -------------------------------------------------------------


grpo_config = GRPOConfig(
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,
    learning_rate               = 5e-6,
    max_steps                   = total_steps,
    warmup_steps                = 20,
    optim                       = "paged_adamw_8bit",
    weight_decay                = 0.01,
    lr_scheduler_type           = "cosine",
    gradient_checkpointing      = True,
    beta                        = 0.04,
    num_generations             = 2,
    max_completion_length       = 100,
    temperature                 = 0.9,
    output_dir                  = grpo_output_dir,
    logging_steps               = 1,
    save_steps                  = 20,
    seed                        = 3407,
    report_to                   = "none",
    fp16                        = not use_bf16,
    bf16                        = use_bf16,
)

grpo_trainer = GRPOTrainer(
    model            = model,
    processing_class = tokenizer,
    reward_funcs     = grpo_reward_fn,
    args             = grpo_config,
    train_dataset    = grpo_dataset,
)

grpo_trainer.add_callback(
    SampleLoggerCallback(tokenizer, model, grpo_dataset, every_n_steps=50, n_examples=3)
)
grpo_trainer.add_callback(
    RewardEvalSaveCallback(
        tokenizer=tokenizer,
        model=model,
        eval_dataset=grpo_val_dataset,
        save_dir=grpo_best_model_dir,
        every_n_steps=grpo_val_every_steps,
    )
)

grpo_stats = grpo_trainer.train()

print(f"\nGRPO training complete")
print(f"VRAM max: {round(torch.cuda.max_memory_reserved()/1e9,2)} GB")
print(f"Time     : {grpo_stats.metrics['train_runtime']:.1f}s")

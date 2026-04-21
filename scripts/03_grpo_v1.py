# =============================================================================
# 03_grpo_v1.py
# GRPO v1 — chrF + HT/MT Naturalness Classifier Reward
# Starting from SFT checkpoint: guerreropaula/translategemma4b-sft-es-va
# =============================================================================

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


# =============================================================================
# Config
# =============================================================================

HF_TOKEN       = ""
BASE_MODEL_ID  = "google/translategemma-4b-it"
SFT_MODEL_ID   = "guerreropaula/translategemma4b-sft-es-va2"
CLF_REPO_ID    = "guerreropaula/ht_mt_classifier_best"
GRPO_OUTPUT_DIR = "./translategemma4b_grpo_es_va"

SOURCE_LANG_CODE = "es"
TARGET_LANG_CODE = "ca"
SOURCE_COL       = "ES"
TARGET_COL       = "VA"

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = torch.cuda.is_bf16_supported()

HT_LABEL_IDX     = 1
CLF_WARMUP_STEPS = 50
CLF_WEIGHT_MAX   = 0.3
TOTAL_STEPS      = 100
GRPO_TRAIN_SAMPLES = 5_000

_reward_step_counter = {"step": 0}

login(token=HF_TOKEN)


# =============================================================================
# Model & Tokenizer
# =============================================================================

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
    bnb_4bit_compute_dtype    = torch.bfloat16 if USE_BF16 else torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_ID, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config = bnb_config,
    device_map          = "auto",
    token               = HF_TOKEN,
    torch_dtype         = torch.bfloat16 if USE_BF16 else torch.float16,
    trust_remote_code   = True,
)
base_model = prepare_model_for_kbit_training(base_model)

model = PeftModel.from_pretrained(base_model, SFT_MODEL_ID, is_trainable=True, token=HF_TOKEN)
model.print_trainable_parameters()


# =============================================================================
# Prompt Template
# =============================================================================

def _make_messages(source_text: str) -> list:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type"             : "text",
                    "source_lang_code" : SOURCE_LANG_CODE,
                    "target_lang_code" : TARGET_LANG_CODE,
                    "text"             : source_text,
                }
            ],
        }
    ]


def make_inference_prompt(source_text: str) -> str:
    return tokenizer.apply_chat_template(
        _make_messages(source_text), tokenize=False, add_generation_prompt=True
    )


# =============================================================================
# GRPO Dataset
# =============================================================================

raw_dataset = load_dataset("gplsi/amic_parallel")


def make_grpo_example(examples):
    return {
        "prompt"   : [make_inference_prompt(src) for src in examples[SOURCE_COL]],
        "reference": list(examples[TARGET_COL]),
    }


grpo_raw     = raw_dataset["train"].shuffle(seed=123).select(range(GRPO_TRAIN_SAMPLES))
grpo_dataset = grpo_raw.map(
    make_grpo_example,
    batched        = True,
    remove_columns = grpo_raw.column_names,
)


# =============================================================================
# Reward Functions
# =============================================================================

_clf_tok   = AutoTokenizer.from_pretrained(CLF_REPO_ID)
_clf_model = AutoModelForSequenceClassification.from_pretrained(CLF_REPO_ID)
_clf_model.eval().to(DEVICE)
print(f"Classifier : {CLF_REPO_ID}")
print(f"Labels     : {_clf_model.config.id2label}")


def _clf_alpha() -> float:
    step = _reward_step_counter["step"]
    if step < CLF_WARMUP_STEPS:
        return 0.0
    progress = min(1.0, (step - CLF_WARMUP_STEPS) / max(1, TOTAL_STEPS - CLF_WARMUP_STEPS))
    return CLF_WEIGHT_MAX * progress


@torch.no_grad()
def translationese_reward(texts: List[str], batch_size: int = 16) -> List[float]:
    rewards = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc   = _clf_tok(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(DEVICE)
        probs = F.softmax(_clf_model(**enc).logits, dim=-1)
        rewards.extend(probs[:, HT_LABEL_IDX].cpu().tolist())
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


# =============================================================================
# Gemma3 Patches
# =============================================================================

importlib.reload(gemma3_module)
gemma3_module._true_original_mask_fn = gemma3_module.create_causal_mask_mapping


def _patched_mask_fn(config, input_embeds, attention_mask, cache_position,
                     past_key_values, position_ids, token_type_ids=None,
                     pixel_values=None, is_training=False,
                     is_first_iteration=None, **kwargs):
    if is_training and token_type_ids is None:
        token_type_ids = torch.zeros(
            input_embeds.shape[:2], dtype=torch.long, device=input_embeds.device
        )
    return gemma3_module._true_original_mask_fn(
        config, input_embeds, attention_mask, cache_position,
        past_key_values, position_ids, token_type_ids,
        pixel_values, is_training, is_first_iteration, **kwargs
    )


gemma3_module.create_causal_mask_mapping = _patched_mask_fn

_ModelClass = type(model)
if not getattr(_ModelClass, "_forward_patched", False):
    _ModelClass._true_original_forward = _ModelClass.forward

    def _patched_forward(self, *args, **kwargs):
        if kwargs.get("token_type_ids") is None and "input_ids" in kwargs:
            kwargs["token_type_ids"] = torch.zeros_like(kwargs["input_ids"])
        return _ModelClass._true_original_forward(self, *args, **kwargs)

    _ModelClass.forward = _patched_forward
    _ModelClass._forward_patched = True


# =============================================================================
# Reward Wrapper & Callbacks
# =============================================================================

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


# =============================================================================
# GRPO Training
# =============================================================================

grpo_config = GRPOConfig(
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,
    learning_rate               = 5e-6,
    max_steps                   = TOTAL_STEPS,
    warmup_steps                = 20,
    optim                       = "paged_adamw_8bit",
    weight_decay                = 0.01,
    lr_scheduler_type           = "cosine",
    gradient_checkpointing      = True,
    beta                        = 0.04,
    num_generations             = 2,
    max_completion_length       = 100,
    temperature                 = 0.9,
    output_dir                  = GRPO_OUTPUT_DIR,
    logging_steps               = 1,
    save_steps                  = 20,
    seed                        = 3407,
    report_to                   = "none",
    fp16                        = False,
    bf16                        = True,
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

grpo_stats = grpo_trainer.train()

print(f"\nGRPO training complete")
print(f"VRAM max: {round(torch.cuda.max_memory_reserved()/1e9,2)} GB")
print(f"Time     : {grpo_stats.metrics['train_runtime']:.1f}s")


# =============================================================================
# Save & Push
# =============================================================================

MERGED_DIR = GRPO_OUTPUT_DIR + "_merged"

merged = model.merge_and_unload()
merged.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)
print(f"Merged model saved to: {MERGED_DIR}")

merged.push_to_hub("guerreropaula/translategemma4b-grpov1-es-va", token=HF_TOKEN)
tokenizer.push_to_hub("guerreropaula/translategemma4b-grpov1-es-va", token=HF_TOKEN)

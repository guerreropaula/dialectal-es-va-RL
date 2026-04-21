# =============================================================================
# 04_grpo_v2.py
# GRPO v2 — Composite Automatic-Metric Reward (chrF + COMET + TTR + copy penalty)
# Starting from SFT checkpoint: guerreropaula/translategemma4b-sft-es-va
# =============================================================================

import os
import gc
import json
import sacrebleu
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import transformers
import peft
import trl

from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer, TrainerCallback
from datasets import load_dataset as ld, load_dataset
from comet import download_model, load_from_checkpoint
from huggingface_hub import login


# =============================================================================
# Config
# =============================================================================

HF_TOKEN      = ""
BASE_MODEL_ID = "google/translategemma-4b-it"
SFT_MODEL_ID  = "guerreropaula/translategemma4b-sft-es-va2"
GRPO_HUB_ID   = "guerreropaula/translategemma4b-grpo2-es-va"
OUTPUT_DIR    = "./translategemma4b_grpo_v2"
BEST_CKPT     = "./translategemma4b_grpo/checkpoint-100"

SOURCE_LANG_CODE = "es"
TARGET_LANG_CODE = "ca"
SOURCE_COL       = "ES"
TARGET_COL       = "VA"

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = torch.cuda.is_bf16_supported()

N_TRAIN  = 10_000
W_CHRF   = 0.5
W_COMET  = 0.3
W_TTR    = 0.2

LOCAL_RANK     = int(os.environ.get("LOCAL_RANK", 0))
IS_DISTRIBUTED = dist.is_initialized()

login(token=HF_TOKEN)
print(f"device: {DEVICE} | bf16: {USE_BF16}")


# =============================================================================
# Model & Tokenizer
# =============================================================================

print(f"torch        : {torch.__version__}")
print(f"transformers : {transformers.__version__}")
print(f"peft         : {peft.__version__}")
print(f"trl          : {trl.__version__}")
print(f"cuda         : {torch.cuda.is_available()}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.bfloat16 if USE_BF16 else torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config = bnb_config,
    device_map          = "auto",
    token               = HF_TOKEN,
    torch_dtype         = torch.bfloat16 if USE_BF16 else torch.float16,
    trust_remote_code   = True,
)

model = PeftModel.from_pretrained(
    base_model,
    SFT_MODEL_ID,
    token        = HF_TOKEN,
    is_trainable = True,
)
model = prepare_model_for_kbit_training(model)
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

raw = load_dataset("gplsi/amic_parallel")


def preprocess(examples):
    return {
        "prompt"    : [make_inference_prompt(s) for s in examples["ES"]],
        "reference" : list(examples["VA"]),
        "source_es" : list(examples["ES"]),
    }


grpo_dataset = (
    raw["train"]
    .shuffle(seed=42)
    .select(range(N_TRAIN))
    .map(preprocess, batched=True, remove_columns=raw["train"].column_names)
)


# =============================================================================
# COMET
# =============================================================================

comet_model = None
if LOCAL_RANK == 0:
    _path       = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(_path).to("cuda")

if IS_DISTRIBUTED:
    dist.barrier()


# =============================================================================
# Reward Functions
# =============================================================================

def chrf_score(hypothesis: str, reference: str) -> float:
    if not hypothesis or not reference:
        return 0.0
    return sacrebleu.sentence_chrf(hypothesis, [reference]).score / 100.0


def ttr_score(hypothesis: str) -> float:
    if not hypothesis:
        return 0.0
    tokens = hypothesis.lower().split()
    if not tokens:
        return 0.0
    ttr = len(set(tokens)) / len(tokens)
    if len(tokens) < 5:
        ttr *= len(tokens) / 5.0
    return float(ttr)


def copy_penalty(source: str, hypothesis: str) -> float:
    if not source or not hypothesis:
        return 0.0
    src = source.strip().lower()
    hyp = hypothesis.strip().lower()
    if src == hyp:
        return -1.0
    sim = sacrebleu.sentence_chrf(hyp, [src]).score / 100.0
    THRESHOLD = 0.7
    if sim > THRESHOLD:
        return -(sim - THRESHOLD) / (1.0 - THRESHOLD)
    return 0.0


def _comet_batch(sources: List[str], hyps: List[str], refs: List[str]) -> List[float]:
    if comet_model is None:
        return [0.5] * len(hyps)
    data   = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hyps, refs)]
    output = comet_model.predict(data, batch_size=8, gpus=0)
    scores = output.scores if hasattr(output, "scores") else output[0]

    if IS_DISTRIBUTED:
        t = torch.tensor(scores, dtype=torch.float32, device="cuda")
        dist.broadcast(t, src=0)
        scores = t.cpu().tolist()
    return scores


def composite_reward(
    completions: List[str],
    reference:   List[str],
    source_es:   List[str],
    **kwargs,
) -> List[float]:
    comet_scores = _comet_batch(source_es, completions, reference)

    rewards = []
    for hyp, ref, src, c_s in zip(completions, reference, source_es, comet_scores):
        hyp = hyp.strip() if isinstance(hyp, str) else ""
        r = (
            W_CHRF  * chrf_score(hyp, ref)
          + W_COMET * c_s
          + W_TTR   * ttr_score(hyp)
          + copy_penalty(src, hyp)
        )
        rewards.append(float(r))
    return rewards


# =============================================================================
# Training Callback
# =============================================================================

class RewardPlotCallback(TrainerCallback):
    def __init__(self, save_path="grpo_reward_curve.png"):
        self.save_path = save_path
        self.steps   = []
        self.rewards = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "reward" in logs:
            self.steps.append(state.global_step)
            self.rewards.append(logs["reward"])

            plt.figure(figsize=(10, 4))
            plt.plot(self.steps, self.rewards, lw=1.5, color="#1D9E75")
            plt.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.4)
            plt.xlabel("step")
            plt.ylabel("mean reward")
            plt.title("GRPO reward")
            plt.grid(alpha=0.25)
            plt.tight_layout()
            plt.savefig(self.save_path, dpi=150, bbox_inches="tight")
            plt.close()


# =============================================================================
# GRPO Training
# =============================================================================

grpo_config = GRPOConfig(
    max_completion_length       = 128,
    num_generations             = 4,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 16,
    max_steps                   = 200,
    learning_rate               = 5e-6,
    warmup_steps                = 20,
    lr_scheduler_type           = "cosine",
    optim                       = "paged_adamw_8bit",
    weight_decay                = 0.01,
    beta                        = 0.04,
    epsilon                     = 0.2,
    bf16                        = USE_BF16,
    fp16                        = not USE_BF16,
    gradient_checkpointing      = True,
    output_dir                  = OUTPUT_DIR,
    logging_steps               = 10,
    save_steps                  = 10,
    report_to                   = "none",
    seed                        = 42,
)

trainer = GRPOTrainer(
    model            = model,
    processing_class = tokenizer,
    reward_funcs     = composite_reward,
    args             = grpo_config,
    train_dataset    = grpo_dataset,
    callbacks        = [RewardPlotCallback("grpo_reward_curve.png")],
)

torch.cuda.empty_cache()
gc.collect()

stats = trainer.train()
print(f"Training complete. Final mean reward: {stats.metrics.get('train_reward', 'N/A')}")


# =============================================================================
# Checkpoint Summary
# =============================================================================

checkpoints = sorted([
    d for d in os.listdir(OUTPUT_DIR)
    if d.startswith("checkpoint-")
], key=lambda x: int(x.split("-")[1]))

for ckpt in checkpoints:
    state_path = os.path.join(OUTPUT_DIR, ckpt, "trainer_state.json")
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        last_log = state["log_history"][-1]
        step   = last_log.get("step", "?")
        reward = last_log.get("reward", "N/A")
        print(f"{ckpt}  →  step {step}  reward {reward}")


# =============================================================================
# Save & Push Best Checkpoint
# =============================================================================

base_model_reload = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype = torch.bfloat16 if USE_BF16 else torch.float16,
    device_map  = "cuda",
)

peft_model    = PeftModel.from_pretrained(base_model_reload, BEST_CKPT)
merged_model  = peft_model.merge_and_unload()

merged_model.push_to_hub(
    "guerreropaula/translategemma4b-grpov2-es-va",
    safe_serialization = True,
    max_shard_size     = "2GB",
)
tokenizer.push_to_hub("guerreropaula/translategemma4b-grpov2-es-va")


# =============================================================================
# Quick Evaluation
# =============================================================================

N_QUICK = 100
test_ds = ld("gplsi/ES-VA_translation_test", split="test").select(range(N_QUICK))
gold_es = [ex["es"] for ex in test_ds]
gold_va = [ex["va"] for ex in test_ds]

merged_model.eval()
tokenizer.padding_side = "left"
hyps = []

for src in gold_es:
    prompt = make_inference_prompt(src)
    enc    = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = merged_model.generate(**enc, max_new_tokens=128, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if "model" in text:
        text = text.split("model")[-1].strip()
    hyps.append(text)

chrf = sacrebleu.corpus_chrf(hyps, [gold_va]).score
bleu = sacrebleu.corpus_bleu(hyps, [gold_va]).score
print(f"Quick eval ({N_QUICK} sents) — chrF: {chrf:.2f} | BLEU: {bleu:.2f}")

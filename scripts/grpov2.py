# grpov2.py
# GRPO v2: composite metric Reward (chrF + COMET + TTR + copy penalty)
# Paula Guerrero Castelló, May 2026
# --------------------------------------------------------------------------

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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
from peft import PeftModel, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset as ld, load_dataset
from comet import download_model, load_from_checkpoint
from huggingface_hub import login



# --- Config ------------------------------------------------------------

hf_token = os.getenv("HF_TOKEN")
base_model_id = "google/translategemma-4b-it"
sft_model_id  = "guerreropaula/translategemma4b-sft-es-va"
grpo_hub_id   = "guerreropaula/translategemma4b-grpov2-es-va"
output_root   = "./outputs"
grpo_run_dir  = os.path.join(output_root, "grpov2")
output_dir    = os.path.join(grpo_run_dir, "checkpoints")
best_model_dir = os.path.join(grpo_run_dir, "best_model")

source_lang_code = "es"
target_lang_code = "ca"
source_col       = "ES"
target_col       = "VA"

device   = "cuda" if torch.cuda.is_available() else "cpu"
use_bf16 = torch.cuda.is_bf16_supported()

n_train  = 10_000
w_chrf   = 0.5
w_comet  = 0.3
w_ttr    = 0.2
grpo_val_split = 0.02
grpo_val_samples = 200
grpo_val_every_steps = 20

local_rank     = int(os.environ.get("LOCAL_RANK", 0))
is_distributed = dist.is_initialized()

if hf_token:
    login(token=hf_token)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(best_model_dir, exist_ok=True)
print(f"device: {device} | bf16: {use_bf16}")


# --- Model & tokenizer ------------------------------------------------------------

print(f"torch        : {torch.__version__}")
print(f"transformers : {transformers.__version__}")
print(f"peft         : {peft.__version__}")
print(f"trl          : {trl.__version__}")
print(f"cuda         : {torch.cuda.is_available()}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.bfloat16 if use_bf16 else torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config = bnb_config,
    device_map          = "auto",
    token               = hf_token,
    torch_dtype         = torch.bfloat16 if use_bf16 else torch.float16,
    trust_remote_code   = True,
)

model = PeftModel.from_pretrained(
    base_model,
    sft_model_id,
    token        = hf_token,
    is_trainable = True,
)
model = prepare_model_for_kbit_training(model)
model.print_trainable_parameters()


# --- Prompt template ------------------------------------------------------------

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

raw = load_dataset("gplsi/amic_parallel")
dataset_split = raw["train"].train_test_split(test_size=grpo_val_split, seed=42)
train_raw = dataset_split["train"]
val_raw = dataset_split["test"]


def preprocess(examples):
    return {
        "prompt"    : [make_inference_prompt(s) for s in examples["ES"]],
        "reference" : list(examples["VA"]),
        "source_es" : list(examples["ES"]),
    }


grpo_dataset = (
    train_raw
    .shuffle(seed=42)
    .select(range(min(n_train, len(train_raw))))
    .map(preprocess, batched=True, remove_columns=train_raw.column_names)
)
grpo_val_dataset = (
    val_raw
    .select(range(min(grpo_val_samples, len(val_raw))))
    .map(preprocess, batched=True, remove_columns=val_raw.column_names)
)


# --- COMET ------------------------------------------------------------

comet_model = None
if local_rank == 0:
    _path       = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(_path).to("cuda")

if is_distributed:
    dist.barrier()


# --- Reward functions------------------------------------------------------------

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
    threshold = 0.7
    if sim > threshold:
        return -(sim - threshold) / (1.0 - threshold)
    return 0.0


def _comet_batch(sources: List[str], hyps: List[str], refs: List[str]) -> List[float]:
    if comet_model is None:
        return [0.5] * len(hyps)
    data   = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hyps, refs)]
    output = comet_model.predict(data, batch_size=8, gpus=0)
    scores = output.scores if hasattr(output, "scores") else output[0]

    if is_distributed:
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
            w_chrf  * chrf_score(hyp, ref)
          + w_comet * c_s
          + w_ttr   * ttr_score(hyp)
          + copy_penalty(src, hyp)
        )
        rewards.append(float(r))
    return rewards


# --- Callback ------------------------------------------------------------

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


class RewardEvalSaveCallback(TrainerCallback):
    def __init__(self, tokenizer, model, eval_dataset, save_dir, every_n_steps=20):
        self.tokenizer = tokenizer
        self.model = model
        self.eval_dataset = eval_dataset
        self.save_dir = save_dir
        self.every_n_steps = every_n_steps
        self.best_reward = float("-inf")

    def _run_reward_eval(self):
        completions, refs, sources = [], [], []
        self.model.eval()
        for sample in self.eval_dataset:
            enc = self.tokenizer(sample["prompt"], return_tensors="pt").to(device)
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            text = self.tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            completions.append(text)
            refs.append(sample["reference"])
            sources.append(sample["source_es"])
        rewards = composite_reward(completions, refs, sources)
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
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



# --- GRPO Training ------------------------------------------------------------

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
    bf16                        = use_bf16,
    fp16                        = not use_bf16,
    gradient_checkpointing      = True,
    output_dir                  = output_dir,
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
    callbacks        = [
        RewardPlotCallback("grpo_reward_curve.png"),
        RewardEvalSaveCallback(
            tokenizer=tokenizer,
            model=model,
            eval_dataset=grpo_val_dataset,
            save_dir=best_model_dir,
            every_n_steps=grpo_val_every_steps,
        ),
    ],
)

torch.cuda.empty_cache()
gc.collect()

stats = trainer.train()
print(f"Training complete. Final mean reward: {stats.metrics.get('train_reward', 'N/A')}")


# --- Checkpoint summary ------------------------------------------------------------

checkpoints = sorted([
    d for d in os.listdir(output_dir)
    if d.startswith("checkpoint-")
], key=lambda x: int(x.split("-")[1]))

for ckpt in checkpoints:
    state_path = os.path.join(output_dir, ckpt, "trainer_state.json")
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        last_log = state["log_history"][-1]
        step   = last_log.get("step", "?")
        reward = last_log.get("reward", "N/A")
        print(f"{ckpt}  →  step {step}  reward {reward}")



# -- Quick Evaluation ------------------------------------------------------------

n_quick = 100
test_ds = ld("gplsi/ES-VA_translation_test", split="test").select(range(n_quick))
gold_es = [ex["es"] for ex in test_ds]
gold_va = [ex["va"] for ex in test_ds]

model.eval()
tokenizer.padding_side = "left"
hyps = []

for src in gold_es:
    prompt = make_inference_prompt(src)
    enc    = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=128, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if "model" in text:
        text = text.split("model")[-1].strip()
    hyps.append(text)

chrf = sacrebleu.corpus_chrf(hyps, [gold_va]).score
bleu = sacrebleu.corpus_bleu(hyps, [gold_va]).score
print(f"Quick eval ({n_quick} sents) — chrF: {chrf:.2f} | BLEU: {bleu:.2f}")

# utils/callbacks.py
# Reusable TrainerCallback implementations shared across SFT and GRPO training.
# Paula Guerrero Castelló, May 2026

import torch
import sacrebleu
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Union
from transformers import TrainerCallback


# ── Loss / reward curve plot ──────────────────────────────────────────────────

class LossPlotCallback(TrainerCallback):
    """Saves a PNG of training loss after every logged step."""

    def __init__(self, save_path: Union[str, Path] = "training_loss.png", metric: str = "loss"):
        self.save_path = Path(save_path)
        self.metric    = metric
        self.steps:  list = []
        self.values: list = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or self.metric not in logs:
            return
        self.steps.append(state.global_step)
        self.values.append(logs[self.metric])
        self._plot()

    def _plot(self):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.steps, self.values, linewidth=1.5, color="#2B5797")
        ax.set_xlabel("Step")
        ax.set_ylabel(self.metric.capitalize())
        ax.set_title(f"Training {self.metric.capitalize()}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_path, dpi=150, bbox_inches="tight")
        plt.close()


class RewardPlotCallback(TrainerCallback):
    """Saves a PNG of mean GRPO reward after every logged step."""

    def __init__(self, save_path: Union[str, Path] = "grpo_reward_curve.png"):
        self.save_path = Path(save_path)
        self.steps:   list = []
        self.rewards: list = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "reward" not in logs:
            return
        self.steps.append(state.global_step)
        self.rewards.append(logs["reward"])
        self._plot()

    def _plot(self):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.steps, self.rewards, lw=1.5, color="#1D9E75")
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.4)
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean reward")
        ax.set_title("GRPO Reward")
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(self.save_path, dpi=150, bbox_inches="tight")
        plt.close()


# ── BLEU-based best-model saver (SFT) ────────────────────────────────────────

class BleuEvalSaveCallback(TrainerCallback):
    """
    Runs BLEU on a small validation set every N steps.
    Saves the model whenever a new best is reached.
    """

    def __init__(self, tokenizer, model, eval_samples, save_dir: Path,
                 make_prompt_fn, cfg, every_n_steps: int = 100):
        self.tokenizer       = tokenizer
        self.model           = model
        self.eval_samples    = eval_samples
        self.save_dir        = Path(save_dir)
        self.make_prompt_fn  = make_prompt_fn   # (src, tok, cfg) → str
        self.cfg             = cfg
        self.every_n_steps   = every_n_steps
        self.best_bleu       = float("-inf")

    def _eval(self) -> float:
        hyps, refs = [], []
        self.model.eval()
        for sample in self.eval_samples:
            prompt = self.make_prompt_fn(sample[self.cfg.source_col], self.tokenizer, self.cfg)
            inputs = self.tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=self.cfg.sft_max_seq_length,
            ).to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs, max_new_tokens=128, do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            new_tok = out[0][inputs["input_ids"].shape[1]:]
            hyps.append(self.tokenizer.decode(new_tok, skip_special_tokens=True).strip())
            refs.append(sample[self.cfg.target_col])
        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
        self.model.train()
        return bleu

    def _maybe_save(self, step: int):
        bleu = self._eval()
        print(f"[val] step={step:4d} | BLEU={bleu:.4f} | best={self.best_bleu:.4f}")
        if bleu > self.best_bleu:
            self.best_bleu = bleu
            self.model.save_pretrained(self.save_dir)
            self.tokenizer.save_pretrained(self.save_dir)
            print(f"[val] New best SFT model saved → {self.save_dir}")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step and state.global_step % self.every_n_steps == 0:
            self._maybe_save(state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if self.best_bleu == float("-inf"):
            self._maybe_save(state.global_step)


# ── Reward-based best-model saver (GRPO) ─────────────────────────────────────

class RewardEvalSaveCallback(TrainerCallback):
    """
    Computes mean reward on a small validation set every N steps.
    Saves the model whenever a new best is reached.

    `reward_fn` must have the signature:
        reward_fn(completions, reference, source_es, **kwargs) → List[float]
    """

    def __init__(self, tokenizer, model, eval_dataset, save_dir: Path,
                 reward_fn, cfg, every_n_steps: int = 20):
        self.tokenizer     = tokenizer
        self.model         = model
        self.eval_dataset  = eval_dataset
        self.save_dir      = Path(save_dir)
        self.reward_fn     = reward_fn
        self.cfg           = cfg
        self.every_n_steps = every_n_steps
        self.best_reward   = float("-inf")
        self._device       = "cuda" if torch.cuda.is_available() else "cpu"

    def _run_eval(self) -> float:
        completions, refs, sources = [], [], []
        self.model.eval()
        for sample in self.eval_dataset:
            enc = self.tokenizer(sample["prompt"], return_tensors="pt").to(self._device)
            with torch.no_grad():
                out = self.model.generate(
                    **enc, max_new_tokens=self.cfg.grpo_max_new,
                    do_sample=False, pad_token_id=self.tokenizer.pad_token_id,
                )
            text = self.tokenizer.decode(
                out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            completions.append(text)
            refs.append(sample["reference"])
            sources.append(sample.get("source_es", ""))
        rewards = self.reward_fn(completions, reference=refs, source_es=sources)
        self.model.train()
        return sum(rewards) / len(rewards) if rewards else 0.0

    def _maybe_save(self, step: int):
        reward = self._run_eval()
        print(f"[val] step={step:4d} | reward={reward:.4f} | best={self.best_reward:.4f}")
        if reward > self.best_reward:
            self.best_reward = reward
            self.model.save_pretrained(self.save_dir)
            self.tokenizer.save_pretrained(self.save_dir)
            print(f"[val] New best GRPO model saved → {self.save_dir}")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step and state.global_step % self.every_n_steps == 0:
            self._maybe_save(state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if self.best_reward == float("-inf"):
            self._maybe_save(state.global_step)

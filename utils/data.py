# utils/data.py
# Dataset loading and prompt-building helpers shared by SFT, GRPO, and eval.
# Paula Guerrero Castelló, May 2026

from typing import Optional
from datasets import load_dataset
from config import Config


# ── Prompt template ───────────────────────────────────────────────────────────

def _make_messages(source_text: str, cfg: Config) -> list:
    """Build the single user-turn needed by TranslateGemma's chat template."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type":              "text",
                    "source_lang_code":  cfg.source_lang_code,
                    "target_lang_code":  cfg.target_lang_code,
                    "text":              source_text,
                }
            ],
        }
    ]


def make_inference_prompt(source_text: str, tokenizer, cfg: Config) -> str:
    """Prompt only — no answer appended. Used for inference and GRPO."""
    return tokenizer.apply_chat_template(
        _make_messages(source_text, cfg),
        tokenize=False,
        add_generation_prompt=True,
    )


def make_sft_example(source_text: str, target_text: str, tokenizer, cfg: Config) -> str:
    """Full prompt + reference answer + EOS. Used to build the SFT dataset."""
    prompt = make_inference_prompt(source_text, tokenizer, cfg)
    return prompt + target_text + tokenizer.eos_token


def make_prompt(source_text: str, tokenizer, cfg: Config) -> str:
    """Compatibility alias for the inference prompt builder."""
    return make_inference_prompt(source_text, tokenizer, cfg)


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_amic(cfg: Config, val_split: Optional[float] = None):
    """
    Load gplsi/amic_parallel.

    Returns
    -------
    train_raw, val_raw  — if val_split is given
    train_raw           — otherwise (full training split)
    """
    raw = load_dataset("gplsi/amic_parallel")
    if val_split is not None:
        split = raw["train"].train_test_split(test_size=val_split, seed=42)
        return split["train"], split["test"]
    return raw["train"]


def load_amic_parallel(cfg: Config, val_split: Optional[float] = None):
    """Compatibility alias for the refactored AMIC loader."""
    return load_amic(cfg, val_split=val_split)


def load_test_set(cfg: Config):
    """Load the 1k ES-VA evaluation test set."""
    ds = load_dataset("gplsi/ES-VA_translation_test", split="test")
    # Sort longest-first (matches original evaluate.py behaviour)
    ds = ds.map(lambda x: {"len": len(x[cfg.eval_src_col])})
    ds = ds.sort("len", reverse=True)
    ds = ds.select(range(cfg.eval_n))
    return (
        [ex[cfg.eval_src_col] for ex in ds],
        [ex[cfg.eval_tgt_col] for ex in ds],
    )


# ── SFT dataset builder ───────────────────────────────────────────────────────

def build_sft_dataset(train_raw, tokenizer, cfg: Config):
    """Format training examples as full prompt+answer strings for SFTTrainer."""
    def _format(examples):
        return {"text": [
            make_sft_example(src, tgt, tokenizer, cfg)
            for src, tgt in zip(examples[cfg.source_col], examples[cfg.target_col])
        ]}
    return train_raw.map(_format, batched=True, remove_columns=train_raw.column_names)


# ── GRPO dataset builder ──────────────────────────────────────────────────────

def build_grpo_dataset(raw, tokenizer, cfg: Config, n: int):
    """Format examples for GRPOTrainer: prompt + reference + source_es."""
    def _format(examples):
        return {
            "prompt":    [make_inference_prompt(s, tokenizer, cfg) for s in examples[cfg.source_col]],
            "reference": list(examples[cfg.target_col]),
            "source_es": list(examples[cfg.source_col]),
        }
    return (
        raw.shuffle(seed=42)
           .select(range(min(n, len(raw))))
           .map(_format, batched=True, remove_columns=raw.column_names)
    )


def preprocess(raw, tokenizer, cfg: Config, n: int):
    """Compatibility alias for GRPO preprocessing."""
    return build_grpo_dataset(raw, tokenizer, cfg, n)

#!/usr/bin/env python
# main.py
# Single entry point for all ES-VA MT experiments.
#
# Usage examples:
#   python main.py --mode sft
#   python main.py --mode classifier
#   python main.py --mode grpov1
#   python main.py --mode grpov2
#   python main.py --mode eval
#
# Override any Config field on the command line:
#   python main.py --mode sft --sft_train_samples 10000 --sft_lr 1e-4
#
# Paula Guerrero Castelló, May 2026

import argparse
import os
import sys
import torch
import transformers
import peft
import trl

from config import Config
from huggingface_hub import login


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ES-VA MT experiments: SFT, GRPO, classifier, evaluation."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["sft", "classifier", "grpov1", "grpov2", "eval"],
        help="Which experiment to run.",
    )

    # Allow overriding any numeric/string Config field
    parser.add_argument("--hf_token",             default=None)
    parser.add_argument("--sft_train_samples",    type=int,   default=None)
    parser.add_argument("--sft_lr",               type=float, default=None)
    parser.add_argument("--sft_max_steps",        type=int,   default=None)
    parser.add_argument("--grpov1_train_samples", type=int,   default=None)
    parser.add_argument("--grpov1_max_steps",     type=int,   default=None)
    parser.add_argument("--grpov2_train_samples", type=int,   default=None)
    parser.add_argument("--grpov2_max_steps",     type=int,   default=None)
    parser.add_argument("--eval_n",               type=int,   default=None)
    parser.add_argument("--w_chrf",               type=float, default=None)
    parser.add_argument("--w_comet",              type=float, default=None)
    parser.add_argument("--w_ttr",                type=float, default=None)

    return parser.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Patch Config with any non-None CLI arguments."""
    for field in vars(args):
        if field == "mode":
            continue
        val = getattr(args, field)
        if val is not None and hasattr(cfg, field):
            setattr(cfg, field, val)
    return cfg


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = apply_overrides(Config(), args)

    # Auth
    if cfg.hf_token:
        login(token=cfg.hf_token)

    # Environment info
    print(f"{'='*55}")
    print(f"  Mode         : {args.mode}")
    print(f"  PyTorch      : {torch.__version__}")
    print(f"  transformers : {transformers.__version__}")
    print(f"  peft         : {peft.__version__}")
    print(f"  trl          : {trl.__version__}")
    print(f"  CUDA         : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU          : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*55}\n")

    if args.mode == "sft":
        from trainers.sft_trainer import run_sft
        run_sft(cfg)

    elif args.mode == "classifier":
        from trainers.classifier_trainer import run_classifier
        run_classifier(cfg)

    elif args.mode == "grpov1":
        from trainers.grpo_v1_trainer import run_grpo_v1
        run_grpo_v1(cfg)

    elif args.mode == "grpov2":
        from trainers.grpo_v2_trainer import run_grpo_v2
        run_grpo_v2(cfg)

    elif args.mode == "eval":
        from evaluate import run_evaluation
        run_evaluation(cfg)


if __name__ == "__main__":
    main()

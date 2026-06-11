# trainers/grpo_v1_trainer.py
# GRPOv1: SFT model continued with chrF + HT/MT naturalness classifier reward.
# Reward alpha anneals linearly from 0 to cfg.grpov1_clf_weight_max.
# Paula Guerrero Castelló, May 2026

import gc
import torch
import sacrebleu

from trl import GRPOConfig, GRPOTrainer

from config import Config
from utils.model import load_sft_model, load_base_tokenizer
from utils.data import load_amic, build_grpo_dataset
from utils.callbacks import RewardPlotCallback, RewardEvalSaveCallback
from rewards.chrf_reward import chrf_reward_batch
from rewards.classifier_reward import ClassifierReward


def run_grpo_v1(cfg: Config) -> None:
    use_bf16 = torch.cuda.is_bf16_supported()

    # Directories
    cfg.grpov1_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = cfg.grpov1_dir / "checkpoints"
    best_dir = cfg.grpov1_dir / "best_model"
    ckpt_dir.mkdir(exist_ok=True)
    best_dir.mkdir(exist_ok=True)

    # Model & tokenizer
    tokenizer = load_base_tokenizer(cfg, padding_side="left")
    model     = load_sft_model(cfg, trainable=True)
    model.print_trainable_parameters()

    # Rewards
    clf_reward = ClassifierReward(cfg)

    # Step counter for alpha annealing (mutable via dict so closure can write)
    state = {"step": 0}

    def _alpha() -> float:
        progress = min(1.0, state["step"] / max(1, cfg.grpov1_clf_warmup))
        return cfg.grpov1_clf_weight_max * progress

    def reward_fn(completions, reference=None, **kwargs):
        state["step"] += 1
        alpha  = _alpha()
        clean  = [c.split("model\n")[-1].strip() for c in completions]
        r_c    = chrf_reward_batch(clean, list(reference) if reference else [""] * len(clean))
        r_t    = clf_reward(clean) if alpha > 0 else [0.0] * len(clean)
        rewards = [(1.0 - alpha) * c + alpha * t for c, t in zip(r_c, r_t)]
        mean_r  = sum(rewards) / len(rewards)
        print(f"[GRPOv1] step={state['step']:3d}  alpha={alpha:.3f}  mean_reward={mean_r:.4f}")
        return rewards

    def eval_reward_fn(completions, reference=None, **kwargs):
        clean = [c.split("model\n")[-1].strip() for c in completions]
        refs = list(reference) if reference else [""] * len(clean)
        r_c = chrf_reward_batch(clean, refs)
        r_t = clf_reward(clean)
        return [
            (1.0 - cfg.grpov1_clf_weight_max) * c + cfg.grpov1_clf_weight_max * t
            for c, t in zip(r_c, r_t)
        ]

    # Data
    train_raw, val_raw = load_amic(cfg, val_split=cfg.grpo_val_split)
    grpo_dataset = build_grpo_dataset(train_raw, tokenizer, cfg, cfg.grpov1_train_samples)
    val_dataset  = build_grpo_dataset(
        val_raw.select(range(min(cfg.grpo_val_samples, len(val_raw)))),
        tokenizer, cfg, cfg.grpo_val_samples,
    )

    # Callbacks
    callbacks = [
        RewardPlotCallback(cfg.grpov1_dir / "grpov1_reward_curve.png"),
        RewardEvalSaveCallback(
            tokenizer=tokenizer,
            model=model,
            eval_dataset=val_dataset,
            save_dir=best_dir,
            reward_fn=eval_reward_fn,
            cfg=cfg,
            every_n_steps=cfg.grpo_val_every,
        ),
    ]

    # Trainer
    grpo_config = GRPOConfig(
        per_device_train_batch_size=cfg.grpo_batch_size,
        gradient_accumulation_steps=cfg.grpov1_grad_accum,
        learning_rate=cfg.grpo_lr,
        max_steps=cfg.grpov1_max_steps,
        warmup_steps=cfg.grpo_warmup_steps,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        beta=cfg.grpo_beta,
        num_generations=cfg.grpov1_num_gen,
        max_completion_length=cfg.grpov1_max_new,
        temperature=0.9,
        output_dir=str(ckpt_dir),
        logging_steps=1,
        save_steps=20,
        seed=3407,
        report_to="none",
        fp16=not use_bf16,
        bf16=use_bf16,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=grpo_dataset,
        callbacks=callbacks,
    )

    torch.cuda.empty_cache()
    gc.collect()

    stats = trainer.train()
    print(f"\nGRPOv1 complete — best model saved → {best_dir}")
    print(f"Training time: {stats.metrics['train_runtime']:.1f}s")

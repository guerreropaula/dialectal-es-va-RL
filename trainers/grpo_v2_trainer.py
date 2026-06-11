# trainers/grpo_v2_trainer.py
# GRPOv2: SFT model continued with composite reward (chrF + COMET + TTR - copy).
# Paula Guerrero Castelló, May 2026

import gc
import torch
import torch.distributed as dist

from trl import GRPOConfig, GRPOTrainer

from config import Config
from utils.model import load_sft_model, load_base_tokenizer
from utils.data import load_amic, build_grpo_dataset
from utils.callbacks import RewardPlotCallback, RewardEvalSaveCallback
from rewards.composite_reward import CompositeReward


def run_grpo_v2(cfg: Config) -> None:
    use_bf16 = torch.cuda.is_bf16_supported()
    local_rank = int(__import__("os").environ.get("LOCAL_RANK", 0))

    # Directories
    cfg.grpov2_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = cfg.grpov2_dir / "checkpoints"
    best_dir = cfg.grpov2_dir / "best_model"
    ckpt_dir.mkdir(exist_ok=True)
    best_dir.mkdir(exist_ok=True)

    # Model & tokenizer
    tokenizer = load_base_tokenizer(cfg, padding_side="left")
    model     = load_sft_model(cfg, trainable=True)
    model.print_trainable_parameters()

    # COMET loaded only on rank-0 (or single GPU)
    comet_model = None
    if local_rank == 0:
        from comet import download_model, load_from_checkpoint
        path        = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(path).to("cuda")
    if dist.is_initialized():
        dist.barrier()

    # Reward
    composite = CompositeReward(cfg, comet_model=comet_model)

    def reward_fn(completions, reference=None, source_es=None, **kwargs):
        clean = [c.strip() if isinstance(c, str) else "" for c in completions]
        refs  = list(reference)  if reference  is not None else [""] * len(clean)
        srcs  = list(source_es)  if source_es  is not None else [""] * len(clean)
        rewards = composite(clean, reference=refs, source_es=srcs)
        mean_r  = sum(rewards) / len(rewards)
        print(f"[GRPOv2] mean_reward={mean_r:.4f}")
        return rewards

    # Data
    train_raw, val_raw = load_amic(cfg, val_split=cfg.grpo_val_split)
    grpo_dataset = build_grpo_dataset(train_raw, tokenizer, cfg, cfg.grpov2_train_samples)
    val_dataset  = build_grpo_dataset(
        val_raw.select(range(min(cfg.grpo_val_samples, len(val_raw)))),
        tokenizer, cfg, cfg.grpo_val_samples,
    )

    # Callbacks
    callbacks = [
        RewardPlotCallback(cfg.grpov2_dir / "grpov2_reward_curve.png"),
        RewardEvalSaveCallback(
            tokenizer=tokenizer,
            model=model,
            eval_dataset=val_dataset,
            save_dir=best_dir,
            reward_fn=reward_fn,
            cfg=cfg,
            every_n_steps=cfg.grpo_val_every,
        ),
    ]

    # Trainer
    grpo_config = GRPOConfig(
        max_completion_length=cfg.grpo_max_new,
        num_generations=cfg.grpo_num_gen,
        per_device_train_batch_size=cfg.grpo_batch_size,
        gradient_accumulation_steps=cfg.grpo_grad_accum,
        max_steps=cfg.grpov2_max_steps,
        learning_rate=cfg.grpo_lr,
        warmup_steps=cfg.grpo_warmup_steps,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        beta=cfg.grpo_beta,
        epsilon=cfg.grpo_epsilon,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        output_dir=str(ckpt_dir),
        logging_steps=10,
        save_steps=10,
        report_to="none",
        seed=42,
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
    print(f"\nGRPOv2 complete — best model saved → {best_dir}")
    print(f"Training time: {stats.metrics['train_runtime']:.1f}s")

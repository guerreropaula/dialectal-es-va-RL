# trainers/sft_trainer.py
# Supervised fine-tuning of TranslateGemma-4B-IT for Spanish-Valencian.
# Paula Guerrero Castelló, May 2026

import gc
import torch

from transformers import DataCollatorForLanguageModeling
from trl import SFTConfig, SFTTrainer

from config import Config
from utils.model import load_base_model, load_base_tokenizer, attach_lora, print_trainable
from utils.data import load_amic, build_sft_dataset, make_inference_prompt
from utils.callbacks import LossPlotCallback, BleuEvalSaveCallback


class Gemma3DataCollator:
    """
    Adds token_type_ids (all zeros) required by Gemma-3's attention mask.
    Wraps the standard causal-LM DataCollator.
    """

    def __init__(self, tokenizer):
        self._base = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def __call__(self, features):
        batch = self._base(features)
        batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
        return batch


def run_sft(cfg: Config) -> None:
    use_bf16 = torch.cuda.is_bf16_supported()

    cfg.sft_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = cfg.sft_dir / "checkpoints"
    best_dir = cfg.sft_dir / "best_model"
    ckpt_dir.mkdir(exist_ok=True)
    best_dir.mkdir(exist_ok=True)

    tokenizer = load_base_tokenizer(cfg, padding_side="right")
    base = load_base_model(cfg, trainable=True)
    model = attach_lora(base, cfg)
    print_trainable(model)

    train_raw, val_raw = load_amic(cfg, val_split=cfg.sft_val_split)
    sft_dataset = build_sft_dataset(train_raw, tokenizer, cfg)
    val_samples = val_raw.select(range(min(cfg.sft_val_samples, len(val_raw))))

    callbacks = [
        LossPlotCallback(cfg.sft_dir / "sft_loss_curve.png"),
        BleuEvalSaveCallback(
            tokenizer=tokenizer,
            model=model,
            eval_samples=val_samples,
            save_dir=best_dir,
            make_prompt_fn=make_inference_prompt,
            cfg=cfg,
            every_n_steps=cfg.sft_val_every_steps,
        ),
    ]

    model.train()
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=sft_dataset.shuffle(seed=42).select(
            range(min(cfg.sft_train_samples, len(sft_dataset)))
        ),
        data_collator=Gemma3DataCollator(tokenizer),
        callbacks=callbacks,
        args=SFTConfig(
            packing=False,
            per_device_train_batch_size=cfg.sft_batch_size,
            gradient_accumulation_steps=cfg.sft_grad_accum,
            warmup_steps=cfg.sft_warmup_steps,
            max_steps=cfg.sft_max_steps,
            learning_rate=cfg.sft_lr,
            logging_steps=25,
            optim="paged_adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=str(ckpt_dir),
            save_steps=25,
            report_to="none",
            fp16=not use_bf16,
            bf16=use_bf16,
            gradient_checkpointing=True,
            dataloader_num_workers=2,
        ),
    )

    torch.cuda.empty_cache()
    gc.collect()

    stats = trainer.train()
    print(f"\nSFT complete — loss: {stats.metrics.get('train_loss', 'N/A'):.4f}")
    print(f"Best model saved → {best_dir}")

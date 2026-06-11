# utils/model.py
# Model and tokenizer loading helpers shared by all training scripts.
# Paula Guerrero Castelló, May 2026

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from config import Config


def get_compute_dtype(cfg: Config) -> torch.dtype:
    use_bf16 = torch.cuda.is_bf16_supported()
    return torch.bfloat16 if use_bf16 else torch.float16


def build_bnb_config(cfg: Config) -> BitsAndBytesConfig:
    """4-bit NF4 quantisation config used by all models."""
    return BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.quant_type,
        bnb_4bit_use_double_quant=cfg.double_quant,
        bnb_4bit_compute_dtype=get_compute_dtype(cfg),
    )


def load_base_model(cfg: Config, trainable: bool = False):
    """Load the base TranslateGemma model with 4-bit quantisation."""
    bnb = build_bnb_config(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        quantization_config=bnb,
        device_map="auto",
        token=cfg.hf_token or None,
        torch_dtype=get_compute_dtype(cfg),
        trust_remote_code=True,
        use_safetensors=True,
    )
    if trainable:
        model = prepare_model_for_kbit_training(model)
    return model


def load_base_tokenizer(cfg: Config, padding_side: str = "right"):
    """Load tokenizer for the base model."""
    tok = AutoTokenizer.from_pretrained(
        cfg.base_model_id, token=cfg.hf_token or None, use_safetensors=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = padding_side
    return tok


def load_sft_model(cfg: Config, trainable: bool = False):
    """Load base model + SFT LoRA adapter."""
    base = load_base_model(cfg, trainable=trainable)
    model = PeftModel.from_pretrained(
        base,
        cfg.sft_model_id,
        token=cfg.hf_token or None,
        is_trainable=trainable,
        use_safetensors=True,
    )
    if trainable:
        model = prepare_model_for_kbit_training(model)
    return model


def attach_lora(model, cfg: Config):
    """Attach a fresh LoRA adapter for SFT training."""
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=cfg.lora_targets,
    )
    return get_peft_model(model, lora_cfg)


def print_trainable(model) -> None:
    model.print_trainable_parameters()

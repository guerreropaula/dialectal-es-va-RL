# config.py
# Central configuration for all ES-VA MT experiments.
# Edit this file to change any hyperparameter, path, or model ID.
# Paula Guerrero Castelló, May 2026

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class Config:
    # ── Authentication ────────────────────────────────────────────────────
    hf_token: str = field(default_factory=lambda: os.getenv("HF_TOKEN", ""))

    # ── Model IDs ─────────────────────────────────────────────────────────
    base_model_id:    str = "google/translategemma-4b-it"
    sft_model_id:     str = "guerreropaula/translategemma4b-sft-es-va"
    grpov1_model_id:  str = "guerreropaula/translategemma4b-grpov1-es-va"
    grpov2_model_id:  str = "guerreropaula/translategemma4b-grpov2-es-va"
    clf_model_id:     str = "guerreropaula/ht_mt_classifier_best"
    roberta_ca_id:    str = "PlanTL-GOB-ES/roberta-base-ca"
    helsinki_id:      str = "Helsinki-NLP/opus-mt-es-ca"
    nllb_id:          str = "facebook/nllb-200-distilled-600M"

    # ── Language codes ────────────────────────────────────────────────────
    source_lang_code: str = "es"
    target_lang_code: str = "ca"
    source_col:       str = "ES"
    target_col:       str = "VA"

    # ── Paths ─────────────────────────────────────────────────────────────
    output_root:       Path = Path("./outputs")
    data_dir:          Path = Path("./data")

    @property
    def sft_dir(self)     -> Path: return self.output_root / "sft"
    @property
    def grpov1_dir(self)  -> Path: return self.output_root / "grpov1"
    @property
    def grpov2_dir(self)  -> Path: return self.output_root / "grpov2"
    @property
    def clf_dir(self)     -> Path: return self.output_root / "classifier"

    # ── Quantization ──────────────────────────────────────────────────────
    load_in_4bit:      bool = True
    quant_type:        str  = "nf4"
    double_quant:      bool = True

    # ── LoRA (SFT) ────────────────────────────────────────────────────────
    lora_r:          int   = 16
    lora_alpha:      int   = 32
    lora_dropout:    float = 0.05
    lora_targets:    list  = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # ── SFT hyperparams ───────────────────────────────────────────────────
    sft_train_samples:   int   = 50_000
    sft_val_split:       float = 0.02
    sft_val_samples:     int   = 200
    sft_val_every_steps: int   = 100
    sft_max_steps:       int   = 2_000
    sft_lr:              float = 2e-4
    sft_batch_size:      int   = 1
    sft_grad_accum:      int   = 32
    sft_warmup_steps:    int   = 25
    sft_max_seq_length:  int   = 256

    # ── GRPO shared hyperparams ───────────────────────────────────────────
    grpo_beta:          float = 0.04
    grpo_epsilon:       float = 0.2
    grpo_lr:            float = 5e-6
    grpo_batch_size:    int   = 1
    grpo_grad_accum:    int   = 16
    grpo_warmup_steps:  int   = 20
    grpo_num_gen:       int   = 4
    grpo_max_new:       int   = 128
    grpo_val_split:     float = 0.02
    grpo_val_samples:   int   = 200
    grpo_val_every:     int   = 20

    # ── GRPOv1 specific ───────────────────────────────────────────────────
    grpov1_train_samples:  int   = 5_000
    grpov1_max_steps:      int   = 100
    grpov1_clf_warmup:     int   = 50      # steps before alpha kicks in
    grpov1_clf_weight_max: float = 0.3     # max alpha for classifier
    grpov1_best_step:      int   = 80
    grpov1_grad_accum:     int   = 8
    grpov1_num_gen:        int   = 2
    grpov1_max_new:        int   = 100

    # ── GRPOv2 specific ───────────────────────────────────────────────────
    grpov2_train_samples: int   = 10_000
    grpov2_max_steps:     int   = 200
    grpov2_best_step:     int   = 100
    w_chrf:               float = 0.5
    w_comet:              float = 0.3
    w_ttr:                float = 0.2

    # ── Classifier training ───────────────────────────────────────────────
    clf_max_per_corpus:  int   = 20_000
    clf_max_length:      int   = 128
    clf_batch_size:      int   = 32
    clf_lr:              float = 2e-5
    clf_epochs:          int   = 5
    clf_ht_label_idx:    int   = 1
    clf_corpora: list = field(default_factory=lambda: [
        "TildeMODEL.es-ca",
        "dogc-es-ca",
        "europarl.es-ca",
    ])

    # ── Evaluation ────────────────────────────────────────────────────────
    eval_n:           int = 1_000
    eval_max_seq:     int = 512
    eval_src_col:     str = "es"
    eval_tgt_col:     str = "va"

    # ── Dialectal feature map (CA form → VA form) ─────────────────────────
    ca_va_features: dict = field(default_factory=lambda: {
        "aquesta": "esta",       "quest": "este",
        "aquestes": "estes",     "aquests": "estos",
        "seva": "seua",          "seves": "seues",
        "darrer": "últim",       "darrers": "últims",
        "darrera": "última",     "tenir": "tindre",
        "obtenir": "obtindre",   "segueix": "seguix",
        "segueixen": "seguixen", "requereix": "requerix",
        "divideix": "dividix",   "constitueixen": "constituïxen",
        "absorbeixen": "absorbixen", "veure": "vore",
        "nens": "xiquets",       "nen": "xiquet",
        "nena": "xiqueta",       "nenes": "xiquetes",
        "petit": "xicotet",      "petits": "xicotets",
        "petita": "xicoteta",    "feina": "faena",
        "feines": "faenes",      "cop": "colp",
        "cops": "colps",         "avui": "hui",
        "servei": "servici",     "serveis": "servicis",
        "mirall": "espill",      "tomàquet": "tomaca",
        "tomàquets": "tomaques",
    })

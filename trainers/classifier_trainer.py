# trainers/classifier_trainer.py
# Fine-tunes roberta-base-ca as an HT/MT binary classifier.
# Paula Guerrero Castelló, May 2026

import gc
import json
import os
import subprocess

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import evaluate

from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    MarianMTModel,
    MarianTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    pipeline as hf_pipeline,
)

from config import Config


# ── Callbacks ─────────────────────────────────────────────────────────────────

class _LossPlotCallback(TrainerCallback):
    def __init__(self, save_path):
        self.save_path    = save_path
        self.train_steps  = []
        self.train_losses = []
        self.eval_steps   = []
        self.eval_losses  = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" in logs:
            self.train_steps.append(state.global_step)
            self.train_losses.append(logs["loss"])
        if "eval_loss" in logs:
            self.eval_steps.append(state.global_step)
            self.eval_losses.append(logs["eval_loss"])
        self._save()

    def _save(self):
        fig, ax = plt.subplots(figsize=(10, 4))
        if self.train_losses:
            ax.plot(self.train_steps, self.train_losses, label="Train", color="#2B5797", lw=1.5)
        if self.eval_losses:
            ax.plot(self.eval_steps, self.eval_losses, label="Val", color="#E05C2A",
                    lw=2, ls="--", marker="o", ms=6)
        ax.set_xlabel("Step"); ax.set_ylabel("Loss")
        ax.set_title("HT vs MT Classifier — Training Loss")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_path, dpi=150, bbox_inches="tight")
        plt.close()


class _VerboseCallback(TrainerCallback):
    def __init__(self, tokenizer, val_dataset):
        self.tokenizer   = tokenizer
        self.val_dataset = val_dataset

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "eval_loss" in logs:
            print(f"\n{'='*55}")
            print(f"  EVAL — epoch {state.epoch:.0f} (step {state.global_step})")
            for k in ("eval_loss", "eval_accuracy", "eval_f1", "eval_precision", "eval_recall"):
                if k in logs:
                    print(f"  {k.replace('eval_', ''):<12} {logs[k]:.4f}")
            print("="*55 + "\n")

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        model.eval()
        label_map = {0: "MT", 1: "HT"}
        print("── 3 validation examples ──")
        for i in range(3):
            s    = self.val_dataset[i]
            ids  = s["input_ids"].unsqueeze(0).to(model.device)
            mask = s["attention_mask"].unsqueeze(0).to(model.device)
            real = label_map[s["labels"].item()]
            with torch.no_grad():
                pred = label_map[model(input_ids=ids, attention_mask=mask).logits.argmax(-1).item()]
            icon = "OK" if real == pred else "WRONG"
            text = self.tokenizer.decode(s["input_ids"], skip_special_tokens=True)[:90]
            print(f"  [{icon}] real={real} pred={pred} | \"{text}...\"")
        print()


# ── Translation helpers ───────────────────────────────────────────────────────

@torch.no_grad()
def _translate_helsinki(texts, helsinki_tok, helsinki_model, device, batch_size=64):
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Helsinki"):
        batch = texts[i : i + batch_size]
        tok   = helsinki_tok(batch, return_tensors="pt", padding=True,
                             truncation=True, max_length=256)
        if device == 0:
            tok = {k: v.cuda() for k, v in tok.items()}
        out = helsinki_model.generate(**tok, max_length=256)
        results.extend(helsinki_tok.batch_decode(out, skip_special_tokens=True))
    return results


@torch.no_grad()
def _translate_nllb(texts, nllb_tok, nllb_model, cat_id, device, batch_size=32):
    results = []
    nllb_tok.src_lang = "spa_Latn"
    for i in tqdm(range(0, len(texts), batch_size), desc="NLLB"):
        batch = texts[i : i + batch_size]
        tok   = nllb_tok(batch, return_tensors="pt", padding=True,
                         truncation=True, max_length=256)
        if device == 0:
            tok = {k: v.cuda() for k, v in tok.items()}
        out = nllb_model.generate(**tok, forced_bos_token_id=cat_id, max_length=256)
        results.extend(nllb_tok.batch_decode(out, skip_special_tokens=True))
    return results


# ── Main entry point ──────────────────────────────────────────────────────────

def run_classifier(cfg: Config) -> None:
    device = 0 if torch.cuda.is_available() else -1

    cfg.clf_dir.mkdir(parents=True, exist_ok=True)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir  = cfg.data_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    ckpt_dir = cfg.clf_dir / "checkpoints"
    best_dir = cfg.clf_dir / "best_model"
    ckpt_dir.mkdir(exist_ok=True)
    best_dir.mkdir(exist_ok=True)

    base_url = "https://github.com/Softcatala/parallel-catalan-corpus/raw/master/spa-cat/"

    # Download corpora
    for corpus in cfg.clf_corpora:
        for lang in ["es", "ca"]:
            fname = f"{corpus}.{lang}"
            out   = raw_dir / fname
            if not out.exists():
                for ext in [".xz", ""]:
                    url = base_url + fname + ext
                    r   = subprocess.run(["wget", "-q", "-O", str(out) + ext, url])
                    if r.returncode == 0:
                        if ext == ".xz":
                            subprocess.run(["xz", "-d", str(out) + ext])
                        print(f"Downloaded: {fname}")
                        break

    # Build HT dataframe
    frames = []
    for corpus in cfg.clf_corpora:
        es_path = raw_dir / f"{corpus}.es"
        ca_path = raw_dir / f"{corpus}.ca"
        if es_path.exists() and ca_path.exists():
            es_lines = [l.strip() for l in es_path.read_text("utf-8").splitlines() if l.strip()]
            ca_lines = [l.strip() for l in ca_path.read_text("utf-8").splitlines() if l.strip()]
            n  = min(len(es_lines), len(ca_lines))
            df = pd.DataFrame({"source_es": es_lines[:n], "ca_human": ca_lines[:n], "corpus": corpus})
            frames.append(df)
            print(f"  {corpus}: {n:,} pairs")

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all[
        (df_all.source_es.str.len() > 20) & (df_all.ca_human.str.len() > 20) &
        (df_all.source_es.str.len() < 500) & (df_all.ca_human.str.len() < 500)
    ]
    df_balanced = (
        df_all.groupby("corpus", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), cfg.clf_max_per_corpus), random_state=42))
        .reset_index(drop=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    print(f"Dataset: {len(df_balanced):,} pairs\n{df_balanced.corpus.value_counts()}")

    # Generate MT translations
    helsinki_tok   = MarianTokenizer.from_pretrained(cfg.helsinki_id, use_safetensors=True)
    helsinki_model = MarianMTModel.from_pretrained(cfg.helsinki_id, use_safetensors=True, device_map="auto")
    helsinki_model.eval()

    nllb_tok   = AutoTokenizer.from_pretrained(cfg.nllb_id, use_safetensors=True)
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(cfg.nllb_id, use_safetensors=True, device_map="auto")
    nllb_model.eval()
    cat_id = nllb_tok.convert_tokens_to_ids("cat_Latn")

    np.random.seed(42)
    assignment   = np.random.choice(["Helsinki", "NLLB"], size=len(df_balanced), p=[0.5, 0.5])
    helsinki_idx = np.where(assignment == "Helsinki")[0]
    nllb_idx     = np.where(assignment == "NLLB")[0]

    mt_trans = [""] * len(df_balanced)
    for i, t in zip(helsinki_idx, _translate_helsinki(
        df_balanced.iloc[helsinki_idx]["source_es"].tolist(),
        helsinki_tok, helsinki_model, device
    )):
        mt_trans[i] = t
    for i, t in zip(nllb_idx, _translate_nllb(
        df_balanced.iloc[nllb_idx]["source_es"].tolist(),
        nllb_tok, nllb_model, cat_id, device
    )):
        mt_trans[i] = t

    df_balanced["ca_mt"]     = mt_trans
    df_balanced["mt_system"] = assignment.tolist()
    df_balanced.to_csv(cfg.data_dir / "df_balanced_mt_ht.csv", index=False)

    del helsinki_model, helsinki_tok, nllb_model, nllb_tok
    torch.cuda.empty_cache(); gc.collect()

    # Build classification dataset
    ht_rows = df_balanced[["ca_human", "corpus"]].copy()
    ht_rows.columns = ["text", "corpus"]
    ht_rows["label"] = 1

    mt_rows = df_balanced[["ca_mt", "corpus", "mt_system"]].copy()
    mt_rows.columns = ["text", "corpus", "source"]
    mt_rows["label"] = 0

    df_clf = pd.concat([ht_rows, mt_rows], ignore_index=True).sample(frac=1, random_state=42)
    train_df, val_df = train_test_split(df_clf, test_size=0.1,
                                        stratify=df_clf["label"], random_state=42)
    train_df.to_csv(cfg.data_dir / "train.csv", index=False)
    val_df.to_csv(cfg.data_dir / "val.csv", index=False)

    # Tokenize
    clf_tokenizer = AutoTokenizer.from_pretrained(cfg.roberta_ca_id, use_safetensors=True)

    def _tokenize(batch):
        enc = clf_tokenizer(batch["text"], truncation=True,
                            max_length=cfg.clf_max_length, padding="max_length")
        enc["labels"] = batch["label"]
        return enc

    train_ds = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
    val_ds   = Dataset.from_pandas(val_df[["text", "label"]].reset_index(drop=True))
    train_tok = train_ds.map(_tokenize, batched=True, remove_columns=["text", "label"])
    val_tok   = val_ds.map(_tokenize,   batched=True, remove_columns=["text", "label"])
    train_tok.set_format("torch"); val_tok.set_format("torch")

    # Metrics
    accuracy_metric  = evaluate.load("accuracy")
    f1_metric        = evaluate.load("f1")
    recall_metric    = evaluate.load("recall")
    precision_metric = evaluate.load("precision")

    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            **accuracy_metric.compute(predictions=preds, references=labels),
            **f1_metric.compute(predictions=preds, references=labels, average="macro"),
            **recall_metric.compute(predictions=preds, references=labels, average="macro"),
            **precision_metric.compute(predictions=preds, references=labels, average="macro"),
        }

    # Model & training
    clf_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.roberta_ca_id, num_labels=2,
        id2label={0: "MT", 1: "HT"}, label2id={"MT": 0, "HT": 1},
        use_safetensors=True,
    )

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=cfg.clf_epochs,
        per_device_train_batch_size=cfg.clf_batch_size,
        per_device_eval_batch_size=cfg.clf_batch_size,
        warmup_ratio=0.06,
        weight_decay=0.01,
        learning_rate=cfg.clf_lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=clf_model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=DataCollatorWithPadding(clf_tokenizer),
        compute_metrics=_compute_metrics,
        callbacks=[
            _LossPlotCallback(str(cfg.clf_dir / "classifier_loss.png")),
            _VerboseCallback(clf_tokenizer, val_tok),
            EarlyStoppingCallback(early_stopping_patience=4),
        ],
    )

    print("Starting classifier training...")
    trainer.train()
    trainer.save_model(str(best_dir))
    clf_tokenizer.save_pretrained(str(best_dir))
    results = trainer.evaluate()
    print("\nValidation results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    print(f"Best classifier saved → {best_dir}")

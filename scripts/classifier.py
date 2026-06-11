# classifier.py
# HT vs. MT Binary Classifier
# Paula Guerrero Castelló, May 2026
# ---------------------------------------------------------------------------

import os
import gc
import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F

from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm
from huggingface_hub import login

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
import evaluate


 
# ---- Config -----------------------------------------------------------

hf_token     = ""
model_name   = "PlanTL-GOB-ES/roberta-base-ca"
helsinki_model_name = "Helsinki-NLP/opus-mt-es-ca"
nllb_model_name     = "facebook/nllb-200-distilled-600M"
clf_repo_id    = "guerreropaula/ht_mt_classifier_best"
output_root    = "./outputs"
classifier_run_dir = os.path.join(output_root, "classifier")
clf_output_dir = os.path.join(classifier_run_dir, "best_model")
ht_label_idx   = 1
training_output_dir = os.path.join(classifier_run_dir, "checkpoints")

max_per_corpus = 20_000
device = 0 if torch.cuda.is_available() else -1

corpora = [
    "TildeMODEL.es-ca",
    "dogc-es-ca",
    "europarl.es-ca",
]

base_url = "https://github.com/Softcatala/parallel-catalan-corpus/raw/master/spa-cat/"

if hf_token:
    login(token=hf_token)

os.makedirs(training_output_dir, exist_ok=True)
os.makedirs(clf_output_dir, exist_ok=True)


# --- Softcatalà corpus ------------------------------------------------------

os.makedirs("data/raw", exist_ok=True)

for corpus in corpora:
    for lang in ["es", "ca"]:
        fname = f"{corpus}.{lang}"
        for ext in [".xz", ""]:
            url = base_url + fname + ext
            out = f"data/raw/{fname}{ext}"
            if not os.path.exists(out.replace(".xz", "")):
                r = subprocess.run(["wget", "-q", "-O", out, url])
                if r.returncode == 0 and ext == ".xz":
                    subprocess.run(["xz", "-d", out])
                    print(f"Downloaded and decompressed: {fname}")
                elif r.returncode == 0:
                    print(f"Downloaded: {fname}")
                break


# --- Load & merge corpora-------------------------------------------------------

frames = []

for corpus in corpora:
    es_path = f"data/raw/{corpus}.es"
    ca_path = f"data/raw/{corpus}.ca"
    if os.path.exists(es_path) and os.path.exists(ca_path):
        with open(es_path, encoding="utf-8") as f:
            es_lines = [l.strip() for l in f if l.strip()]
        with open(ca_path, encoding="utf-8") as f:
            ca_lines = [l.strip() for l in f if l.strip()]
        n = min(len(es_lines), len(ca_lines))
        df = pd.DataFrame({
            "source_es": es_lines[:n],
            "ca_human" : ca_lines[:n],
            "corpus"   : corpus,
        })
        frames.append(df)
        print(f"  {corpus}: {n:,} pairs")
    else:
        print(f"  WARNING — not found: {corpus}")

df_all = pd.concat(frames, ignore_index=True)
print(f"\nTotal: {len(df_all):,} pairs")

df_all = df_all[
    (df_all.source_es.str.len() > 20) &
    (df_all.ca_human.str.len()  > 20) &
    (df_all.source_es.str.len() < 500) &
    (df_all.ca_human.str.len()  < 500)
]

df_balanced = (
    df_all.groupby("corpus", group_keys=False)
    .apply(lambda x: x.sample(min(len(x), max_per_corpus), random_state=42))
    .reset_index(drop=True)
)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Dataset after filtering and sampling: {len(df_balanced):,} pairs")
print(df_balanced.corpus.value_counts())


# --- Generate MT examples -----------------------------------------------------

print("Loading Helsinki-NLP/opus-mt-es-ca...")
helsinki_tok   = MarianTokenizer.from_pretrained(helsinki_model_name)
helsinki_model = MarianMTModel.from_pretrained(helsinki_model_name)
if device == 0:
    helsinki_model = helsinki_model.cuda()
helsinki_model.eval()

print("\nLoading facebook/nllb-200-distilled-600M...")
nllb_tok   = AutoTokenizer.from_pretrained(nllb_model_name)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name)
if device == 0:
    nllb_model = nllb_model.cuda()
nllb_model.eval()
cat_token_id = nllb_tok.convert_tokens_to_ids("cat_Latn")
print(f"NLLB model ready — cat_Latn token id: {cat_token_id}")


@torch.no_grad()
def batch_translate_helsinki(texts: list, batch_size: int = 64) -> list:
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
def batch_translate_nllb(texts: list, batch_size: int = 32) -> list:
    results = []
    nllb_tok.src_lang = "spa_Latn"
    for i in tqdm(range(0, len(texts), batch_size), desc="NLLB"):
        batch = texts[i : i + batch_size]
        tok   = nllb_tok(batch, return_tensors="pt", padding=True,
                         truncation=True, max_length=256)
        if device == 0:
            tok = {k: v.cuda() for k, v in tok.items()}
        out = nllb_model.generate(**tok, forced_bos_token_id=cat_token_id, max_length=256)
        results.extend(nllb_tok.batch_decode(out, skip_special_tokens=True))
    return results


np.random.seed(42)
assignment   = np.random.choice(["Helsinki", "NLLB"], size=len(df_balanced), p=[0.5, 0.5])
helsinki_idx = np.where(assignment == "Helsinki")[0]
nllb_idx     = np.where(assignment == "NLLB")[0]

mt_translations = [""] * len(df_balanced)

print("Translating with Helsinki...")
helsinki_results = batch_translate_helsinki(df_balanced.iloc[helsinki_idx]["source_es"].tolist())
for i, t in zip(helsinki_idx, helsinki_results):
    mt_translations[i] = t

print("\nTranslating with NLLB...")
nllb_results = batch_translate_nllb(df_balanced.iloc[nllb_idx]["source_es"].tolist())
for i, t in zip(nllb_idx, nllb_results):
    mt_translations[i] = t

df_balanced["ca_mt"]     = mt_translations
df_balanced["mt_system"] = assignment.tolist()

df_balanced.to_csv("df_balanced_mt_ht.csv", index=False)

del helsinki_model, helsinki_tok, nllb_model, nllb_tok
torch.cuda.empty_cache()
gc.collect()


# --- Build dataset ---------------------------------------------

ht_rows = df_balanced[["ca_human", "corpus"]].copy()
ht_rows.columns = ["text", "corpus"]
ht_rows["label"]  = 1
ht_rows["source"] = "human"

mt_rows = df_balanced[["ca_mt", "corpus", "mt_system"]].copy()
mt_rows.columns = ["text", "corpus", "source"]
mt_rows["label"]  = 0

df_clf = pd.concat([ht_rows, mt_rows], ignore_index=True)
df_clf = df_clf.sample(frac=1, random_state=42).reset_index(drop=True)

os.makedirs("data", exist_ok=True)
train_df, val_df = train_test_split(
    df_clf, test_size=0.1, stratify=df_clf["label"], random_state=42
)

train_ds = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
val_ds   = Dataset.from_pandas(val_df[["text", "label"]].reset_index(drop=True))

train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
df_clf.to_parquet("data/ht_mt_classifier_dataset.parquet", index=False)


# --- Tokenization -----------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize(batch):
    encoded = tokenizer(
        batch["text"],
        truncation  = True,
        max_length  = 128,
        padding     = "max_length",
    )
    encoded["labels"] = batch["label"]
    return encoded


train_tok = train_ds.map(tokenize, batched=True, remove_columns=["text", "label"])
val_tok   = val_ds.map(tokenize,   batched=True, remove_columns=["text", "label"])

train_tok.set_format("torch")
val_tok.set_format("torch")


# --- Classifier & metrics -----------------------------------------------------------

clf_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels = 2,
    id2label   = {0: "MT", 1: "HT"},
    label2id   = {"MT": 0, "HT": 1},
)

accuracy_metric  = evaluate.load("accuracy")
f1_metric        = evaluate.load("f1")
recall_metric    = evaluate.load("recall")
precision_metric = evaluate.load("precision")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        **accuracy_metric.compute(predictions=predictions, references=labels),
        **f1_metric.compute(predictions=predictions, references=labels, average="macro"),
        **recall_metric.compute(predictions=predictions, references=labels, average="macro"),
        **precision_metric.compute(predictions=predictions, references=labels, average="macro"),
    }


# --- Callbacks -----------------------------------------------------------

class VerboseCallback(TrainerCallback):
    def __init__(self, tokenizer, val_dataset):
        self.tokenizer   = tokenizer
        self.val_dataset = val_dataset

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step  = state.global_step
        epoch = state.epoch or 0

        if "loss" in logs:
            print(f"  step {step:>5} | epoch {epoch:.2f} | train_loss: {logs['loss']:.4f}")

        if "eval_loss" in logs:
            print("\n" + "="*60)
            print(f"  EVAL — epoch {epoch:.0f}  (step {step})")
            print(f"  {'loss':<12} {logs.get('eval_loss', 0):.4f}")
            print(f"  {'accuracy':<12} {logs.get('eval_accuracy', 0):.4f}")
            print(f"  {'f1-macro':<12} {logs.get('eval_f1', 0):.4f}")
            print(f"  {'precision':<12} {logs.get('eval_precision', 0):.4f}")
            print(f"  {'recall':<12} {logs.get('eval_recall', 0):.4f}")
            print("="*60 + "\n")

            history_path = os.path.join(args.output_dir, "training_history.json")
            history = json.load(open(history_path)) if os.path.exists(history_path) else []
            history.append({k: v for k, v in logs.items()})
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        model.eval()
        label_map = {0: "MT", 1: "HT"}
        print("\n── 3 validation examples ──")
        for i in range(3):
            sample = self.val_dataset[i]
            ids    = sample["input_ids"].unsqueeze(0).to(model.device)
            mask   = sample["attention_mask"].unsqueeze(0).to(model.device)
            real   = label_map[sample["labels"].item()]
            with torch.no_grad():
                logits = model(input_ids=ids, attention_mask=mask).logits
            pred = label_map[logits.argmax(-1).item()]
            icon = "OK" if real == pred else "WRONG"
            text = self.tokenizer.decode(sample["input_ids"], skip_special_tokens=True)[:90]
            print(f"  [{icon}] real={real} pred={pred} | \"{text}...\"")
        print()


class LossPlotCallback(TrainerCallback):
    def __init__(self, save_path="classifier_loss.png"):
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
        self._save_plot()

    def _save_plot(self):
        fig, ax = plt.subplots(figsize=(10, 4))
        if self.train_losses:
            ax.plot(self.train_steps, self.train_losses,
                    label="Train loss", color="#2B5797", linewidth=1.5)
        if self.eval_losses:
            ax.plot(self.eval_steps, self.eval_losses,
                    label="Val loss", color="#E05C2A",
                    linewidth=2, linestyle="--", marker="o", markersize=6)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("HT vs MT Classifier -- Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_path, dpi=150, bbox_inches="tight")
        plt.close()


# --- Training -----------------------------------------------------------

os.makedirs(training_output_dir, exist_ok=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir                  = training_output_dir,
    num_train_epochs            = 5,
    per_device_train_batch_size = 32,
    per_device_eval_batch_size  = 32,
    warmup_ratio                = 0.06,
    weight_decay                = 0.01,
    learning_rate               = 2e-5,
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    save_total_limit            = 2,
    load_best_model_at_end      = True,
    metric_for_best_model       = "f1",
    greater_is_better           = True,
    logging_steps               = 50,
    fp16                        = torch.cuda.is_available(),
    report_to                   = "none",
)

trainer = Trainer(
    model           = clf_model,
    args            = training_args,
    train_dataset   = train_tok,
    eval_dataset    = val_tok,
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
    callbacks       = [
        LossPlotCallback("classifier_loss.png"),
        VerboseCallback(tokenizer=tokenizer, val_dataset=val_tok),
        EarlyStoppingCallback(early_stopping_patience=4),
    ],
)

print("Starting training...")
trainer.train()
trainer.save_model(clf_output_dir)
tokenizer.save_pretrained(clf_output_dir)

results = trainer.evaluate()
print("\nValidation results:")
for k, v in results.items():
    print(f"  {k}: {v:.4f}")



# --- Translation test ---------------------------------------------------------

test_hf = load_dataset("gplsi/ES-VA_translation_test", split="test")

nllb_tok_test   = AutoTokenizer.from_pretrained(nllb_model_name)
nllb_model_test = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name)
if device == 0:
    nllb_model_test = nllb_model_test.cuda()
nllb_model_test.eval()
cat_token_id_test = nllb_tok_test.convert_tokens_to_ids("cat_Latn")


@torch.no_grad()
def translate_nllb_test(texts: list, batch_size: int = 32) -> list:
    results = []
    nllb_tok_test.src_lang = "spa_Latn"
    for i in tqdm(range(0, len(texts), batch_size), desc="NLLB test"):
        batch = texts[i : i + batch_size]
        tok   = nllb_tok_test(batch, return_tensors="pt", padding=True,
                              truncation=True, max_length=256)
        if device == 0:
            tok = {k: v.cuda() for k, v in tok.items()}
        out = nllb_model_test.generate(
            **tok, forced_bos_token_id=cat_token_id_test, max_length=256
        )
        results.extend(nllb_tok_test.batch_decode(out, skip_special_tokens=True))
    return results


test_rows = []
for row in test_hf:
    if row.get("va"):
        test_rows.append({"text": row["va"], "label": 1})

es_texts   = [row["es"] for row in test_hf if row.get("es")]
mt_results = translate_nllb_test(es_texts, batch_size=32)
for mt_text in mt_results:
    test_rows.append({"text": mt_text, "label": 0})

test_df = pd.DataFrame(test_rows)

del nllb_model_test, nllb_tok_test
torch.cuda.empty_cache()
gc.collect()

clf_pipe = hf_pipeline(
    "text-classification",
    model       = clf_output_dir,
    tokenizer   = clf_output_dir,
    device      = 0 if torch.cuda.is_available() else -1,
    truncation  = True,
    max_length  = 128,
    batch_size  = 64,
)

predictions_raw = clf_pipe(test_df["text"].tolist())
preds  = [1 if p["label"] == "HT" else 0 for p in predictions_raw]
labels = test_df["label"].tolist()

print("RESULTS ON gplsi/ES-VA_translation_test")
print(classification_report(labels, preds, target_names=["MT (0)", "HT (1)"]))

cm  = confusion_matrix(labels, preds)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["MT", "HT"], yticklabels=["MT", "HT"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix — gplsi/ES-VA_translation_test")
plt.tight_layout()
plt.savefig("confusion_matrix_gplsi.png", dpi=150)
print("Saved → confusion_matrix_gplsi.png")

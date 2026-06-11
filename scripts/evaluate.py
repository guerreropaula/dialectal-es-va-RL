# evaluation.py
# Final evaluation: baseline vs. SFT vs. GRPOv1 vs. GRPOv2
# Metrics: chrF, BLEU, TER, BLEURT, COMET + dialectal valencian score (DVS)
# Paula Guerrero Castelló, May 2026
# --------------------------------------------------------------------------

import re
import gc
import json
import sacrebleu
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from bleurt import score as bleurt_score
from comet import download_model, load_from_checkpoint
from datasets import load_dataset
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# --- Config -------------------------------------------------------------

hf_token        = ""
base_model_id   = "google/translategemma-4b-it"
sft_model_id    = "guerreropaula/translategemma4b-sft-es-va"
grpov1_model_id = "guerreropaula/translategemma4b-grpov1-es-va"
grpov2_model_id = "guerreropaula/translategemma4b-grpov2-es-va"

source_lang_code = "es"
target_lang_code = "ca"
eval_n           = 1000
eval_src         = "es"
eval_tgt         = "va"
max_seq_eval     = 512

device   = "cuda" if torch.cuda.is_available() else "cpu"
use_bf16 = torch.cuda.is_bf16_supported()

models = ["baseline", "sft", "grpov1", "grpov2"]
labels = {"baseline": "BASE", "sft": "SFT", "grpov1": "GRPOv1", "grpov2": "GRPOv2"}
colors = {
    "baseline": "#6c757d",
    "sft":      "#1f77b4",
    "grpov1":   "#ff7f0e",
    "grpov2":   "#2ca02c",
}

ca_va_features = {
    "aquesta": "esta", "quest": "este", "aquestes": "estes", "aquests": "estos",
    "seva": "seua", "seves": "seues", "darrer": "últim", "darrers": "últims",
    "darrera": "última", "tenir": "tindre", "obtenir": "obtindre",
    "segueix": "seguix", "segueixen": "seguixen", "requereix": "requerix",
    "divideix": "dividix", "constitueixen": "constituïxen", "absorbeixen": "absorbixen",
    "veure": "vore", "nens": "xiquets", "nen": "xiquet", "nena": "xiqueta",
    "nenes": "xiquetes", "petit": "xicotet", "petits": "xicotets",
    "petita": "xicoteta", "feina": "faena", "feines": "faenes", "cop": "colp",
    "cops": "colps", "avui": "hui", "servei": "servici", "serveis": "servicis",
    "mirall": "espill", "tomàquet": "tomaca", "tomàquets": "tomaques",
}

if hf_token:
    login(token=hf_token)


# --- Setup -------------------------------------------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.bfloat16 if use_bf16 else torch.float16,
    bnb_4bit_use_double_quant = True,
)

bleurt_scorer = bleurt_score.BleurtScorer("bleurt-base-128")

comet_path  = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_path)


# --- Prompt template -------------------------------------------------------------

def make_eval_prompt(tok, source_text: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type"             : "text",
                    "source_lang_code" : source_lang_code,
                    "target_lang_code" : target_lang_code,
                    "text"             : source_text,
                }
            ],
        }
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# --- Inference & metrics -------------------------------------------------------------

def translate_all(model, tok, sources, refs, label):
    model.eval()
    hyps, skipped_idx = [], []

    for i, src in enumerate(sources):
        prompt = make_eval_prompt(tok, src)
        n_tok  = len(tok(prompt).input_ids)

        if n_tok > max_seq_eval:
            hyps.append("[SKIPPED]")
            skipped_idx.append(i)
            print(f"[{label}] [{i+1:3d}/{len(sources)}] SKIPPED (prompt too long)")
            continue

        inputs  = tok(prompt, return_tensors="pt",
                      truncation=True, max_length=max_seq_eval).to(model.device)
        src_len = len(tok(src).input_ids)
        max_new = min(512, max(80, int(src_len * 1.2)))

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens = max_new,
                do_sample      = False,
                pad_token_id   = tok.eos_token_id,
            )

        new_tok = out[0][inputs["input_ids"].shape[1]:]
        hyp     = tok.decode(new_tok, skip_special_tokens=True).strip()
        hyps.append(hyp)

        chrf_s = sacrebleu.sentence_chrf(hyp, [refs[i]]).score
        print(f"[{label}] [{i+1:3d}/{len(sources)}] chrF={chrf_s:5.1f} | {src[:55]}")

    return hyps, skipped_idx


def comet_scores(sources, hyps, refs):
    data = [{"src": src, "mt": hyp, "ref": ref} for src, hyp, ref in zip(sources, hyps, refs)]
    output = comet_model.predict(data, batch_size=8, gpus=1 if device == "cuda" else 0)
    return output.scores if hasattr(output, "scores") else output[0]


def compute_metrics(hyps, refs, skipped_idx, model_name):
    kept = [i for i in range(len(hyps)) if i not in skipped_idx]
    s = [gold_es[i] for i in kept]
    h = [hyps[i] for i in kept]
    r = [refs[i] for i in kept]
    chrf = sacrebleu.corpus_chrf(h, [r])
    bleu = sacrebleu.corpus_bleu(h, [r])
    ter  = sacrebleu.corpus_ter(h, [r])
    blrt = bleurt_scorer.score(references=r, candidates=h)
    comet = comet_scores(s, h, r)
    return {
        "model"  : model_name,
        "n_eval" : len(h),
        "skipped": len(skipped_idx),
        "chrF"   : round(chrf.score, 4),
        "BLEU"   : round(bleu.score, 4),
        "TER"    : round(ter.score,  4),
        "BLEURT" : round(sum(blrt) / len(blrt), 4),
        "COMET"  : round(sum(comet) / len(comet), 4),
    }


def sentence_metrics(hyp, ref):
    chrf = sacrebleu.sentence_chrf(hyp, [ref]).score
    bleu = sacrebleu.sentence_bleu(hyp, [ref]).score
    ter  = sacrebleu.sentence_ter(hyp, [ref]).score
    blrt = bleurt_scorer.score(references=[ref], candidates=[hyp])[0]
    comet = comet_scores([""], [hyp], [ref])[0]
    return chrf, bleu, ter, blrt, comet


def evaluate_only(model, tokenizer, label="EVAL", n_samples=1000):
    hyps, skipped = translate_all(model, tokenizer, gold_es[:n_samples], gold_va[:n_samples], label)
    metrics = compute_metrics(hyps, gold_va[:n_samples], skipped, label.lower())
    metrics["model"] = label.upper()
    metrics["skipped_indices"] = skipped
    print(
        f"  {label.upper()} -> chrF: {metrics['chrF']:.2f} | "
        f"BLEU: {metrics['BLEU']:.2f} | COMET: {metrics['COMET']:.4f}"
    )
    return metrics, hyps, skipped


# --- Test set -------------------------------------------------------------

eval_dataset = load_dataset("gplsi/ES-VA_translation_test", split="test")

eval_sorted = eval_dataset.map(lambda x: {"len": len(x[eval_src])})
eval_sorted = eval_sorted.sort("len", reverse=True)
eval_raw    = eval_sorted.select(range(eval_n))

gold_es = [ex[eval_src] for ex in eval_raw]
gold_va = [ex[eval_tgt] for ex in eval_raw]

print(f"Selected  : {len(gold_es)} sentences")
print(f"Avg length: {sum(len(s) for s in gold_es) / len(gold_es):.1f} chars")


# --- 1.Baseline -------------------------------------------------------------

tok_base   = AutoTokenizer.from_pretrained(base_model_id)
model_base = AutoModelForCausalLM.from_pretrained(
    base_model_id, quantization_config=bnb_config, device_map="auto"
)

m_base, hyps_base, skip_base = evaluate_only(model_base, tok_base, "BASE", eval_n)

with open("baselinetranslategemma_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "dataset": "gplsi/ES-VA_translation_test",
        "n_total": eval_n,
        "results": [m_base],
        "samples": [{"id": i, "source_es": gold_es[i], "reference_va": gold_va[i],
                     "baseline": hyps_base[i]} for i in range(eval_n)]
    }, f, ensure_ascii=False, indent=2)

del model_base
torch.cuda.empty_cache()
gc.collect()


# --- 2.SFT -------------------------------------------------------------

base_sft  = AutoModelForCausalLM.from_pretrained(
    base_model_id, quantization_config=bnb_config, device_map="auto"
)
tok_sft   = AutoTokenizer.from_pretrained(base_model_id)
model_sft = PeftModel.from_pretrained(base_sft, sft_model_id)
model_sft.eval()

m_sft, hyps_sft, skip_sft = evaluate_only(model_sft, tok_sft, "SFT", eval_n)

with open("sft_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "dataset": "gplsi/ES-VA_translation_test",
        "n_total": eval_n,
        "results": [m_sft],
        "samples": [{"id": i, "source_es": gold_es[i], "reference_va": gold_va[i],
                     "sft": hyps_sft[i]} for i in range(eval_n)]
    }, f, ensure_ascii=False, indent=2)

del model_sft, base_sft
torch.cuda.empty_cache()
gc.collect()



# --- 3. GRPOv1 -------------------------------------------------------------

tok_grpov1   = AutoTokenizer.from_pretrained(grpov1_model_id)
model_grpov1 = AutoModelForCausalLM.from_pretrained(
    grpov1_model_id, quantization_config=bnb_config, device_map="auto"
)
model_grpov1.eval()

m_grpov1, hyps_grpov1, skip_grpov1 = evaluate_only(model_grpov1, tok_grpov1, "GRPOv1", eval_n)

with open("grpov1_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "dataset": "gplsi/ES-VA_translation_test",
        "n_total": eval_n,
        "results": [m_grpov1],
        "samples": [{"id": i, "source_es": gold_es[i], "reference_va": gold_va[i],
                     "grpov1": hyps_grpov1[i]} for i in range(eval_n)]
    }, f, ensure_ascii=False, indent=2)

del model_grpov1
torch.cuda.empty_cache()
gc.collect()



# --- 4.GRPOv2 -------------------------------------------------------------

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, quantization_config=bnb_config, device_map="auto"
)
model_grpov2 = PeftModel.from_pretrained(base_model, grpov2_model_id)
tok_grpov2   = AutoTokenizer.from_pretrained(base_model_id)

m_grpov2, hyps_grpov2, skip_grpov2 = evaluate_only(model_grpov2, tok_grpov2, "GRPOv2", eval_n)

with open("grpov2_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "dataset": "gplsi/ES-VA_translation_test",
        "n_total": eval_n,
        "results": [m_grpov2],
        "samples": [{"id": i, "source_es": gold_es[i], "reference_va": gold_va[i],
                     "grpov2": hyps_grpov2[i]} for i in range(eval_n)]
    }, f, ensure_ascii=False, indent=2)

del model_grpov2
torch.cuda.empty_cache()
gc.collect()


# --- Summary -------------------------------------------------------------

all_metrics = [m_base, m_sft, m_grpov1, m_grpov2]
hyps_all    = {
    "baseline": hyps_base,
    "sft":      hyps_sft,
    "grpov1":   hyps_grpov1,
    "grpov2":   hyps_grpov2,
}

best_chrf   = max(m["chrF"]   for m in all_metrics)
best_bleurt = max(m["BLEURT"] for m in all_metrics)
best_comet  = max(m["COMET"]  for m in all_metrics)

print("\n" + "="*84)
print(f"  {'Model':<10} {'chrF':>7} {'BLEU':>7} {'TER':>7} {'BLEURT':>8} {'COMET':>8} {'Eval':>5} {'Skip':>5}")
print("="*84)
for m in all_metrics:
    print(f"  {m['model']:<10} {m['chrF']:>7.2f} {m['BLEU']:>7.2f} "
          f"{m['TER']:>7.2f} {m['BLEURT']:>8.4f} {m['COMET']:>8.4f} {m['n_eval']:>5} {m['skipped']:>5}")
print("="*84)
print("  TER: lower is better  |  chrF / BLEU / BLEURT / COMET: higher is better")

with open("eval_results_combined.json", "w", encoding="utf-8") as f:
    json.dump({
        "dataset": "gplsi/ES-VA_translation_test",
        "n_total": eval_n,
        "results": all_metrics,
        "samples": [
            {
                "id"          : i,
                "source_es"   : gold_es[i],
                "reference_va": gold_va[i],
                "baseline"    : hyps_base[i],
                "sft"         : hyps_sft[i],
                "grpov1"      : hyps_grpov1[i],
                "grpov2"      : hyps_grpov2[i],
            }
            for i in range(eval_n)
        ]
    }, f, ensure_ascii=False, indent=2)


# --- Dialectal valencian score-------------------------------------------------------------

def dialectal_score(hypotheses, label):
    valid_hyps = [h.lower() for h in hypotheses if h not in ("[SKIPPED]", "[EMPTY]", None)]
    corpus = " ".join(valid_hyps)

    per_feature = {}
    total_va, total_ca = 0, 0

    for ca_form, va_form in ca_va_features.items():
        va_hits = len(re.findall(r'\b' + re.escape(va_form) + r'\b', corpus))
        ca_hits = len(re.findall(r'\b' + re.escape(ca_form) + r'\b', corpus))
        total   = va_hits + ca_hits
        va_rate = va_hits / total if total > 0 else None
        per_feature[ca_form] = {
            "va_form": va_form,
            "va_hits": va_hits,
            "ca_hits": ca_hits,
            "va_rate": va_rate,
        }
        total_va += va_hits
        total_ca += ca_hits

    total   = total_va + total_ca
    overall = total_va / total if total > 0 else 0.0
    print(f"[{label}] Overall DVS: {overall:.2%}  (VA: {total_va} | CA: {total_ca})")
    return overall, per_feature


scores = {}
feats  = {}
for model_key in models:
    scores[model_key], feats[model_key] = dialectal_score(hyps_all[model_key], model_key.upper())



fig, ax = plt.subplots(figsize=(7, 4.5))
vals     = [scores[m]*100 for m in models]
best_val = max(vals)
bars     = ax.bar(range(len(models)), vals, color=[colors[m] for m in models],
                  width=0.55, zorder=3, edgecolor="white")
for bar, m, v in zip(bars, models, vals):
    bar.set_alpha(1.0 if v == best_val else 0.72)
    ax.text(bar.get_x()+bar.get_width()/2, v+0.8, f"{v:.1f}%",
            ha="center", va="bottom", fontsize=10,
            fontweight="bold" if v == best_val else "normal")
ax.set_xticks(range(len(models)))
ax.set_xticklabels([labels[m] for m in models], fontsize=10)
ax.set_ylabel("Valencian Form Usage Rate (%)", fontsize=10)
ax.set_title("Dialectal Valencian Score", fontsize=12, fontweight="bold")
ax.set_ylim(0, max(vals)*1.2)
ax.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig("fig_dialectal_score.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: fig_dialectal_score.png")


rows = []
for ca, va in ca_va_features.items():
    row = {"CA form": ca, "VA form": va}
    for m in models:
        r = feats[m][ca]["va_rate"]
        row[f"{labels[m]} VA%"]   = f"{r:.0%}" if r is not None else "-"
        row[f"{labels[m]} va/ca"] = f"{feats[m][ca]['va_hits']}/{feats[m][ca]['ca_hits']}"
    rows.append(row)

df_feat = pd.DataFrame(rows).sort_values("CA form")
print(df_feat.to_string(index=False))



dialect_summary = {
    "dataset": "gplsi/ES-VA_translation_test",
    "n_total": len(gold_va),
    "dialectal_scores": {labels[m]: round(scores[m], 4) for m in models},
    "per_feature": {
        ca: {
            "va_form": va,
            **{labels[m]: {
                "va_hits": feats[m][ca]["va_hits"],
                "ca_hits": feats[m][ca]["ca_hits"],
                "va_rate": round(feats[m][ca]["va_rate"], 4) if feats[m][ca]["va_rate"] is not None else None,
            } for m in models}
        }
        for ca, va in ca_va_features.items()
    }
}

with open("eval_dialect_summary.json", "w", encoding="utf-8") as f:
    json.dump(dialect_summary, f, ensure_ascii=False, indent=2)

print("\nDialectal VA Score Summary:")
print("-"*40)
for m in models:
    print(f"  {labels[m]:<12}: {scores[m]:.1%}")

# utils/metrics.py
# Evaluation metrics (chrF, BLEU, TER, BLEURT, COMET) and dialectal analysis.
# Paula Guerrero Castelló, May 2026

import re
import json
import sacrebleu
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Dict, Tuple, Optional

from config import Config


# ── Lazy-loaded heavy scorers ─────────────────────────────────────────────────

_bleurt_scorer = None
_comet_model   = None


def get_bleurt():
    global _bleurt_scorer
    if _bleurt_scorer is None:
        from bleurt import score as bleurt_score
        _bleurt_scorer = bleurt_score.BleurtScorer("bleurt-base-128")
    return _bleurt_scorer


def get_comet():
    global _comet_model
    if _comet_model is None:
        from comet import download_model, load_from_checkpoint
        path = download_model("Unbabel/wmt22-comet-da")
        _comet_model = load_from_checkpoint(path)
    return _comet_model


# ── Sentence-level helpers ────────────────────────────────────────────────────

def comet_corpus(sources: List[str], hyps: List[str], refs: List[str]) -> List[float]:
    model  = get_comet()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data   = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hyps, refs)]
    out    = model.predict(data, batch_size=8, gpus=1 if device == "cuda" else 0)
    return out.scores if hasattr(out, "scores") else out[0]


def sentence_metrics(
    hyp: str,
    ref: str,
    src: str = "",
) -> Dict[str, float]:
    chrf  = sacrebleu.sentence_chrf(hyp, [ref]).score
    bleu  = sacrebleu.sentence_bleu(hyp, [ref]).score
    ter   = sacrebleu.sentence_ter(hyp, [ref]).score
    blrt  = get_bleurt().score(references=[ref], candidates=[hyp])[0]
    comet = comet_corpus([src or hyp], [hyp], [ref])[0]
    ttr   = _ttr(hyp)
    return dict(chrF=chrf, BLEU=bleu, TER=ter, BLEURT=blrt, COMET=comet, TTR=ttr)


def _ttr(text: str) -> float:
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    ttr = len(set(tokens)) / len(tokens)
    if len(tokens) < 5:
        ttr *= len(tokens) / 5.0
    return float(ttr)


# ── Corpus-level evaluation ───────────────────────────────────────────────────

def compute_corpus_metrics(
    model_label: str,
    hyps: List[str],
    refs: List[str],
    sources: List[str],
    skipped_idx: List[int],
) -> Dict:
    kept = [i for i in range(len(hyps)) if i not in skipped_idx]
    h = [hyps[i] for i in kept]
    r = [refs[i]  for i in kept]
    s = [sources[i] for i in kept]

    chrf  = sacrebleu.corpus_chrf(h, [r]).score
    bleu  = sacrebleu.corpus_bleu(h, [r]).score
    ter   = sacrebleu.corpus_ter(h, [r]).score
    blrt  = get_bleurt().score(references=r, candidates=h)
    comet = comet_corpus(s, h, r)

    return {
        "model":   model_label,
        "n_eval":  len(h),
        "skipped": len(skipped_idx),
        "chrF":    round(chrf, 4),
        "BLEU":    round(bleu, 4),
        "TER":     round(ter, 4),
        "BLEURT":  round(float(np.mean(blrt)), 4),
        "COMET":   round(float(np.mean(comet)), 4),
    }


def compute_metrics(
    model_label: str,
    hyps: List[str],
    refs: List[str],
    sources: List[str],
    skipped_idx: List[int],
) -> Dict:
    """Compatibility alias for corpus-level evaluation."""
    return compute_corpus_metrics(model_label, hyps, refs, sources, skipped_idx)


def print_results_table(all_metrics: List[Dict]) -> None:
    print("\n" + "=" * 84)
    print(f"  {'Model':<10} {'chrF':>7} {'BLEU':>7} {'TER':>7} {'BLEURT':>8} {'COMET':>8} {'Eval':>5} {'Skip':>5}")
    print("=" * 84)
    for m in all_metrics:
        print(
            f"  {m['model']:<10} {m['chrF']:>7.2f} {m['BLEU']:>7.2f} "
            f"{m['TER']:>7.2f} {m['BLEURT']:>8.4f} {m['COMET']:>8.4f} "
            f"{m['n_eval']:>5} {m['skipped']:>5}"
        )
    print("=" * 84)
    print("  TER: lower is better  |  chrF / BLEU / BLEURT / COMET: higher is better")


# ── Dialectal Valencian score ─────────────────────────────────────────────────

def dialectal_score(
    hypotheses: List[str],
    label: str,
    cfg: Config,
) -> Tuple[float, Dict]:
    """
    Return overall VA-form rate and per-feature breakdown.
    Skipped / empty hypotheses are excluded.
    """
    valid = [h.lower() for h in hypotheses if h not in ("[SKIPPED]", "[EMPTY]", None)]
    corpus = " ".join(valid)

    per_feature: Dict = {}
    total_va, total_ca = 0, 0

    for ca_form, va_form in cfg.ca_va_features.items():
        va_hits = len(re.findall(r'\b' + re.escape(va_form) + r'\b', corpus))
        ca_hits = len(re.findall(r'\b' + re.escape(ca_form) + r'\b', corpus))
        total   = va_hits + ca_hits
        per_feature[ca_form] = {
            "va_form": va_form,
            "va_hits": va_hits,
            "ca_hits": ca_hits,
            "va_rate": va_hits / total if total > 0 else None,
        }
        total_va += va_hits
        total_ca += ca_hits

    total   = total_va + total_ca
    overall = total_va / total if total > 0 else 0.0
    print(f"[{label}] Dialectal VA Score: {overall:.2%}  (VA: {total_va} | CA: {total_ca})")
    return overall, per_feature


def save_dialect_summary(
    scores: Dict[str, float],
    feats:  Dict[str, Dict],
    cfg: Config,
    out_path: Path,
) -> None:
    labels = list(scores.keys())
    summary = {
        "dataset":           "gplsi/ES-VA_translation_test",
        "n_total":           cfg.eval_n,
        "dialectal_scores":  {k: round(v, 4) for k, v in scores.items()},
        "per_feature": {
            ca: {
                "va_form": va,
                **{
                    lbl: {
                        "va_hits": feats[lbl][ca]["va_hits"],
                        "ca_hits": feats[lbl][ca]["ca_hits"],
                        "va_rate": (
                            round(feats[lbl][ca]["va_rate"], 4)
                            if feats[lbl][ca]["va_rate"] is not None
                            else None
                        ),
                    }
                    for lbl in labels
                },
            }
            for ca, va in cfg.ca_va_features.items()
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Dialect summary saved → {out_path}")


def plot_dialectal_scores(
    scores: Dict[str, float],
    save_path: Path,
    colors: Optional[Dict[str, str]] = None,
) -> None:
    if colors is None:
        colors = {}
    models = list(scores.keys())
    vals   = [scores[m] * 100 for m in models]
    best   = max(vals)
    default_colors = ["#6c757d", "#1f77b4", "#ff7f0e", "#2ca02c"]
    bar_colors = [colors.get(m, default_colors[i % len(default_colors)]) for i, m in enumerate(models)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(range(len(models)), vals, color=bar_colors, width=0.55,
                  zorder=3, edgecolor="white")
    for bar, v in zip(bars, vals):
        bar.set_alpha(1.0 if v == best else 0.72)
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8,
                f"{v:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold" if v == best else "normal")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Valencian Form Usage Rate (%)")
    ax.set_title("Dialectal Valencian Score", fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.2)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Dialectal plot saved → {save_path}")

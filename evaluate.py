# evaluate.py
# Full evaluation: Baseline vs SFT vs GRPOv1 vs GRPOv2.
# Metrics: chrF, BLEU, TER, BLEURT, COMET + Dialectal Valencian Score.
# Paula Guerrero Castelló, May 2026

import gc
import json
import torch

from config import Config
from utils.data import load_test_set, make_inference_prompt
from utils.metrics import (
    compute_corpus_metrics,
    dialectal_score,
    save_dialect_summary,
    plot_dialectal_scores,
    print_results_table,
)
from utils.model import build_bnb_config, get_compute_dtype, load_base_tokenizer

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from peft import PeftModel


# ── Inference helper ──────────────────────────────────────────────────────────

def translate_all(model, tokenizer, sources, refs, label, cfg):
    model.eval()
    hyps, skipped_idx = [], []

    for i, src in enumerate(sources):
        prompt = make_inference_prompt(src, tokenizer, cfg)
        n_tok  = len(tokenizer(prompt).input_ids)

        if n_tok > cfg.eval_max_seq:
            hyps.append("[SKIPPED]")
            skipped_idx.append(i)
            continue

        src_len = len(tokenizer(src).input_ids)
        max_new = min(512, max(80, int(src_len * 1.2)))

        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=cfg.eval_max_seq,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )

        new_tok = out[0][inputs["input_ids"].shape[1]:]
        hyp     = tokenizer.decode(new_tok, skip_special_tokens=True).strip()
        hyps.append(hyp)

        import sacrebleu
        chrf_s = sacrebleu.sentence_chrf(hyp, [refs[i]]).score
        print(f"[{label}] [{i+1:3d}/{len(sources)}] chrF={chrf_s:5.1f} | {src[:55]}")

    return hyps, skipped_idx


def _save_results(path, hyps, sources, refs, metrics, label_key):
    data = {
        "dataset":  "gplsi/ES-VA_translation_test",
        "n_total":  len(sources),
        "results":  [metrics],
        "samples": [
            {"id": i, "source_es": sources[i], "reference_va": refs[i], label_key: hyps[i]}
            for i in range(len(sources))
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Results saved → {path}")


def _load_eval_model(cfg: Config, label: str, bnb):
    if label == "baseline":
        tokenizer = load_base_tokenizer(cfg)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_id,
            quantization_config=bnb,
            device_map="auto",
            token=cfg.hf_token or None,
            use_safetensors=True,
        )
        return model, tokenizer

    if label == "sft":
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_id,
            quantization_config=bnb,
            device_map="auto",
            token=cfg.hf_token or None,
            use_safetensors=True,
        )
        tokenizer = load_base_tokenizer(cfg)
        model = PeftModel.from_pretrained(
            base_model,
            cfg.sft_model_id,
            token=cfg.hf_token or None,
            use_safetensors=True,
        )
        return model, tokenizer

    if label == "grpov1":
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.grpov1_model_id,
            token=cfg.hf_token or None,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(
            cfg.grpov1_model_id,
            quantization_config=bnb,
            device_map="auto",
            token=cfg.hf_token or None,
            use_safetensors=True,
        )
        return model, tokenizer

    if label == "grpov2":
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model_id,
            quantization_config=bnb,
            device_map="auto",
            token=cfg.hf_token or None,
            use_safetensors=True,
        )
        tokenizer = load_base_tokenizer(cfg)
        model = PeftModel.from_pretrained(
            base_model,
            cfg.grpov2_model_id,
            token=cfg.hf_token or None,
            use_safetensors=True,
        )
        return model, tokenizer

    raise ValueError(f"Unsupported evaluation label: {label}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_evaluation(cfg: Config) -> None:
    gold_es, gold_va = load_test_set(cfg)
    bnb              = build_bnb_config(cfg)
    results_dir      = cfg.output_root / "eval"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    all_hyps    = {}

    models_to_eval = ["baseline", "sft", "grpov1", "grpov2"]

    for label in models_to_eval:
        print(f"\n{'='*55}")
        print(f"  Evaluating: {label.upper()}")
        print(f"{'='*55}")

        model, tokenizer = _load_eval_model(cfg, label, bnb)
        model.eval()
        hyps, skipped = translate_all(model, tokenizer, gold_es, gold_va, label.upper(), cfg)
        metrics = compute_corpus_metrics(label.upper(), hyps, gold_va, gold_es, skipped)

        all_metrics.append(metrics)
        all_hyps[label] = hyps

        _save_results(
            results_dir / f"{label}_results.json",
            hyps, gold_es, gold_va, metrics, label,
        )

        del model
        torch.cuda.empty_cache(); gc.collect()

    # Summary table
    print_results_table(all_metrics)

    # Combined results file
    combined = {
        "dataset": "gplsi/ES-VA_translation_test",
        "n_total": cfg.eval_n,
        "results": all_metrics,
        "samples": [
            {
                "id":           i,
                "source_es":    gold_es[i],
                "reference_va": gold_va[i],
                **{lbl: all_hyps[lbl][i] for lbl in all_hyps},
            }
            for i in range(cfg.eval_n)
        ],
    }
    combined_path = results_dir / "eval_results_combined.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"\nCombined results saved → {combined_path}")

    # Dialectal analysis
    model_labels = ["baseline", "sft", "grpov1", "grpov2"]
    scores, feats = {}, {}
    for lbl in model_labels:
        scores[lbl.upper()], feats[lbl.upper()] = dialectal_score(all_hyps[lbl], lbl.upper(), cfg)

    save_dialect_summary(scores, feats, cfg, results_dir / "eval_dialect_summary.json")
    plot_dialectal_scores(scores, results_dir / "fig_dialectal_score.png")

    print("\nDialectal VA Score Summary:")
    print("-" * 40)
    for lbl, score in scores.items():
        print(f"  {lbl:<12}: {score:.1%}")

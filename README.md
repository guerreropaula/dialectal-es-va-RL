# Enhancing LLM Translation Performance for Spanish-Valencian through Supervised Fine-tuning and Reinforcement Learning

**Fine-tuning `google/translategemma-4b-it` for the low-resource ES-VA direction using SFT and GRPO**

> MSc Language Analysis and Processing · University of the Basque Country · 2025–2026  
> Author: Paula Guerrero

---

## Overview
This project adapts **TranslateGemma-4B** using three strategies:

| Strategy | Description |
|---|---|
| **Baseline** | `google/translategemma-4b-it` zero-shot (ES→CA, no Valencian awareness) |
| **SFT** | Supervised fine-tuning on 50k ES→VA pairs from `gplsi/amic_parallel` |
| **GRPO v1** | RL from SFT checkpoint — reward: chrF + P(HT\|text) translationese classifier |
| **GRPO v2** ★ | RL from SFT checkpoint — reward: chrF + COMET + TTR + copy penalty — **best** |

---

## Results

Evaluated on 1,000 sentences from [`gplsi/ES-VA_translation_test`](https://huggingface.co/datasets/gplsi/ES-VA_translation_test).

| Model | chrF ↑ | BLEU ↑ | TER ↓ | BLEURT ↑ | COMET ↑ | Dialectal VA ↑ |
|---|---|---|---|---|---|---|
| Baseline | 69.02 | 39.22 | 40.30 | 0.258 | 0.906 | 3.2% |
| SFT | 83.16 | 60.16 | 22.80 | 0.524 | 0.934 | **41.0%** |
| GRPO v1 | 81.65 | 56.94 | 23.96 | 0.481 | 0.926 | 15.9% |
| **GRPO v2** ★ | **84.68** | **62.16** | **20.63** | **0.544** | **0.936** | 36.2% |

<img src="assets/translation_metrics.jpeg" width="600">
<br>

<img src="assets/multidimensional_eval.jpeg" width="450">
<br>

<img src="assets/dialectal_score.jpeg" width="450">

---

## Pipeline

```
google/translategemma-4b-it (baseline)
        │
        ▼
   SFT on 50k ES→VA pairs (gplsi/amic_parallel)
   LoRA r=16, α=32, lr=2e-4, 2000 steps
        │
        ├──────────────────────────────────────────┐
        ▼                                          ▼
  GRPO v1 (5k pairs)                     GRPO v2 (10k pairs) ★
  reward = (1-α)·chrF + α·P(HT|text)     reward = 0.5·chrF + 0.3·COMET
  α: 0→0.3 warm-up over 50 steps                 + 0.2·TTR − copy_penalty
  200 steps, lr=5e-6, β=0.04             200 steps, lr=5e-6, β=0.04
```

---

## Repository Structure

```
.
├── notebooks/
│   ├── 01_sft.ipynb                  # SFT — LoRA fine-tuning on 50k ES→VA pairs
│   ├── 02_ht_mt_classifier.ipynb     # HT vs MT translationese classifier (RoBERTa-ca)
│   ├── 03_grpo_v1.ipynb              # GRPO v1 — reward: chrF + P(HT|text)
│   ├── 04_grpo_v2.ipynb              # GRPO v2 — composite reward ★
│   └── 05_evaluation.ipynb           # Full evaluation + dialectal analysis
├── results/
│   ├── summary_metrics.xlsx          # Final metrics for all 4 models
│   └── eval_results_1k.xlsx          # Per-sentence metrics (1k samples)
├── assets/
│   ├── translation_metrics.jpeg
│   ├── multidimensional_eval.jpeg
│   └── dialectal_score.jpeg
├── requirements.txt
└── README.md
```

---

## HuggingFace Models

| Model | Hub |
|---|---|
| SFT | [`guerreropaula/translategemma4b-sft-es-va2`](https://huggingface.co/guerreropaula/translategemma4b-sft-es-va2) |
| GRPO v1 | [`guerreropaula/80translategemma4b-grpo-es-va`](https://huggingface.co/guerreropaula/80translategemma4b-grpo-es-va) |
| GRPO v2 ★ | [`guerreropaula/translategemma4b-grpo2-es-va`](https://huggingface.co/guerreropaula/translategemma4b-grpo2-es-va) |
| HT/MT Classifier | [`guerreropaula/ht_mt_classifier_best`](https://huggingface.co/guerreropaula/ht_mt_classifier_best) |

---

## Datasets

| Dataset | Usage |
|---|---|
| [`gplsi/amic_parallel`](https://huggingface.co/datasets/gplsi/amic_parallel) | SFT + GRPO training (ES–VA parallel) |
| [`gplsi/ES-VA_translation_test`](https://huggingface.co/datasets/gplsi/ES-VA_translation_test) | Held-out evaluation |
| [Softcatalà parallel corpus](https://github.com/Softcatala/parallel-catalan-corpus) | HT/MT classifier training |

---

Run notebooks in order on **Google Colab** (A100 recommended):

1. `01_sft.ipynb` — SFT training
2. `02_ht_mt_classifier.ipynb` — train the translationese classifier first
3. `03_grpo_v1.ipynb` — GRPO v1
4. `04_grpo_v2.ipynb` — GRPO v2 ★
5. `05_evaluation.ipynb` — full evaluation for 4 approaches + dialectal analysis

---

## Reward Functions

**GRPO v1**
```python
r = (1 - α) * chrF(hyp, ref) / 100 + α * P(HT | hyp)
# α = 0 → 0.3  (linear warm-up after 50 steps)
```

**GRPO v2** ★
```python
r = 0.5 * chrF(hyp, ref) + 0.3 * COMET(src, hyp, ref) + 0.2 * TTR(hyp) + copy_penalty(src, hyp)
# copy_penalty = -1.0 if hyp == src else 0.0
```

---

## Citation

```bibtex
@misc{guerrero2026translategemma_va,
  title   = {Enhancing LLM Translation Performance for Spanish-Valencian through Supervised Fine-tuning and Reinforcement Learning},
  author  = {Guerrero, Paula},
  year    = {2026},
  note    = {MSc Deep Learning Project, University of the Basque Country}
}
```

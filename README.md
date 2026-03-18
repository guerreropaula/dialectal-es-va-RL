# TranslateGemma-4B Fine-tuning: Spanish → Valencian NMT

**Fine-tuning `google/translategemma-4b-it` for the low-resource ES→VA translation task using SFT and GRPO with composite reward shaping.**

> MSc Deep Learning Project · Universitat Politècnica de València · 2025–2026  
> Author: Paula Guerrero

---

## Overview

Valencian is a southern variety of Catalan with distinct morphological and lexical features. Standard multilingual MT systems conflate it with standard Catalan. This project adapts **TranslateGemma-4B** using three strategies:

| Strategy | Description |
|---|---|
| **Baseline** | `google/translategemma-4b-it` zero-shot (ES→CA, no Valencian awareness) |
| **SFT** | Supervised fine-tuning on 50k ES→VA pairs from `gplsi/amic_parallel` |
| **GRPO v1** | RL from SFT checkpoint — composite reward: chrF + HT/MT translationese classifier |
| **GRPO v2** | RL from SFT checkpoint — composite reward: chrF + COMET + TTR + copy penalty |

---

## Results (1,000 held-out test sentences — `gplsi/ES-VA_translation_test`)

| Model | chrF ↑ | BLEU ↑ | TER ↓ | BLEURT ↑ | COMET ↑ | Dialectal VA ↑ | Composite ↑ |
|---|---|---|---|---|---|---|---|
| Baseline | 69.02 | 39.22 | 40.30 | 0.258 | 0.906 | 3.2% | 0.809 |
| SFT | 83.16 | 60.16 | 22.80 | 0.524 | 0.934 | **41.0%** | 0.835 |
| **GRPO v1** ★ | **84.68** | **62.16** | **20.63** | **0.544** | **0.936** | 36.2% | **0.882** |
| GRPO v2 | 81.65 | 56.94 | 23.96 | 0.481 | 0.926 | 15.9% | 0.864 |

★ Best on all primary translation quality metrics.

---

## Repository Structure

```
.
├── notebooks/
│   ├── 01_sft_grpo_v1_train_eval.ipynb   # SFT training + GRPO v1 (chrF + HT/MT classifier) + evaluation
│   ├── 02_ht_mt_classifier.ipynb          # HT vs MT translationese classifier training
│   ├── 03_grpo_v2_composite_reward.ipynb  # GRPO v2 (chrF + COMET + TTR + copy penalty)
│   └── 04_evaluation_dialect_analysis.ipynb # Dialect analysis from pre-computed outputs
├── results/
│   ├── summary_metrics.csv               # Final metrics for all 4 models
│   └── eval_results_1k.xlsx              # Per-sentence metrics for all 4 models (1k samples)
├── scripts/
│   └── dialectal_score.py                # Standalone dialectal VA scorer
├── assets/
│   ├── fig1_translation_metrics.png
│   ├── fig2_neural_metrics.png
│   ├── fig3_dialectal_score.png
│   ├── fig4_composite_reward.png
│   └── fig5_radar.png
├── requirements.txt
└── README.md
```

---

## Pipeline

```
google/translategemma-4b-it (baseline)
        │
        ▼
   SFT on 50k ES→VA pairs (gplsi/amic_parallel)
   LoRA r=16, α=32, lr=2e-4, 2000 steps
        │
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
  GRPO v1 (5k pairs)                         GRPO v2 (10k pairs)
  reward = (1-α)·chrF + α·P(HT|text)         reward = 0.5·chrF + 0.3·COMET
  α: 0→0.3 warm-up over 50 steps                     + 0.2·TTR − copy_penalty
  200 steps, lr=5e-6, β=0.04                 200 steps, lr=5e-6, β=0.04
```

---

## HuggingFace Models

| Model | HuggingFace Hub |
|---|---|
| SFT | [`guerreropaula/translategemma4b-sft-es-va2`](https://huggingface.co/guerreropaula/translategemma4b-sft-es-va2) |
| GRPO v1 | [`guerreropaula/80translategemma4b-grpo-es-va`](https://huggingface.co/guerreropaula/80translategemma4b-grpo-es-va) |
| GRPO v2 | [`guerreropaula/translategemma4b-grpo2-es-va`](https://huggingface.co/guerreropaula/translategemma4b-grpo2-es-va) |
| HT/MT Classifier | [`guerreropaula/ht_mt_classifier_best`](https://huggingface.co/guerreropaula/ht_mt_classifier_best) |

---

## Datasets

| Dataset | Usage |
|---|---|
| [`gplsi/amic_parallel`](https://huggingface.co/datasets/gplsi/amic_parallel) | SFT + GRPO training (ES–VA parallel) |
| [`gplsi/ES-VA_translation_test`](https://huggingface.co/datasets/gplsi/ES-VA_translation_test) | Evaluation test set (held-out) |
| [Softcatalà parallel corpus](https://github.com/Softcatala/parallel-catalan-corpus) | HT/MT classifier training (ES–CA) |

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/translategemma-es-va.git
cd translategemma-es-va
pip install -r requirements.txt
```

Run notebooks in order on **Google Colab** (A100 recommended):

1. `02_ht_mt_classifier.ipynb` — train the translationese classifier first
2. `01_sft_grpo_v1_train_eval.ipynb` — SFT training, GRPO v1, and main evaluation
3. `03_grpo_v2_composite_reward.ipynb` — GRPO v2 training
4. `04_evaluation_dialect_analysis.ipynb` — dialect analysis (requires results XLSX)

---

## Reward Functions

### GRPO v1 — chrF + HT/MT Classifier
```python
r_final = (1 - α) * r_c + α * r_t

# r_c = chrF(hypothesis, reference) / 100   ∈ [0, 1]
# r_t = P(HT | hypothesis)                  ∈ [0, 1]  (from roberta-base-ca classifier)
# α   = 0 → 0.3  (linear warm-up after 50 steps)
```

### GRPO v2 — Multi-component Composite
```python
r = 0.5 * chrF(hyp, ref) + 0.3 * COMET(src, hyp, ref) + 0.2 * TTR(hyp) + copy_penalty(src, hyp)

# TTR = type-token ratio  (penalises vocabulary collapse)
# copy_penalty = -1.0 if hyp == src  else 0.0
```

---

## Dialectal Valencian Score

We measure how often models produce Valencian-specific morpho-lexical forms vs standard Catalan equivalents, using regex matching on a 30-feature contrastive vocabulary:

| CA form | VA form | Example |
|---|---|---|
| aquesta | esta | *esta característica* |
| seva | seua | *la seua opinió* |
| tenir | tindre | *ha de tindre en compte* |
| feina | faena | *la faena del dia* |
| nens | xiquets | *els xiquets de l'escola* |

Run the standalone scorer:
```bash
python scripts/dialectal_score.py --input your_hypotheses.txt
```

---

## Citation

```bibtex
@misc{guerrero2026translategemma_va,
  title     = {Fine-tuning TranslateGemma-4B for Spanish–Valencian Neural Machine Translation
               via SFT and GRPO with Composite Reward Shaping},
  author    = {Guerrero, Paula},
  year      = {2026},
  institution = {Universitat Politècnica de València},
  note      = {MSc Deep Learning Project}
}
```

---

## License

Code: MIT. Model weights follow the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

# Enhancing LLM Translation Performance for Spanish–Valencian through Supervised Fine-Tuning and Reinforcement Learning

> **EAMT 2026** · University of the Basque Country (UPV/EHU)  
> Paula Guerrero Castelló · `pguerrero005@ikasle.ehu.eus`

This repository contains the code, models, and evaluation scripts for our EAMT 2026 paper on adapting a translation-specialized LLM to the low-resource Spanish→Valencian (ES→VA) direction using supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO).

---

## Abstract

Valencian lacks a dedicated language code in most multilingual MT systems and is systematically rendered as standard (Eastern) Catalan. We adapt **TranslateGemma-4B-IT** through three post-training strategies—SFT, GRPO with a learned naturalness classifier (GRPOv1), and GRPO with a composite automatic-metric reward (GRPOv2)—using only publicly available corpora and QLoRA-based training on commodity hardware. Our results show that SFT delivers the largest single-step gain, while GRPOv2 surpasses SFT on all five standard MT metrics. A classifier-based naturalness reward (GRPOv1) causes dialect suppression, underscoring that reward-function alignment with the target variety is critical for dialectal MT.

---

## Results

Evaluated on 1,000 sentences from [`gplsi/ES-VA_translation_test`](https://huggingface.co/datasets/gplsi/ES-VA_translation_test).

### Automatic MT Quality

| System | chrF ↑ | BLEU ↑ | TER ↓ | BLEURT ↑ | COMET ↑ |
|---|---|---|---|---|---|
| Baseline | 69.02 | 39.22 | 40.30 | 0.258 | 0.906 |
| SFT | 83.16 | 60.16 | 22.80 | 0.524 | 0.934 |
| GRPOv1 | 81.65 | 56.94 | 23.96 | 0.481 | 0.926 |
| **GRPOv2** ★ | **84.68** | **62.16** | **20.63** | **0.544** | **0.936** |

### Dialectal Valencian Score (DVS)

DVS measures the rate at which a system produces Valencian-specific morpho-lexical forms over 30 contrastive CA–VA pairs.

| System | DVS ↑ |
|---|---|
| Baseline | 3.2% |
| **SFT** | **41.0%** |
| GRPOv1 | 15.9% |
| GRPOv2 ★ | 36.2% |

### Feature-level DVS (selected pairs)

| CA form | VA form | Baseline | SFT | GRPOv1 | GRPOv2 |
|---|---|---|---|---|---|
| *petit* | *xicotet* | 0% | 100% | 0% | 100% |
| *veure* | *vore* | 0% | 83% | 0% | 17% |
| *avui* | *hui* | 0% | 71% | 0% | 86% |
| *seva* | *seua* | 0% | 100% | 45% | 100% |

---

## Training Pipeline

```
TranslateGemma-4B-IT (zero-shot baseline)
        │
        ▼
   SFT — 50k ES–VA pairs (gplsi/amic_parallel)
   QLoRA r=16, α=32, 4-bit NF4, 3 epochs
        │
        ├─────────────────────────┐
        ▼                         ▼
   GRPOv1 (5k pairs)         GRPOv2 (10k pairs) ★
   r = (1−α)·chrF + α·P(HT)  r = 0.5·chrF + 0.3·COMET
   α: 0→0.3 warm-up           + 0.2·TTR − copy_penalty
   100 steps, β=0.04          200 steps, β=0.04
```

---

## Reward Functions

**GRPOv1**
```python
r = (1 - alpha) * chrF(hyp, ref) / 100 + alpha * P(HT | hyp)
# alpha annealed from 0 to 0.3 over first 50 steps
```

**GRPOv2** ★
```python
r = 0.5 * chrF(hyp, ref) + 0.3 * COMET(src, hyp, ref) \
  + 0.2 * TTR(hyp) - copy_penalty(src, hyp)
# copy_penalty = 1.0 if hyp == src else 0.0
```

---

## Repository Structure

```
.
├── scripts/
│   ├── 01_sft.py                     # SFT — QLoRA fine-tuning on 50k ES–VA pairs
│   ├── 02_ht_mt_classifier.py        # HT/MT translationese classifier (RoBERTa-ca)
│   ├── 03_grpo_v1.py                 # GRPOv1 — chrF + naturalness classifier reward
│   ├── 04_grpo_v2.py                 # GRPOv2 — composite reward ★
│   └── 05_evaluation.py              # Full evaluation + dialectal analysis
├── results/
│   ├── summary_metrics.xlsx          # Aggregated metrics for all systems
│   └── eval_results_1k.xlsx          # Per-sentence metrics (1,000 sentences)
├── requirements.txt
└── README.md
```

Run scripts in order on GPU.

---

## Models and Datasets

### HuggingFace Models

| System | Model Hub |
|---|---|
| SFT | [`guerreropaula/translategemma4b-sft-es-va2`](https://huggingface.co/guerreropaula/translategemma4b-sft-es-va) |
| GRPOv1 | [`guerreropaula/80translategemma4b-grpo-es-va`](https://huggingface.co/guerreropaula/translategemma4b-grpov1-es-va) |
| GRPOv2 ★ | [`guerreropaula/translategemma4b-grpo2-es-va`](https://huggingface.co/guerreropaula/translategemma4b-grpov2-es-va) |
| HT/MT Classifier | [`guerreropaula/ht_mt_classifier_best`](https://huggingface.co/guerreropaula/ht_mt_classifier_best) |

### Datasets

| Dataset | Usage |
|---|---|
| [`gplsi/amic_parallel`](https://huggingface.co/datasets/gplsi/amic_parallel) | SFT and GRPO training (ES–VA parallel) |
| [`gplsi/ES-VA_translation_test`](https://huggingface.co/datasets/gplsi/ES-VA_translation_test) | Evaluation test set|
| TildeMODEL, DOGC, Europarl | HT/MT classifier training (standard Catalan) |

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{guerrero-2026-enhancing,
  title     = {Enhancing {LLM} Translation Performance for {Spanish}-{Valencian}
               through Supervised Fine-tuning and Reinforcement Learning},
  author    = {Guerrero Castell{\'o}, Paula},
  booktitle = {Proceedings of the 25th Annual Conference of the
               European Association for Machine Translation (EAMT 2026)},
  year      = {2026}
}
```

---

## License

This work is licensed under a [Creative Commons Attribution-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nd/4.0/) (CC BY-ND 4.0).

© 2026 The authors. No derivative works permitted.

# Enhancing LLM Translation Performance for Spanish–Valencian through Supervised Fine-Tuning and Reinforcement Learning


[![Paper](https://img.shields.io/badge/Paper-PDF-red)](paper/paper.pdf)
[![Models](https://img.shields.io/badge/HuggingFace-Models-yellow)](https://huggingface.co/collections/guerreropaula/spanish-valencian-mt-rl)

> **EAMT 2026 Proceedings** · University of the Basque Country (UPV/EHU)  
> Paula Guerrero Castelló · `pguerrero005@ikasle.ehu.eus`

This repository contains the code, models, and evaluation scripts for our submission to EAMT 2026  on adapting a translation-specialized LLM to a low-resource dialect (Valencian) using supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO).

---

## Abstract
 
Valencian, the Western Catalan variety used in the Valencian Community of Spain, lacks a dedicated language code in most multilingual machine translation (MT) systems, and is systematically rendered closer to the standard written Eastern Catalan used in Catalonia. We address this gap by adapting **TranslateGemma-4B-IT**, a 4-billion-parameter instruction-tuned (IT) large language model (LLM) specialized for translation, via three post-training strategies using only public corpora and Quantized Low-Rank Adaptation (QLoRA): (i) supervised fine-tuning (SFT); (ii) Group Relative Policy Optimization (GRPO), a reinforcement learning (RL) technique, with chrF plus a naturalness reward (GRPOv1); and (iii) GRPO with a composite automatic-metric reward (GRPOv2). Our results suggest that reward-function alignment with the target dialect is a key determinant of RL success in low-resource dialectal MT.


---
## Results
 
| Model | chrF ↑ | BLEU ↑ | TER ↓ | BLEURT ↑ | COMET ↑ | Dialectal VA ↑ |
|---|---|---|---|---|---|---|
| Baseline | 69.02 | 39.22 | 40.30 | 0.258 | 0.906 | 3.2% |
| SFT | 83.16 | 60.16 | 22.80 | 0.524 | 0.934 | **41.0%** |
| GRPOv1 (clf) | 81.65 | 56.94 | 23.96 | 0.481 | 0.926 | 15.9% |
| **GRPOv2** | **84.68** | **62.16** | **20.63** | **0.544** | **0.936** | 36.1% |
 
Test set: [`gplsi/ES-VA_translation_test`](https://huggingface.co/datasets/gplsi/ES-VA_translation_test) (1,000 sentences).

---

 
## Training Pipeline
 
The four steps below must be run in order. Steps 3a and 3b both depend on the SFT checkpoint from step 1; step 3a additionally requires the classifier from step 2.
 
| Step | Model | Init | Data | Objective |
|---|---|---|---|---|
| 1 | Baseline | `TranslateGemma-4B-IT` | — | Zero-shot inference |
| 2 | SFT | Baseline | 50k ES–VA pairs | QLoRA supervised fine-tuning |
| 3a | GRPOv1 | SFT checkpoint | 5k ES–VA pairs | chrF + naturalness classifier |
| 3b | GRPOv2 ★ | SFT checkpoint | 10k ES–VA pairs | chrF + COMET + TTR − copy penalty |
 
---

## Quick Start
 
```bash
# Step 1 SFT (required before any GRPO step)
python main.py --mode sft
 
# Step 2 Train the HT/MT classifier (required for GRPOv1 only)
python main.py --mode classifier
 
# Step 3a GRPOv1: chrF + classifier reward (requires steps 1 & 2)
python main.py --mode grpov1
 
# Step 3b GRPOv2: composite reward ★ best (requires step 1 only)
python main.py --mode grpov2
 
# Step 4 Full evaluation (all models, all metrics + dialectal analysis)
python main.py --mode eval
```

---
 
## Repository Structure
 
```
.
├── main.py                        # Single entry point
├── evaluate.py                    # Full evaluation + dialectal analysis
├── config.py                      # All hyperparameters and paths
│
├── trainers/
│   ├── sft_trainer.py             # QLoRA SFT on 50k ES–VA pairs
│   ├── classifier_trainer.py      # HT/MT translationese classifier (RoBERTa-ca)
│   ├── grpo_v1_trainer.py         # GRPOv1 — chrF + classifier reward
│   └── grpo_v2_trainer.py         # GRPOv2 — composite reward ★
│
├── rewards/
│   ├── chrf_reward.py             # chrF scaled to [0, 1]
│   ├── classifier_reward.py       # P(HT | text) from fine-tuned RoBERTa-ca
│   └── composite_reward.py        # GRPOv2: chrF + COMET + TTR − copy penalty
│
├── utils/
│   ├── model.py                   # Model/tokenizer loading, 4-bit quant, LoRA
│   ├── data.py                    # Dataset loading, prompt templates, formatters
│   ├── metrics.py                 # chrF / BLEU / TER / BLEURT / COMET + dialectal
│   └── callbacks.py               # LossPlot, RewardPlot, BleuEvalSave, RewardEvalSave
│
├── results/
│   ├── summary_metrics.xlsx       # Aggregated metrics for all systems
│   └── eval_results_1k.xlsx       # Per-sentence metrics (1,000 sentences)
│
├── requirements.txt
└── README.md
```

---
## Installation

Install PyTorch with CUDA 12.1 support:

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

Install remaining dependencies:

```bash
pip install -r requirements.txt
```

Install BLEURT separately:

```bash
pip install git+https://github.com/google-research/bleurt.git

wget https://storage.googleapis.com/bleurt-oss/bleurt-base-128.zip
unzip bleurt-base-128.zip
```

---

## Models and Datasets

### HuggingFace Models

| System | Model Hub |
|---|---|
| SFT | [`guerreropaula/translategemma4b-sft-es-va`](https://huggingface.co/guerreropaula/translategemma4b-sft-es-va) |
| GRPOv1 | [`guerreropaula/translategemma4b-grpov1-es-va`](https://huggingface.co/guerreropaula/translategemma4b-grpov1-es-va) |
| GRPOv2 ★ | [`guerreropaula/translategemma4b-grpov2-es-va`](https://huggingface.co/guerreropaula/translategemma4b-grpov2-es-va) |
| HT/MT Classifier | [`guerreropaula/ht_mt_classifier_best`](https://huggingface.co/guerreropaula/ht_mt_classifier_best) |

### Datasets

| Dataset | Usage |
|---|---|
| [`gplsi/amic_parallel`](https://huggingface.co/datasets/gplsi/amic_parallel) | SFT and GRPO training (ES–VLCA parallel) |
| [`gplsi/ES-VA_translation_test`](https://huggingface.co/datasets/gplsi/ES-VA_translation_test) | Evaluation test set|
| [SoftCatalà Parallel Corpus](https://github.com/Softcatala/parallel-catalan-corpus) | HT/MT classifier training |

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

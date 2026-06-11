# Scripts

This directory contains all scripts required to reproduce the experiments reported in the paper, including supervised fine-tuning, reward model training, reinforcement learning, and evaluation.

## Execution order

1. `sft.py`
   Trains the SFT adapter for Spanish-Valencian with QLoRA on `gplsi/amic_parallel`.

2. `classifier.py`
   Trains the HT/MT classifier used later as part of the `GRPOv1` reward.

3. `grpov1.py`
   Continues training from the SFT model with a reward based on `chrF` plus the HT/MT classifier.

4. `grpov2.py`
   Continues training from the SFT model with a composite reward based on `chrF`, `COMET`, `TTR`, and copy penalty.

5. `evaluate.py`
   Runs full evaluation on the test set and produces aggregate metrics plus dialectal analysis.


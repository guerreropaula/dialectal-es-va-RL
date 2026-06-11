# rewards/classifier_reward.py
# Naturalness reward: P(HT | text) from a fine-tuned RoBERTa-ca classifier.
# Used in GRPOv1.
# Paula Guerrero Castelló, May 2026

from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import Config


class ClassifierReward:
    """
    Wraps the HT/MT classifier as a callable reward function.

    Usage
    -----
    reward = ClassifierReward(cfg)
    scores = reward(["La seua participació...", ...])   # List[float] in [0, 1]
    """

    def __init__(self, cfg: Config):
        self.cfg       = cfg
        self._device   = "cuda" if torch.cuda.is_available() else "cpu"
        self._tok      = AutoTokenizer.from_pretrained(cfg.clf_model_id)
        self._model    = AutoModelForSequenceClassification.from_pretrained(cfg.clf_model_id)
        self._model.eval().to(self._device)
        print(f"[ClassifierReward] loaded {cfg.clf_model_id}")
        print(f"[ClassifierReward] labels: {self._model.config.id2label}")

    @torch.no_grad()
    def __call__(self, texts: List[str], batch_size: int = 16) -> List[float]:
        """Return P(HT | text) for each text in [0, 1]."""
        rewards = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc   = self._tok(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=256,
            ).to(self._device)
            probs = F.softmax(self._model(**enc).logits, dim=-1)
            rewards.extend(probs[:, self.cfg.clf_ht_label_idx].cpu().tolist())
        return rewards

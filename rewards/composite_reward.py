# rewards/composite_reward.py
# Composite reward for GRPOv2: chrF + COMET + TTR − copy penalty.
# Paula Guerrero Castelló, May 2026

from typing import List

import torch
import torch.distributed as dist
import sacrebleu

from config import Config
from rewards.chrf_reward import chrf_reward


# ── Lexical diversity ─────────────────────────────────────────────────────────

def ttr_score(hypothesis: str) -> float:
    """Type-token ratio, down-weighted for very short outputs."""
    if not hypothesis:
        return 0.0
    tokens = hypothesis.lower().split()
    if not tokens:
        return 0.0
    ttr = len(set(tokens)) / len(tokens)
    if len(tokens) < 5:
        ttr *= len(tokens) / 5.0
    return float(ttr)


# ── Copy penalty ──────────────────────────────────────────────────────────────

def copy_penalty(source: str, hypothesis: str) -> float:
    """
    Returns a penalty in [-1, 0]:
      -1    if hypothesis == source (exact copy)
      -(proportional)  if chrF(hyp, src) > 0.7
       0    otherwise
    """
    if not source or not hypothesis:
        return 0.0
    src = source.strip().lower()
    hyp = hypothesis.strip().lower()
    if src == hyp:
        return -1.0
    sim       = sacrebleu.sentence_chrf(hyp, [src]).score / 100.0
    threshold = 0.7
    if sim > threshold:
        return -(sim - threshold) / (1.0 - threshold)
    return 0.0


# ── COMET batch (with optional DDP sync) ──────────────────────────────────────

def comet_batch(
    sources: List[str],
    hyps: List[str],
    refs: List[str],
    comet_model,
) -> List[float]:
    """
    Run COMET on a batch.
    If distributed training is active, broadcasts scores from rank-0
    so all ranks stay in sync.
    """
    if comet_model is None:
        return [0.5] * len(hyps)

    data   = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hyps, refs)]
    output = comet_model.predict(data, batch_size=8, gpus=0)
    scores = output.scores if hasattr(output, "scores") else output[0]

    if dist.is_initialized():
        t = torch.tensor(scores, dtype=torch.float32, device="cuda")
        dist.broadcast(t, src=0)
        scores = t.cpu().tolist()

    return scores


# ── GRPOv2 composite ──────────────────────────────────────────────────────────

class CompositeReward:
    """
    r = w_chrf * chrF + w_comet * COMET + w_ttr * TTR + copy_penalty

    Weights are taken from Config. COMET model must be passed in
    (loaded once in the trainer, not here) to avoid reloading.
    """

    def __init__(self, cfg: Config, comet_model=None):
        self.cfg         = cfg
        self.comet_model = comet_model

    def __call__(
        self,
        completions: List[str],
        reference:   List[str],
        source_es:   List[str],
        **kwargs,
    ) -> List[float]:
        comet_scores = comet_batch(source_es, completions, reference, self.comet_model)

        rewards = []
        for hyp, ref, src, c_s in zip(completions, reference, source_es, comet_scores):
            hyp = hyp.strip() if isinstance(hyp, str) else ""
            r = (
                self.cfg.w_chrf  * chrf_reward(hyp, ref)
              + self.cfg.w_comet * c_s
              + self.cfg.w_ttr   * ttr_score(hyp)
              + copy_penalty(src, hyp)
            )
            rewards.append(float(r))
        return rewards

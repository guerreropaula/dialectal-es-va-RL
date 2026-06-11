# rewards/chrf_reward.py
# Character-level F-score (chrF) reward: measures fidelity to the reference.
# Paula Guerrero Castelló, May 2026

from typing import List
import sacrebleu


def chrf_reward(hypothesis: str, reference: str) -> float:
    """Single-sentence chrF scaled to [0, 1]."""
    if not hypothesis or not reference:
        return 0.0
    return sacrebleu.sentence_chrf(hypothesis, [reference]).score / 100.0


def chrf_reward_batch(hypotheses: List[str], references: List[str]) -> List[float]:
    return [chrf_reward(h, r) for h, r in zip(hypotheses, references)]

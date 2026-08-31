"""Development-split calibration for the deterministic abstention threshold."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ScoredCase:
    """A development case represented only by confidence and answerability."""

    score: float
    answerable: bool


def select_threshold(dev_scores: Sequence[ScoredCase]) -> float:
    """Select the highest threshold with the best answerability F1 score."""

    if not dev_scores:
        raise ValueError("at least one development score is required")
    best_threshold = max(score.score for score in dev_scores)
    best_f1 = -1.0
    for threshold in sorted({item.score for item in dev_scores}):
        true_positive = sum(item.answerable and item.score >= threshold for item in dev_scores)
        false_positive = sum(not item.answerable and item.score >= threshold for item in dev_scores)
        false_negative = sum(item.answerable and item.score < threshold for item in dev_scores)
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = (2 * true_positive / denominator) if denominator else 0.0
        if f1 >= best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold

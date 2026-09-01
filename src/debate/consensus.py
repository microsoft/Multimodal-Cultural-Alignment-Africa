"""Consensus checking and final score aggregation."""

from __future__ import annotations

from config.settings import AGREEMENT_THRESHOLD
from src.schemas.models import CulturalRating


def check_consensus(ratings: list[CulturalRating], threshold: int | None = None) -> bool:
    """Return True if the spread of scores is within the threshold."""
    if len(ratings) < 2:
        return True
    t = threshold if threshold is not None else AGREEMENT_THRESHOLD
    scores = [r.score for r in ratings]
    return (max(scores) - min(scores)) <= t


def compute_final_score(ratings: list[CulturalRating]) -> float:
    """Return the mean score across all valid ratings."""
    if not ratings:
        return 0.0
    return round(sum(r.score for r in ratings) / len(ratings), 2)

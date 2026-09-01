"""Dependency-free metrics for score-model evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True, slots=True)
class ScoreMetrics:
    """Aggregate regression and directional metrics for one evaluation slice."""

    count: int
    mae: float
    sign_accuracy: float
    strong_count: int
    strong_mae: float | None


def _finite_scores(name: str, values: Iterable[float]) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in normalized):
        raise ValueError(f"{name} must contain only finite values")
    return normalized


def score_direction(value: float, *, neutral_threshold: float = 0.05) -> int:
    """Map a score to negative, neutral, or positive direction."""

    if not 0.0 <= neutral_threshold < 1.0:
        raise ValueError("neutral_threshold must be in [0, 1)")
    if value < -neutral_threshold:
        return -1
    if value > neutral_threshold:
        return 1
    return 0


def compute_score_metrics(
    targets: Iterable[float],
    predictions: Iterable[float],
    *,
    neutral_threshold: float = 0.05,
    strong_threshold: float = 0.5,
) -> ScoreMetrics:
    """Compute MAE, directional accuracy, and strong-impact MAE."""

    actual = _finite_scores("targets", targets)
    predicted = _finite_scores("predictions", predictions)
    if len(actual) != len(predicted):
        raise ValueError("targets and predictions must have equal length")
    if not actual:
        raise ValueError("at least one target/prediction pair is required")
    if not 0.0 <= strong_threshold <= 1.0:
        raise ValueError("strong_threshold must be in [0, 1]")

    errors = tuple(abs(left - right) for left, right in zip(actual, predicted))
    correct_directions = sum(
        score_direction(left, neutral_threshold=neutral_threshold)
        == score_direction(right, neutral_threshold=neutral_threshold)
        for left, right in zip(actual, predicted)
    )
    strong_errors = tuple(
        error
        for target, error in zip(actual, errors)
        if abs(target) >= strong_threshold
    )
    return ScoreMetrics(
        count=len(actual),
        mae=math.fsum(errors) / len(errors),
        sign_accuracy=correct_directions / len(actual),
        strong_count=len(strong_errors),
        strong_mae=(
            math.fsum(strong_errors) / len(strong_errors)
            if strong_errors
            else None
        ),
    )

"""Deterministic ratio distributions for synthetic company generation.

Every sampler requires an explicit :class:`random.Random` instance.  The module
never reads or mutates Python's process-global random state, which makes dataset
generation reproducible and safe to compose with other seeded generators.
"""

from __future__ import annotations

import math
import random


def _random_source(rng: random.Random) -> random.Random:
    if not isinstance(rng, random.Random):
        raise TypeError("rng must be a random.Random instance")
    return rng


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(name: str, value: object) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _bounds(low: object, high: object) -> tuple[float, float]:
    lower = _finite("low", low)
    upper = _finite("high", high)
    if not 0.0 <= lower <= upper <= 1.0:
        raise ValueError("ratio bounds must satisfy 0 <= low <= high <= 1")
    return lower, upper


def _within(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def beta_ratio(
    rng: random.Random,
    alpha: float,
    beta: float,
    *,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    """Sample a scaled beta-distributed ratio within ``[low, high]``.

    Beta distributions are useful for realistic structural values because they
    can favor low, central, or high ratios without the flat tails produced by a
    uniform sampler.
    """

    source = _random_source(rng)
    alpha_value = _positive("alpha", alpha)
    beta_value = _positive("beta", beta)
    lower, upper = _bounds(low, high)
    if lower == upper:
        return lower
    unit_value = source.betavariate(alpha_value, beta_value)
    return _within(lower + (upper - lower) * unit_value, lower, upper)


def truncated_ratio(
    rng: random.Random,
    mean: float,
    stddev: float,
    *,
    low: float = 0.0,
    high: float = 1.0,
    max_attempts: int = 128,
) -> float:
    """Sample a normally distributed ratio truncated to ``[low, high]``.

    Rejection sampling retains the intended interior distribution.  The
    deterministic fallback clamps the final draw only in extremely narrow-tail
    cases, ensuring the helper always terminates for valid inputs.
    """

    source = _random_source(rng)
    lower, upper = _bounds(low, high)
    mean_value = _finite("mean", mean)
    deviation = _finite("stddev", stddev)
    if deviation < 0.0:
        raise ValueError("stddev must be non-negative")
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
        raise TypeError("max_attempts must be an integer")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if not lower <= mean_value <= upper:
        raise ValueError("mean must lie within the truncation bounds")
    if lower == upper or deviation == 0.0:
        return mean_value

    last_draw = mean_value
    for _ in range(max_attempts):
        last_draw = source.normalvariate(mean_value, deviation)
        if lower <= last_draw <= upper:
            return last_draw
    return _within(last_draw, lower, upper)


def correlated_ratio(
    rng: random.Random,
    anchor: float,
    correlation: float,
    *,
    mean: float = 0.5,
    stddev: float = 0.15,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    """Sample a bounded ratio correlated with another normalized factor.

    ``correlation`` lies in ``[-1, 1]``.  Positive values pull toward the
    normalized position of ``anchor``; negative values pull toward its mirror
    within the configured bounds.  Residual variation comes from an explicit
    truncated draw, so even strongly related company attributes are not copies.
    """

    source = _random_source(rng)
    lower, upper = _bounds(low, high)
    anchor_value = _finite("anchor", anchor)
    if not lower <= anchor_value <= upper:
        raise ValueError("anchor must lie within the ratio bounds")
    strength = _finite("correlation", correlation)
    if not -1.0 <= strength <= 1.0:
        raise ValueError("correlation must be between -1 and 1")

    independent = truncated_ratio(
        source,
        mean,
        stddev,
        low=lower,
        high=upper,
    )
    if lower == upper:
        return lower
    normalized_anchor = (anchor_value - lower) / (upper - lower)
    target_unit = normalized_anchor if strength >= 0.0 else 1.0 - normalized_anchor
    target = lower + target_unit * (upper - lower)
    weight = abs(strength)
    return _within(weight * target + (1.0 - weight) * independent, lower, upper)


# Descriptive aliases for callers that prefer sampler-prefixed names.
sample_beta_ratio = beta_ratio
sample_truncated_ratio = truncated_ratio
sample_correlated_ratio = correlated_ratio


__all__ = [
    "beta_ratio",
    "correlated_ratio",
    "sample_beta_ratio",
    "sample_correlated_ratio",
    "sample_truncated_ratio",
    "truncated_ratio",
]

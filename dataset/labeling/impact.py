"""Aggregate canonical state changes into one bounded economic impact.

The aggregation is intentionally separate from mechanism formulas.  A
positive revenue, demand, margin, or competitive-position delta is beneficial;
a positive operating-cost or supply-risk delta is adverse.  The coefficients
are explicit label-contract parameters, not learned values or probabilities.

The unbounded linear contribution is::

    0.75 * revenue_expectation
  + 0.50 * demand
  - 0.75 * operating_costs
  + 1.00 * margin_expectation
  - 0.50 * supply_risk
  + 0.25 * competitive_position

The result is finally clamped to ``[-1, 1]``.  Keeping this final boundary in
the impact layer ensures every mechanism follows the same
``Event -> StateDelta -> economic_impact`` path.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from types import MappingProxyType

from dataset.schema import STATE_DELTA_FIELDS, StateDelta


IMPACT_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        "revenue_expectation": 0.75,
        "demand": 0.50,
        "operating_costs": -0.75,
        "margin_expectation": 1.00,
        "supply_risk": -0.50,
        "competitive_position": 0.25,
    }
)

if set(IMPACT_WEIGHTS) != set(STATE_DELTA_FIELDS):
    raise RuntimeError("impact weights must cover every StateDelta field exactly once")


def compute_economic_impact(state_delta: StateDelta) -> float:
    """Return the deterministic bounded impact of one complete state delta."""

    if not isinstance(state_delta, StateDelta):
        raise TypeError("state_delta must be a StateDelta")

    raw_impact = math.fsum(
        IMPACT_WEIGHTS[name] * getattr(state_delta, name)
        for name in STATE_DELTA_FIELDS
    )
    if not math.isfinite(raw_impact):
        raise ValueError("impact aggregation produced a non-finite value")
    return max(-1.0, min(1.0, raw_impact))


__all__ = ["IMPACT_WEIGHTS", "compute_economic_impact"]

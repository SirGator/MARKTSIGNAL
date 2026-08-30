"""Deterministic economic labeling: state deltas, impact, and verification."""

from .impact import IMPACT_WEIGHTS, compute_economic_impact
from .state_delta import STATE_DELTA_LABELERS, StateLabeler, compute_state_delta
from .verifier import (
    RecordVerificationError,
    VerificationIssue,
    VerificationResult,
    require_valid_record,
    verify_record,
)

__all__ = [
    "IMPACT_WEIGHTS",
    "RecordVerificationError",
    "STATE_DELTA_LABELERS",
    "StateLabeler",
    "VerificationIssue",
    "VerificationResult",
    "compute_economic_impact",
    "compute_state_delta",
    "require_valid_record",
    "verify_record",
]

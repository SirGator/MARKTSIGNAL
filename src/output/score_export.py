"""Stable, dependency-free export for external analysis systems."""

from __future__ import annotations

import json
from collections.abc import Mapping

from src.domain import ImpactAssessment


SCORE_OUTPUT_SCHEMA_VERSION = "economy-score-output-v1"


def assessment_to_record(assessment: ImpactAssessment) -> Mapping[str, object]:
    """Return the complete, audit-ready score record for an external consumer.

    ``score`` is the only numerical judgement.  Its semantic type is explicitly
    included so a technical-analysis or trading system cannot mistake it for a
    return, probability, target price or recommendation.
    """

    if not isinstance(assessment, ImpactAssessment):
        raise TypeError("assessment must be an ImpactAssessment")
    return {
        "schema_version": SCORE_OUTPUT_SCHEMA_VERSION,
        "kind": "economic_impact_score",
        "is_market_prediction": False,
        "event_id": assessment.event_id,
        "case_id": assessment.case_id,
        "horizon": assessment.horizon,
        "score": assessment.score,
        "confidence": assessment.confidence,
        "assessed_at": assessment.assessed_at.isoformat(),
        "context_cutoff": assessment.context_cutoff.isoformat(),
        "model_version": assessment.model_version,
        "feature_schema_version": assessment.feature_schema_version,
        "input_hash": assessment.input_hash,
        "evidence_ids": list(assessment.evidence_ids),
    }


def assessment_to_json(assessment: ImpactAssessment) -> str:
    """Serialize :func:`assessment_to_record` in a deterministic JSON form."""

    return json.dumps(
        assessment_to_record(assessment),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )

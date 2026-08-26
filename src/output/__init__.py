"""The explicit boundary after economy scoring.

This package exports an auditable numeric score for a caller-owned downstream
system.  It intentionally contains no market, technical-analysis or prediction
logic.
"""

from .score_export import (
    SCORE_OUTPUT_SCHEMA_VERSION,
    assessment_to_json,
    assessment_to_record,
)

__all__ = [
    "SCORE_OUTPUT_SCHEMA_VERSION",
    "assessment_to_json",
    "assessment_to_record",
]

from __future__ import annotations

from datetime import datetime, timezone
import json
import unittest

from src.domain import ImpactAssessment
from src.output import (
    SCORE_OUTPUT_SCHEMA_VERSION,
    assessment_to_json,
    assessment_to_record,
)


UTC = timezone.utc
TIME = datetime(2026, 8, 22, 14, tzinfo=UTC)


def assessment() -> ImpactAssessment:
    return ImpactAssessment(
        event_id="event:oil-price-rise",
        case_id="case:airline-x",
        horizon="30d",
        score=-0.78,
        assessed_at=TIME,
        context_cutoff=TIME,
        model_version="economy-v1",
        feature_schema_version="economy-context-v1",
        input_hash="a" * 64,
        evidence_ids=("news:oil", "filing:costs"),
        confidence=0.72,
    )


class ScoreExportTests(unittest.TestCase):
    def test_export_is_an_auditable_score_not_a_market_prediction(self) -> None:
        record = assessment_to_record(assessment())

        self.assertEqual(record["schema_version"], SCORE_OUTPUT_SCHEMA_VERSION)
        self.assertEqual(record["kind"], "economic_impact_score")
        self.assertIs(record["is_market_prediction"], False)
        self.assertEqual(record["score"], -0.78)
        self.assertNotIn("expected_return", record)
        self.assertNotIn("direction", record)

    def test_json_export_is_deterministic_and_machine_readable(self) -> None:
        encoded = assessment_to_json(assessment())

        self.assertEqual(encoded, assessment_to_json(assessment()))
        self.assertEqual(json.loads(encoded)["case_id"], "case:airline-x")


if __name__ == "__main__":
    unittest.main()

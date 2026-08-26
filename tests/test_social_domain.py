from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
import math
import unittest

from src.sources.social import EngagementSnapshot, SocialObservation


UTC = timezone.utc
BASE = datetime(2026, 1, 1, 12, tzinfo=UTC)


def observation(**overrides: object) -> SocialObservation:
    values: dict[str, object] = {
        "source": "Reddit",
        "external_id": "post-1",
        "entity_ids": ("company:Example",),
        "authored_at": BASE,
        "known_at": BASE + timedelta(minutes=1),
        "ingested_at": BASE + timedelta(minutes=2),
        "text": "Example looks strong",
        "content_hash": "content-001",
        "author_hash": "author-001",
        "stance": 0.8,
        "stance_confidence": 0.75,
        "uri": "https://example.invalid/post-1",
    }
    values.update(overrides)
    return SocialObservation(**values)  # type: ignore[arg-type]


class SocialObservationTests(unittest.TestCase):
    def test_observation_is_immutable_normalised_and_auditable(self) -> None:
        item = observation(
            source="  Reddit  ",
            entity_ids=("ticker:EX", "company:Example"),
        )

        self.assertEqual(item.source, "reddit")
        self.assertEqual(item.entity_ids, ("company:Example", "ticker:EX"))
        self.assertEqual(item.evidence_id, "reddit:post-1")
        self.assertEqual(item.provenance, "https://example.invalid/post-1")
        self.assertEqual(item.available_at, BASE + timedelta(minutes=2))
        with self.assertRaises(FrozenInstanceError):
            item.text = "mutated"  # type: ignore[misc]

    def test_timezone_is_converted_to_utc(self) -> None:
        plus_two = timezone(timedelta(hours=2))
        item = observation(
            authored_at=datetime(2026, 1, 1, 14, tzinfo=plus_two),
            known_at=datetime(2026, 1, 1, 14, 1, tzinfo=plus_two),
            ingested_at=datetime(2026, 1, 1, 14, 2, tzinfo=plus_two),
        )
        self.assertEqual(item.authored_at, BASE)
        self.assertEqual(item.ingested_at, BASE + timedelta(minutes=2))

    def test_availability_times_must_be_ordered_and_aware(self) -> None:
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            observation(authored_at=datetime(2026, 1, 1, 12))
        with self.assertRaisesRegex(ValueError, "known_at"):
            observation(known_at=BASE - timedelta(seconds=1))
        with self.assertRaisesRegex(ValueError, "ingested_at"):
            observation(ingested_at=BASE)

    def test_entities_and_stance_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one entity"):
            observation(entity_ids=())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            observation(entity_ids=("company:X", "COMPANY:x"))
        for invalid in (-1.01, 1.01, math.nan, math.inf):
            with self.subTest(stance=invalid), self.assertRaises(ValueError):
                observation(stance=invalid)
        with self.assertRaisesRegex(ValueError, "supplied together"):
            observation(stance=None)
        with self.assertRaisesRegex(ValueError, "supplied together"):
            observation(stance_confidence=None)

    def test_missing_stance_is_distinct_from_neutral_stance(self) -> None:
        missing = observation(stance=None, stance_confidence=None)
        neutral = observation(stance=0.0, stance_confidence=0.9)
        self.assertIsNone(missing.stance)
        self.assertEqual(neutral.stance, 0.0)


class EngagementSnapshotTests(unittest.TestCase):
    def test_version_and_metrics_are_validated(self) -> None:
        valid = EngagementSnapshot(
            source="Reddit",
            external_id="post-1",
            version=1,
            known_at=BASE + timedelta(minutes=3),
            ingested_at=BASE + timedelta(minutes=4),
            score=-2,
            comments=4,
        )
        self.assertEqual(valid.source, "reddit")
        self.assertEqual(valid.score, -2.0)
        self.assertEqual(valid.evidence_id, "reddit:post-1:engagement:v1")

        with self.assertRaisesRegex(ValueError, "at least 1"):
            EngagementSnapshot(
                source="reddit",
                external_id="post-1",
                version=0,
                known_at=BASE,
                ingested_at=BASE,
                likes=1,
            )
        with self.assertRaisesRegex(ValueError, "at least one metric"):
            EngagementSnapshot(
                source="reddit",
                external_id="post-1",
                version=1,
                known_at=BASE,
                ingested_at=BASE,
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            EngagementSnapshot(
                source="reddit",
                external_id="post-1",
                version=1,
                known_at=BASE,
                ingested_at=BASE,
                likes=-1,
            )


if __name__ == "__main__":
    unittest.main()

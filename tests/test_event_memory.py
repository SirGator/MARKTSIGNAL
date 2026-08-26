from __future__ import annotations

from datetime import datetime, timezone
import math
import unittest

from src.domain import (
    CanonicalEvent,
    CaseRef,
    ContextBundle,
    ContextFact,
    ImpactAssessment,
    ProvenanceRecord,
)
from src.memory import EventMemory


UTC = timezone.utc


def instant(day: int) -> datetime:
    return datetime(2026, 1, day, 12, tzinfo=UTC)


def event(
    event_id: str,
    *,
    observed_day: int,
    summary: str = "Oil price rises 25%",
    source_id: str = "news-a",
    confidence: float = 1.0,
) -> CanonicalEvent:
    source = ProvenanceRecord(
        source_id=source_id,
        source_type="news",
        observed_at=instant(observed_day),
        external_id=f"article-{source_id}",
    )
    return CanonicalEvent(
        event_id=event_id,
        event_type="commodity_price_change",
        summary=summary,
        occurred_at=instant(1),
        observed_at=instant(observed_day),
        subject_ids=("commodity:oil",),
        direction="up",
        magnitude=25,
        unit="percent",
        confidence=confidence,
        provenance=(source,),
    )


class DomainContractTests(unittest.TestCase):
    def test_impact_score_is_bounded_and_finite(self) -> None:
        valid = ImpactAssessment(
            event_id="event-1",
            case_id="case-airline",
            horizon="30d",
            score=-0.78,
            assessed_at=instant(3),
            context_cutoff=instant(2),
            model_version="economy-v1",
            feature_schema_version="features-v1",
            input_hash="a" * 64,
            evidence_ids=("fact:fuel-share",),
        )
        self.assertEqual(valid.score, -0.78)

        for invalid_score in (-1.01, 1.01, math.nan, math.inf):
            with self.subTest(score=invalid_score), self.assertRaises(ValueError):
                ImpactAssessment(
                    event_id="event-1",
                    case_id="case-airline",
                    horizon="30d",
                    score=invalid_score,
                    assessed_at=instant(3),
                    context_cutoff=instant(2),
                    model_version="economy-v1",
                    feature_schema_version="features-v1",
                    input_hash="a" * 64,
                )

    def test_naive_times_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ProvenanceRecord(
                source_id="source",
                source_type="news",
                observed_at=datetime(2026, 1, 1),
            )

    def test_event_cannot_predate_all_of_its_evidence(self) -> None:
        late_source = ProvenanceRecord(
            source_id="late-source",
            source_type="news",
            observed_at=instant(5),
        )
        with self.assertRaisesRegex(ValueError, "before all event provenance"):
            CanonicalEvent(
                event_id="backdated",
                event_type="commodity_price_change",
                summary="Oil rises",
                occurred_at=instant(1),
                observed_at=instant(2),
                subject_ids=("commodity:oil",),
                provenance=(late_source,),
            )

    def test_case_insensitive_id_collisions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "case-insensitive duplicate"):
            CanonicalEvent(
                event_id="ambiguous-ids",
                event_type="test",
                summary="Ambiguous entity ids",
                occurred_at=instant(1),
                observed_at=instant(1),
                subject_ids=("ENTITY:A", "entity:a"),
            )

    def test_context_bundle_rejects_post_cutoff_information(self) -> None:
        late_fact = ContextFact(
            fact_id="fact:fuel-share",
            subject_id="company:airline",
            predicate="operating_cost_share",
            value=31,
            unit="percent",
            observed_at=instant(4),
        )
        with self.assertRaises(ValueError):
            ContextBundle(
                event=event("event-1", observed_day=2),
                case=CaseRef(
                    case_id="case-airline",
                    entity_id="company:airline",
                ),
                cutoff=instant(3),
                retrieved_at=instant(5),
                horizon="30d",
                facts=(late_fact,),
            )

    def test_context_bundle_rejects_duplicate_and_non_historical_items(self) -> None:
        primary = event("event-primary", observed_day=2)
        case = CaseRef(case_id="case-airline", entity_id="company:airline")
        fact = ContextFact(
            fact_id="fact:duplicate",
            subject_id=case.entity_id,
            predicate="CONSUMES",
            object_id="commodity:oil",
            observed_at=instant(1),
        )

        with self.assertRaisesRegex(ValueError, "unique fact_id"):
            ContextBundle(
                event=primary,
                case=case,
                cutoff=instant(3),
                retrieved_at=instant(3),
                horizon="30d",
                facts=(fact, fact),
            )
        with self.assertRaisesRegex(ValueError, "primary event"):
            ContextBundle(
                event=primary,
                case=case,
                cutoff=instant(3),
                retrieved_at=instant(3),
                horizon="30d",
                historical_events=(primary,),
            )


class EventMemoryTests(unittest.TestCase):
    def test_exact_duplicate_merges_only_provenance(self) -> None:
        memory = EventMemory()
        first, was_created = memory.add(event("event-1", observed_day=2))
        duplicate, duplicate_was_created = memory.add(
            event("event-2", observed_day=3, source_id="news-b")
        )

        self.assertTrue(was_created)
        self.assertFalse(duplicate_was_created)
        self.assertEqual(len(memory), 1)
        self.assertEqual(duplicate.event_id, first.event_id)
        self.assertEqual(len(duplicate.provenance), 2)
        self.assertEqual(memory.get("event-2"), duplicate)

    def test_similar_event_is_not_automatically_deduplicated(self) -> None:
        memory = EventMemory()
        memory.add(event("event-1", observed_day=2))
        memory.add(
            event(
                "event-2",
                observed_day=2,
                summary="Oil price rises 24%",
            )
        )
        self.assertEqual(len(memory), 2)

    def test_search_never_returns_information_observed_after_cutoff(self) -> None:
        memory = EventMemory()
        memory.add(event("early", observed_day=2, source_id="early-source"))
        memory.add(event("late", observed_day=5, summary="Oil supply falls"))

        visible = memory.search(cutoff=instant(3))

        self.assertEqual([item.event_id for item in visible], ["early"])
        self.assertTrue(
            all(
                source.observed_at <= instant(3)
                for item in visible
                for source in item.provenance
            )
        )

    def test_cutoff_snapshot_removes_later_merged_provenance(self) -> None:
        memory = EventMemory()
        memory.add(event("event-1", observed_day=2, source_id="news-a"))
        memory.add(event("event-2", observed_day=5, source_id="news-b"))

        visible = memory.get("event-1", cutoff=instant(3))

        self.assertIsNotNone(visible)
        assert visible is not None
        self.assertEqual(
            [source.source_id for source in visible.provenance],
            ["news-a"],
        )

    def test_backfilled_duplicate_does_not_leak_later_confidence(self) -> None:
        memory = EventMemory()
        memory.add(
            event(
                "later-first",
                observed_day=5,
                source_id="later",
                confidence=0.9,
            )
        )
        memory.add(
            event(
                "earlier-backfill",
                observed_day=2,
                source_id="earlier",
                confidence=0.2,
            )
        )

        visible = memory.get("later-first", cutoff=instant(3))

        self.assertIsNotNone(visible)
        assert visible is not None
        self.assertEqual(visible.confidence, 0.2)
        self.assertEqual([item.source_id for item in visible.provenance], ["earlier"])


if __name__ == "__main__":
    unittest.main()

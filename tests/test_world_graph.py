from __future__ import annotations

from datetime import datetime, timezone
import unittest

from src.domain import ContextFact, ProvenanceRecord
from src.graph import InMemoryWorldGraph


UTC = timezone.utc


def instant(day: int) -> datetime:
    return datetime(2026, 1, day, 12, tzinfo=UTC)


class InMemoryWorldGraphTests(unittest.TestCase):
    def test_add_is_idempotent_but_fact_ids_are_immutable(self) -> None:
        graph = InMemoryWorldGraph()
        fact = ContextFact(
            fact_id="fact:fuel-share:v1",
            subject_id="company:airline",
            predicate="operating_cost_share",
            value=31,
            unit="percent",
            observed_at=instant(2),
        )

        graph.add_fact(fact)
        graph.add_fact(fact)
        self.assertEqual(len(graph), 1)

        changed_revision_with_reused_id = ContextFact(
            fact_id=fact.fact_id,
            subject_id=fact.subject_id,
            predicate=fact.predicate,
            value=35,
            unit=fact.unit,
            observed_at=instant(3),
        )
        with self.assertRaisesRegex(ValueError, "new fact_id for a revision"):
            graph.add_fact(changed_revision_with_reused_id)

        self.assertEqual(graph.get_fact(fact.fact_id), fact)

    def test_cutoff_validity_and_provenance_are_enforced(self) -> None:
        early_source = ProvenanceRecord(
            source_id="filing-early",
            source_type="company_report",
            observed_at=instant(2),
        )
        later_source = ProvenanceRecord(
            source_id="filing-later",
            source_type="company_report",
            observed_at=instant(6),
        )
        safe_fact = ContextFact(
            fact_id="fact:consumes-fuel",
            subject_id="company:airline",
            predicate="CONSUMES",
            object_id="commodity:jet_fuel",
            observed_at=instant(2),
            valid_from=instant(1),
            valid_to=instant(10),
            provenance=(early_source, later_source),
        )
        future_fact = ContextFact(
            fact_id="fact:future-hedge",
            subject_id="company:airline",
            predicate="HEDGED_BY",
            value="long-term swaps",
            observed_at=instant(5),
        )
        expired_fact = ContextFact(
            fact_id="fact:expired-contract",
            subject_id="company:airline",
            predicate="DEPENDS_ON",
            object_id="supplier:old",
            observed_at=instant(1),
            valid_to=instant(2),
        )
        graph = InMemoryWorldGraph((safe_fact, future_fact, expired_fact))

        visible = graph.neighborhood(
            ("company:airline",),
            cutoff=instant(3),
            valid_at=instant(3),
        )

        self.assertEqual([fact.fact_id for fact in visible], [safe_fact.fact_id])
        self.assertEqual(
            [source.source_id for source in visible[0].provenance],
            [early_source.source_id],
        )
        # Storage keeps the complete record; only the returned snapshot is cut.
        stored = graph.get_fact(safe_fact.fact_id)
        assert stored is not None
        self.assertEqual(len(stored.provenance), 2)

    def test_future_edge_cannot_open_a_second_hop(self) -> None:
        graph = InMemoryWorldGraph(
            (
                ContextFact(
                    fact_id="fact:future-company-edge",
                    subject_id="company:airline",
                    predicate="CONSUMES",
                    object_id="commodity:jet_fuel",
                    observed_at=instant(5),
                ),
                ContextFact(
                    fact_id="fact:fuel-oil-edge",
                    subject_id="commodity:jet_fuel",
                    predicate="DERIVED_FROM",
                    object_id="commodity:oil",
                    observed_at=instant(1),
                ),
            )
        )

        visible = graph.neighborhood(
            ("company:airline",),
            cutoff=instant(3),
            valid_at=instant(3),
            max_hops=2,
        )

        self.assertEqual(visible, ())

    def test_search_and_ties_are_deterministic(self) -> None:
        facts = tuple(
            ContextFact(
                fact_id=f"fact:{suffix}",
                subject_id="company:airline",
                predicate="CONSUMES",
                object_id="commodity:fuel",
                observed_at=instant(2),
            )
            for suffix in ("b", "a")
        )
        graph = InMemoryWorldGraph(facts)

        first = graph.search(
            cutoff=instant(3),
            predicates=("consumes",),
            limit=1,
        )

        self.assertEqual([fact.fact_id for fact in first], ["fact:a"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from datetime import datetime, timezone
import unittest

from src.domain import CanonicalEvent, CaseRef, ContextFact, ProvenanceRecord
from src.graph import InMemoryWorldGraph
from src.memory import EventMemory
from src.retrieval import ContextRetriever


UTC = timezone.utc


def instant(day: int) -> datetime:
    return datetime(2026, 1, day, 12, tzinfo=UTC)


def event(
    event_id: str,
    *,
    event_type: str = "commodity_price_change",
    summary: str = "Crude oil price rises 25 percent",
    occurred_day: int = 10,
    observed_day: int | None = None,
    subject_ids: tuple[str, ...] = ("commodity:oil",),
    direction: str | None = "up",
    provenance: tuple[ProvenanceRecord, ...] = (),
) -> CanonicalEvent:
    return CanonicalEvent(
        event_id=event_id,
        event_type=event_type,
        summary=summary,
        occurred_at=instant(occurred_day),
        observed_at=instant(observed_day or occurred_day),
        subject_ids=subject_ids,
        direction=direction,
        magnitude=25 if event_type == "commodity_price_change" else None,
        unit="percent" if event_type == "commodity_price_change" else None,
        provenance=provenance,
    )


class ContextRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.airline = CaseRef(
            case_id="case:airline",
            entity_id="company:airline",
            name="Airline X",
        )
        self.producer = CaseRef(
            case_id="case:producer",
            entity_id="company:producer",
            name="Oil Producer Y",
        )
        self.current_event = event("event:current")
        self.graph = InMemoryWorldGraph(
            (
                ContextFact(
                    fact_id="fact:airline-consumes-fuel",
                    subject_id=self.airline.entity_id,
                    predicate="CONSUMES",
                    object_id="commodity:jet_fuel",
                    observed_at=instant(2),
                ),
                ContextFact(
                    fact_id="fact:jet-fuel-from-oil",
                    subject_id="commodity:jet_fuel",
                    predicate="DERIVED_FROM",
                    object_id="commodity:oil",
                    observed_at=instant(2),
                ),
                ContextFact(
                    fact_id="fact:airline-fuel-cost-share",
                    subject_id=self.airline.entity_id,
                    predicate="OPERATING_COST_SHARE",
                    value=31,
                    unit="percent",
                    observed_at=instant(2),
                ),
                ContextFact(
                    fact_id="fact:airline-headquarters",
                    subject_id=self.airline.entity_id,
                    predicate="HEADQUARTERED_IN",
                    object_id="city:berlin",
                    observed_at=instant(2),
                ),
                ContextFact(
                    fact_id="fact:producer-produces-oil",
                    subject_id=self.producer.entity_id,
                    predicate="PRODUCES",
                    object_id="commodity:oil",
                    observed_at=instant(2),
                ),
                ContextFact(
                    fact_id="fact:future-airline-hedge",
                    subject_id=self.airline.entity_id,
                    predicate="HEDGED_BY",
                    value="long-term hedge",
                    observed_at=instant(13),
                ),
            )
        )

        self.memory = EventMemory()
        self.memory.extend(
            (
                event(
                    "event:old-oil",
                    summary="Oil price rose 18 percent",
                    occurred_day=4,
                ),
                event(
                    "event:old-airline",
                    event_type="company_cost_warning",
                    summary="Airline warned about jet fuel costs",
                    occurred_day=6,
                    subject_ids=(self.airline.entity_id,),
                    direction="down",
                ),
                event(
                    "event:unrelated",
                    event_type="cyber_incident",
                    summary="Retailer reports a cyber incident",
                    occurred_day=7,
                    subject_ids=("company:retailer",),
                    direction=None,
                ),
                event(
                    "event:learned-later",
                    summary="Oil price rose before but was reported later",
                    occurred_day=8,
                    observed_day=13,
                ),
                self.current_event,
            )
        )

    def test_same_event_gets_case_conditioned_company_facts(self) -> None:
        retriever = ContextRetriever(
            self.graph,
            self.memory,
            fact_limit=3,
            historical_event_limit=2,
        )

        airline_context = retriever.retrieve(
            self.current_event,
            self.airline,
            cutoff=instant(12),
            horizon="30d",
            retrieved_at=instant(12),
        )
        producer_context = retriever.retrieve(
            self.current_event,
            self.producer,
            cutoff=instant(12),
            horizon="30d",
            retrieved_at=instant(12),
        )

        airline_fact_ids = [fact.fact_id for fact in airline_context.facts]
        producer_fact_ids = [fact.fact_id for fact in producer_context.facts]
        self.assertIn("fact:airline-consumes-fuel", airline_fact_ids)
        self.assertIn("fact:jet-fuel-from-oil", airline_fact_ids)
        self.assertIn("fact:airline-fuel-cost-share", airline_fact_ids)
        self.assertNotIn("fact:producer-produces-oil", airline_fact_ids)
        self.assertEqual(producer_fact_ids[0], "fact:producer-produces-oil")
        self.assertNotIn("fact:airline-consumes-fuel", producer_fact_ids)
        self.assertNotIn("fact:airline-fuel-cost-share", producer_fact_ids)
        self.assertNotIn("fact:future-airline-hedge", airline_fact_ids)

    def test_two_hop_context_never_crosses_into_a_peer_company(self) -> None:
        retriever = ContextRetriever(
            self.graph,
            self.memory,
            fact_limit=8,
            historical_event_limit=0,
        )

        airline_context = retriever.retrieve(
            self.current_event,
            self.airline,
            cutoff=instant(12),
            horizon="30d",
            retrieved_at=instant(12),
        )
        producer_context = retriever.retrieve(
            self.current_event,
            self.producer,
            cutoff=instant(12),
            horizon="30d",
            retrieved_at=instant(12),
        )

        self.assertNotIn(
            "fact:producer-produces-oil",
            {fact.fact_id for fact in airline_context.facts},
        )
        self.assertNotIn(
            "fact:airline-consumes-fuel",
            {fact.fact_id for fact in producer_context.facts},
        )

    def test_unrelated_company_fact_is_not_an_event_link(self) -> None:
        retailer = CaseRef(
            case_id="case:retailer",
            entity_id="company:retailer",
        )
        graph = InMemoryWorldGraph(
            (
                ContextFact(
                    fact_id="fact:retailer-hq",
                    subject_id=retailer.entity_id,
                    predicate="HEADQUARTERED_IN",
                    object_id="city:berlin",
                    observed_at=instant(2),
                ),
            )
        )

        context = ContextRetriever(graph).retrieve(
            self.current_event,
            retailer,
            cutoff=instant(12),
            horizon="30d",
            retrieved_at=instant(12),
        )

        self.assertEqual(context.facts, ())
        self.assertEqual(context.historical_events, ())

    def test_historical_events_are_relevant_past_and_cutoff_safe(self) -> None:
        context = ContextRetriever(
            self.graph,
            self.memory,
            historical_event_limit=2,
        ).retrieve(
            self.current_event,
            self.airline,
            cutoff=instant(12),
            horizon="30d",
            retrieved_at=instant(12),
        )

        event_ids = [item.event_id for item in context.historical_events]
        self.assertEqual(event_ids, ["event:old-oil", "event:old-airline"])
        self.assertNotIn(self.current_event.event_id, event_ids)
        self.assertNotIn("event:learned-later", event_ids)
        self.assertNotIn("event:unrelated", event_ids)

    def test_event_and_fact_provenance_are_snapshotted_at_cutoff(self) -> None:
        early = ProvenanceRecord(
            source_id="early",
            source_type="news",
            observed_at=instant(10),
        )
        later = ProvenanceRecord(
            source_id="later",
            source_type="news",
            observed_at=instant(14),
        )
        merged_event = event(
            "event:with-later-source",
            provenance=(early, later),
        )
        fact_early = ProvenanceRecord(
            source_id="fact-early",
            source_type="filing",
            observed_at=instant(2),
        )
        fact_later = ProvenanceRecord(
            source_id="fact-later",
            source_type="filing",
            observed_at=instant(14),
        )
        graph = InMemoryWorldGraph(
            (
                ContextFact(
                    fact_id="fact:mixed-provenance",
                    subject_id=self.airline.entity_id,
                    predicate="CONSUMES",
                    object_id="commodity:oil",
                    observed_at=instant(2),
                    provenance=(fact_early, fact_later),
                ),
            )
        )

        context = ContextRetriever(graph).retrieve(
            merged_event,
            self.airline,
            cutoff=instant(12),
            horizon="30d",
            retrieved_at=instant(12),
        )

        self.assertEqual(
            [source.source_id for source in context.event.provenance],
            [early.source_id],
        )
        self.assertEqual(
            [source.source_id for source in context.facts[0].provenance],
            [fact_early.source_id],
        )

    def test_top_k_ties_use_fact_id(self) -> None:
        tied_graph = InMemoryWorldGraph(
            tuple(
                ContextFact(
                    fact_id=f"fact:{suffix}",
                    subject_id=self.airline.entity_id,
                    predicate="DEPENDS_ON",
                    object_id="commodity:oil",
                    observed_at=instant(2),
                )
                for suffix in ("b", "a")
            )
        )

        context = ContextRetriever(tied_graph, fact_limit=1).retrieve(
            self.current_event,
            self.airline,
            cutoff=instant(12),
            horizon="30d",
            retrieved_at=instant(12),
        )

        self.assertEqual([fact.fact_id for fact in context.facts], ["fact:a"])

    def test_too_small_budget_does_not_return_an_orphan_second_hop(self) -> None:
        bridge_graph = InMemoryWorldGraph(
            (
                ContextFact(
                    fact_id="fact:bridge",
                    subject_id=self.airline.entity_id,
                    predicate="CONSUMES",
                    object_id="system:fuel",
                    observed_at=instant(2),
                ),
                ContextFact(
                    fact_id="fact:leaf",
                    subject_id="system:fuel",
                    predicate="DERIVED_FROM",
                    object_id="commodity:oil",
                    observed_at=instant(2),
                ),
            )
        )

        context = ContextRetriever(bridge_graph, fact_limit=1).retrieve(
            self.current_event,
            self.airline,
            cutoff=instant(12),
            horizon="30d",
            retrieved_at=instant(12),
        )

        self.assertEqual(context.facts, ())

    def test_direction_or_unit_alone_does_not_make_history_relevant(self) -> None:
        memory = EventMemory()
        memory.extend(
            (
                event(
                    "event:space",
                    event_type="space_launch",
                    summary="Rocket reaches orbit",
                    occurred_day=4,
                    subject_ids=("vehicle:rocket",),
                    direction="up",
                ),
                event(
                    "event:employment",
                    event_type="employment_report",
                    summary="Employment grows",
                    occurred_day=5,
                    subject_ids=("macro:employment",),
                    direction="up",
                ),
            )
        )

        context = ContextRetriever(
            self.graph,
            memory,
            historical_event_limit=8,
        ).retrieve(
            self.current_event,
            self.airline,
            cutoff=instant(12),
            horizon="30d",
            retrieved_at=instant(12),
        )

        self.assertEqual(context.historical_events, ())


if __name__ == "__main__":
    unittest.main()

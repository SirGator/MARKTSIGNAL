"""Bridge: EconomicScenario -> ContextBundle -> ContextSerializer.

Training and production use EXACTLY the same serialization path:

    Scenario -> to_context_bundle() -> ContextSerializer.serialize() -> Tokenizer -> Model

This guarantees the model sees the same input format in training and inference.
"""

from __future__ import annotations

from datetime import datetime, timezone

from src.domain import (
    CanonicalEvent,
    CaseRef,
    ContextBundle,
    ContextFact,
    ProvenanceRecord,
)
from src.models.context_serializer import (
    ContextSerializer,
    SUMMARY_MODE_FULL,
    normalize_summary_mode,
)

from .scenarios import EconomicScenario


def scenario_to_context_bundle(
    scenario: EconomicScenario,
    *,
    base_time: datetime | None = None,
) -> ContextBundle:
    """Convert a synthetic EconomicScenario into a real ContextBundle.

    The ContextBundle uses the same domain types as production:
    CanonicalEvent, CaseRef, ContextFact with predicates, values, units.
    """
    if base_time is None:
        base_time = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)

    occurred_at = base_time
    observed_at = base_time
    cutoff = base_time
    retrieved_at = base_time

    provenance = ProvenanceRecord(
        source_id="synthetic-training",
        source_type="training",
        observed_at=observed_at,
    )

    event = CanonicalEvent(
        event_id=f"event:{scenario.event_type}:{id(scenario)}",
        event_type=scenario.event_type,
        summary=scenario.event_text,
        occurred_at=occurred_at,
        observed_at=observed_at,
        subject_ids=(scenario.subject,) if scenario.subject else ("entity:unknown",),
        direction=scenario.direction if scenario.direction != "loss" else None,
        magnitude=scenario.magnitude if scenario.magnitude > 0 else None,
        unit=scenario.magnitude_unit if scenario.magnitude > 0 and scenario.magnitude_unit != "none" else None,
        confidence=0.9,
        provenance=(provenance,),
    )

    case = CaseRef(
        case_id=f"case:{scenario.case_role}",
        entity_id=f"company:{scenario.case_role}",
        name=f"Company {scenario.case_role}",
        case_type=scenario.case_type,
    )

    facts = _build_facts(scenario, case, observed_at)

    return ContextBundle(
        event=event,
        case=case,
        cutoff=cutoff,
        retrieved_at=retrieved_at,
        horizon=f"{scenario.horizon_days}d",
        facts=facts,
    )


def _build_facts(
    scenario: EconomicScenario,
    case: CaseRef,
    observed_at: datetime,
) -> tuple[ContextFact, ...]:
    """Build ContextFacts from the scenario's context variables.

    Each label-influencing variable becomes a fact with predicate, value, unit.
    This mirrors how a real World Graph would store the same information.
    """
    facts: list[ContextFact] = []
    fact_idx = 0

    def add_fact(predicate: str, value, unit: str | None = None) -> None:
        nonlocal fact_idx
        facts.append(
            ContextFact(
                fact_id=f"fact:{fact_idx}",
                subject_id=case.entity_id,
                predicate=predicate,
                value=value,
                unit=unit,
                observed_at=observed_at,
                confidence=0.9,
            )
        )
        fact_idx += 1

    add_fact("role", scenario.case_role)

    commodity_key = scenario.event_type
    if scenario.event_type == "input_price_change":
        add_fact(f"{commodity_key}_exposure", scenario.exposure, "percent")
        if scenario.case_role == "consumer":
            add_fact("hedging_ratio", scenario.hedging * 100, "percent")
            add_fact("pricing_power", scenario.pricing_power * 100, "percent")
            add_fact("substitution_ability", scenario.substitution * 100, "percent")
        elif scenario.case_role == "producer":
            add_fact("hedging_ratio", scenario.hedging * 100, "percent")
    elif scenario.event_type == "demand_change":
        add_fact(f"{commodity_key}_exposure", scenario.exposure, "percent")
        if scenario.case_role == "producer":
            add_fact("pricing_power", scenario.pricing_power * 100, "percent")
    elif scenario.event_type == "interest_rate_change":
        if scenario.case_role == "bank":
            add_fact("rate_sensitivity", scenario.pricing_power * 100, "percent")
        elif scenario.case_role == "real_estate":
            add_fact("debt_ratio", scenario.exposure / 100)
            add_fact("rate_sensitivity", scenario.pricing_power * 100, "percent")
    elif scenario.event_type == "customer_loss":
        add_fact("customer_exposure", scenario.exposure, "percent")
        if scenario.case_role in ("supplier_high", "supplier_low"):
            add_fact("replacement_difficulty", scenario.substitution * 100, "percent")
    elif scenario.event_type == "export_restriction":
        add_fact(f"{commodity_key}_exposure", scenario.exposure, "percent")
        if scenario.case_role == "consumer":
            add_fact("substitution_ability", scenario.substitution * 100, "percent")
        elif scenario.case_role == "competitor":
            add_fact("market_share_gain_potential", scenario.pricing_power * 100, "percent")
            add_fact("substitution_barrier", scenario.substitution * 100, "percent")
    elif scenario.event_type == "regulatory_fine":
        add_fact("fine_magnitude", scenario.magnitude, "percent")
        add_fact("exposure", scenario.exposure, "percent")
        add_fact("pricing_power", scenario.pricing_power * 100, "percent")
    elif scenario.event_type == "supply_disruption":
        add_fact("disruption_magnitude", scenario.magnitude, "percent")
        add_fact("exposure", scenario.exposure, "percent")
        add_fact("substitution_ability", scenario.substitution * 100, "percent")

    return tuple(facts)


def serialize_scenario(
    scenario: EconomicScenario,
    *,
    summary_mode: str = SUMMARY_MODE_FULL,
) -> str:
    """Serialize a scenario through the PRODUCTION path.

    This is the single entry point used by both training and evaluation.
    It builds a ContextBundle and serializes it with ContextSerializer.serialize()
    — the exact same code path as production inference.
    """
    mode = normalize_summary_mode(summary_mode)
    bundle = scenario_to_context_bundle(scenario)
    # The production serializer owns neutralization, including historical
    # event summaries.  It never mutates the frozen ContextBundle/scenario.
    serializer = ContextSerializer(summary_mode=mode)
    return serializer.serialize(bundle, horizon=f"{scenario.horizon_days}d")

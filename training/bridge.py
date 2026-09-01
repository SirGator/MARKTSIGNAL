"""Bridge: EconomicScenario / DatasetRecord -> ContextBundle -> ContextSerializer.

Training and production use EXACTLY the same serialization path:

    Scenario/Record -> to_context_bundle() -> ContextSerializer.serialize() -> Tokenizer -> Model

This guarantees the model sees the same input format in training and inference.
Two source types feed the same ContextBundle contract:

    - training.scenarios.EconomicScenario  (legacy parametric generator)
    - dataset.schema.DatasetRecord          (frozen, versioned dataset)
"""

from __future__ import annotations

from datetime import datetime, timezone

from .domain import (
    CanonicalEvent,
    CaseRef,
    ContextBundle,
    ContextFact,
    ProvenanceRecord,
)
from .modeling.context_serializer import (
    ContextSerializer,
    SUMMARY_MODE_FULL,
    normalize_summary_mode,
)

from .scenarios import EconomicScenario

# DatasetRecord is imported lazily inside the functions that need it to keep
# the training dependency graph shallow and avoid importing dataset.* unless a
# frozen dataset is actually used.


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


# ---------------------------------------------------------------------------
# DatasetRecord bridge — frozen dataset records feed the same ContextBundle.
# ---------------------------------------------------------------------------

def _subject_token_for_record(record) -> str:
    """Return the subject string used in the serialized event line."""

    return f"{record.event.subject_class}:{record.event.subject}"


def record_to_context_bundle(
    record,
    *,
    base_time: datetime | None = None,
):
    """Convert a frozen :class:`dataset.schema.DatasetRecord` into a ContextBundle.

    The ContextBundle uses the same domain types as production
    (CanonicalEvent, CaseRef, ContextFact) and therefore flows through the
    exact same :class:`ContextSerializer` path as scenario-based training and
    production inference.  Only the structural context factors that the record
    actually carries are emitted as facts; absent (unknown) factors are omitted
    just as in production retrieval.
    """

    from dataset.schema import DatasetRecord  # local import; see module note

    if not isinstance(record, DatasetRecord):
        raise TypeError("record must be a DatasetRecord")
    from .input_contract import validate_dataset_projection

    validate_dataset_projection(record)

    if base_time is None:
        base_time = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)

    occurred_at = base_time
    observed_at = base_time
    cutoff = base_time
    retrieved_at = base_time

    provenance = ProvenanceRecord(
        source_id=record.metadata.source,
        source_type="dataset",
        observed_at=observed_at,
    )

    summary = record.event.summary or _default_record_summary(record)

    event = CanonicalEvent(
        event_id=f"event:{record.id}",
        event_type=record.event.mechanism,
        summary=summary,
        occurred_at=occurred_at,
        observed_at=observed_at,
        subject_ids=(_subject_token_for_record(record),),
        direction=record.event.direction,
        magnitude=record.event.magnitude,
        unit=record.event.unit or "fraction",
        confidence=record.confidence,
        provenance=(provenance,),
    )

    case = CaseRef(
        case_id=f"case:{record.target.entity_id}",
        entity_id=record.target.entity_id,
        name=record.target.entity_id,
        case_type="company",
    )

    facts = _build_record_facts(record, case, observed_at)

    return ContextBundle(
        event=event,
        case=case,
        cutoff=cutoff,
        retrieved_at=retrieved_at,
        horizon=f"{record.horizon_days}d",
        facts=facts,
    )


def _default_record_summary(record) -> str:
    """Synthesize a terse deterministic summary when the record has none.

    Structured-only datasets legitimately omit ``event.summary``.  The
    serializer still needs a non-empty string, so we render the mechanism,
    direction, subject, and magnitude in a stable template.  When the run
    uses ``summary_mode='none'`` this summary is replaced by ``[NO_SUMMARY]``
    downstream and never reaches the model.
    """

    direction = record.event.direction
    mechanism = record.event.mechanism
    subject = record.event.subject or record.event.subject_class
    magnitude = record.event.magnitude
    return f"{subject} {mechanism} {direction} {magnitude:g}"


def _build_record_facts(record, case, observed_at):
    """Emit one ContextFact per supplied structural context factor.

    Every supplied factor is a normalized ratio in [0, 1]; we surface it as a
    ratio-valued fact so the model sees the same ``predicate=value;unit``
    shape that production retrieval produces.
    """

    from dataset.schema import CONTEXT_FACTOR_FIELDS

    facts: list[ContextFact] = []
    for index, name in enumerate(CONTEXT_FACTOR_FIELDS):
        value = getattr(record.context, name)
        if value is None:
            continue
        facts.append(
            ContextFact(
                fact_id=f"fact:{index}",
                subject_id=case.entity_id,
                predicate=name,
                value=float(value),
                unit="ratio",
                observed_at=observed_at,
                confidence=record.confidence,
            )
        )
    return tuple(facts)


def serialize_record(
    record,
    *,
    summary_mode: str = SUMMARY_MODE_FULL,
) -> str:
    """Serialize a frozen dataset record through the PRODUCTION path.

    Mirror of :func:`serialize_scenario` for :class:`DatasetRecord` values.
    """

    mode = normalize_summary_mode(summary_mode)
    bundle = record_to_context_bundle(record)
    serializer = ContextSerializer(summary_mode=mode)
    return serializer.serialize(bundle, horizon=f"{record.horizon_days}d")

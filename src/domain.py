"""Shared, storage-agnostic contracts for the economic-impact pipeline.

The domain objects deliberately distinguish when something happened from when
the system learned about it.  Retrieval and backtests must use ``observed_at``
for their information cutoff; ``occurred_at`` alone is not sufficient to avoid
look-ahead leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import math
import re
from typing import TypeAlias


ScalarValue: TypeAlias = str | int | float | bool


def _required_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_text(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    return _required_text(name, value)


def _utc_datetime(name: str, value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _optional_utc_datetime(
    name: str,
    value: datetime | None,
) -> datetime | None:
    if value is None:
        return None
    return _utc_datetime(name, value)


def _confidence(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return result


def _finite_number(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _normalised_ids(name: str, values: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{name} must be a tuple of strings, not a string")
    cleaned = tuple(_required_text(name, value) for value in values)
    by_casefold: dict[str, str] = {}
    for value in cleaned:
        key = value.casefold()
        previous = by_casefold.get(key)
        if previous is not None and previous != value:
            raise ValueError(
                f"{name} contains case-insensitive duplicate ids "
                f"{previous!r} and {value!r}"
            )
        by_casefold[key] = value
    return tuple(sorted(by_casefold.values(), key=lambda value: (value.casefold(), value)))


def _normalised_text(value: str | None) -> str | None:
    if value is None:
        return None
    return re.sub(r"\s+", " ", value).strip().casefold()


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """A single piece of source evidence and its availability time."""

    source_id: str
    source_type: str
    observed_at: datetime
    published_at: datetime | None = None
    uri: str | None = None
    external_id: str | None = None
    content_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _required_text("source_id", self.source_id))
        object.__setattr__(
            self,
            "source_type",
            _required_text("source_type", self.source_type),
        )
        object.__setattr__(
            self,
            "observed_at",
            _utc_datetime("observed_at", self.observed_at),
        )
        object.__setattr__(
            self,
            "published_at",
            _optional_utc_datetime("published_at", self.published_at),
        )
        if self.published_at is not None and self.observed_at < self.published_at:
            raise ValueError("observed_at must not be before published_at")
        object.__setattr__(self, "uri", _optional_text("uri", self.uri))
        object.__setattr__(
            self,
            "external_id",
            _optional_text("external_id", self.external_id),
        )
        object.__setattr__(
            self,
            "content_hash",
            _optional_text("content_hash", self.content_hash),
        )

    @property
    def evidence_id(self) -> str:
        """Return a stable, human-inspectable identifier for this evidence."""

        if self.external_id is not None:
            return f"{self.source_id}:{self.external_id}"
        if self.content_hash is not None:
            return f"{self.source_id}:content-hash:{self.content_hash.casefold()}"
        if self.uri is not None:
            return f"{self.source_id}:{self.uri}"
        return f"{self.source_id}:{self.observed_at.isoformat()}"


def _normalised_provenance(
    values: tuple[ProvenanceRecord, ...],
) -> tuple[ProvenanceRecord, ...]:
    if isinstance(values, ProvenanceRecord):
        raise TypeError("provenance must be a tuple of ProvenanceRecord values")
    for value in values:
        if not isinstance(value, ProvenanceRecord):
            raise TypeError("provenance must contain only ProvenanceRecord values")
    unique = {value: None for value in values}
    return tuple(
        sorted(
            unique,
            key=lambda item: (
                item.observed_at,
                item.source_id.casefold(),
                item.evidence_id.casefold(),
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class CanonicalEvent:
    """A source-independent economic event produced by event extraction."""

    event_id: str
    event_type: str
    summary: str
    occurred_at: datetime
    observed_at: datetime
    subject_ids: tuple[str, ...]
    object_ids: tuple[str, ...] = ()
    direction: str | None = None
    magnitude: float | None = None
    unit: str | None = None
    confidence: float = 1.0
    provenance: tuple[ProvenanceRecord, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _required_text("event_id", self.event_id))
        object.__setattr__(
            self,
            "event_type",
            _required_text("event_type", self.event_type),
        )
        object.__setattr__(self, "summary", _required_text("summary", self.summary))
        object.__setattr__(
            self,
            "occurred_at",
            _utc_datetime("occurred_at", self.occurred_at),
        )
        observed_at = _utc_datetime("observed_at", self.observed_at)
        object.__setattr__(
            self,
            "subject_ids",
            _normalised_ids("subject_ids", self.subject_ids),
        )
        if not self.subject_ids:
            raise ValueError("subject_ids must contain at least one entity")
        object.__setattr__(
            self,
            "object_ids",
            _normalised_ids("object_ids", self.object_ids),
        )
        object.__setattr__(
            self,
            "direction",
            _optional_text("direction", self.direction),
        )
        object.__setattr__(
            self,
            "magnitude",
            _finite_number("magnitude", self.magnitude),
        )
        object.__setattr__(self, "unit", _optional_text("unit", self.unit))
        confidence = _confidence("confidence", self.confidence)
        if confidence is None:
            raise TypeError("confidence must be a number")
        object.__setattr__(self, "confidence", confidence)
        provenance = _normalised_provenance(self.provenance)
        object.__setattr__(self, "provenance", provenance)
        if provenance and observed_at < min(item.observed_at for item in provenance):
            raise ValueError("observed_at must not be before all event provenance")
        object.__setattr__(self, "observed_at", observed_at)

    @property
    def deduplication_key(self) -> tuple[object, ...]:
        """Strict semantic key used for automatic deduplication.

        The full timestamp, magnitude and normalized summary are intentionally
        retained.  Near matches are not automatically merged because a false
        merge is more damaging than keeping two candidates in V1.
        """

        magnitude = None if self.magnitude is None else self.magnitude.hex()
        return (
            _normalised_text(self.event_type),
            tuple(value.casefold() for value in self.subject_ids),
            tuple(value.casefold() for value in self.object_ids),
            _normalised_text(self.direction),
            magnitude,
            _normalised_text(self.unit),
            self.occurred_at,
            _normalised_text(self.summary),
        )

    def is_available_at(self, cutoff: datetime) -> bool:
        return self.observed_at <= _utc_datetime("cutoff", cutoff)

    def as_observed_at(self, cutoff: datetime) -> CanonicalEvent | None:
        """Return a cutoff-safe snapshot with later provenance removed."""

        safe_cutoff = _utc_datetime("cutoff", cutoff)
        if self.observed_at > safe_cutoff:
            return None
        visible_provenance = tuple(
            item for item in self.provenance if item.observed_at <= safe_cutoff
        )
        return replace(self, provenance=visible_provenance)


@dataclass(frozen=True, slots=True)
class CaseRef:
    """Identity of the company or other entity being assessed."""

    case_id: str
    entity_id: str
    name: str | None = None
    case_type: str = "company"

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _required_text("case_id", self.case_id))
        object.__setattr__(self, "entity_id", _required_text("entity_id", self.entity_id))
        object.__setattr__(self, "name", _optional_text("name", self.name))
        object.__setattr__(
            self,
            "case_type",
            _required_text("case_type", self.case_type),
        )


@dataclass(frozen=True, slots=True)
class ContextFact:
    """A provenance- and validity-aware fact retrieved for a case."""

    fact_id: str
    subject_id: str
    predicate: str
    observed_at: datetime
    object_id: str | None = None
    value: ScalarValue | None = None
    unit: str | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    confidence: float = 1.0
    provenance: tuple[ProvenanceRecord, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fact_id", _required_text("fact_id", self.fact_id))
        object.__setattr__(
            self,
            "subject_id",
            _required_text("subject_id", self.subject_id),
        )
        object.__setattr__(
            self,
            "predicate",
            _required_text("predicate", self.predicate),
        )
        observed_at = _utc_datetime("observed_at", self.observed_at)
        object.__setattr__(
            self,
            "object_id",
            _optional_text("object_id", self.object_id),
        )
        if self.object_id is None and self.value is None:
            raise ValueError("ContextFact requires object_id or value")
        if self.value is not None and not isinstance(self.value, (str, int, float, bool)):
            raise TypeError("value must be a string, integer, float, boolean, or None")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("value must be finite")
        object.__setattr__(self, "unit", _optional_text("unit", self.unit))
        valid_from = _optional_utc_datetime("valid_from", self.valid_from)
        valid_to = _optional_utc_datetime("valid_to", self.valid_to)
        if valid_from is not None and valid_to is not None and valid_from > valid_to:
            raise ValueError("valid_from must be before or equal to valid_to")
        object.__setattr__(self, "valid_from", valid_from)
        object.__setattr__(self, "valid_to", valid_to)
        confidence = _confidence("confidence", self.confidence)
        if confidence is None:
            raise TypeError("confidence must be a number")
        object.__setattr__(self, "confidence", confidence)
        provenance = _normalised_provenance(self.provenance)
        object.__setattr__(self, "provenance", provenance)
        if provenance and observed_at < min(item.observed_at for item in provenance):
            raise ValueError("observed_at must not be before all fact provenance")
        object.__setattr__(self, "observed_at", observed_at)

    def is_available_at(self, cutoff: datetime) -> bool:
        return self.observed_at <= _utc_datetime("cutoff", cutoff)

    def as_observed_at(self, cutoff: datetime) -> ContextFact | None:
        """Return a cutoff-safe snapshot with later evidence removed."""

        safe_cutoff = _utc_datetime("cutoff", cutoff)
        if self.observed_at > safe_cutoff:
            return None
        visible_provenance = tuple(
            item for item in self.provenance if item.observed_at <= safe_cutoff
        )
        return replace(self, provenance=visible_provenance)

    def is_valid_at(self, at: datetime) -> bool:
        instant = _utc_datetime("at", at)
        if self.valid_from is not None and instant < self.valid_from:
            return False
        if self.valid_to is not None and instant > self.valid_to:
            return False
        return True


@dataclass(frozen=True, slots=True)
class ContextBundle:
    """Cutoff-safe input context for one event/case assessment."""

    event: CanonicalEvent
    case: CaseRef
    cutoff: datetime
    retrieved_at: datetime
    horizon: str
    facts: tuple[ContextFact, ...] = field(default_factory=tuple)
    historical_events: tuple[CanonicalEvent, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.event, CanonicalEvent):
            raise TypeError("event must be a CanonicalEvent")
        if not isinstance(self.case, CaseRef):
            raise TypeError("case must be a CaseRef")
        cutoff = _utc_datetime("cutoff", self.cutoff)
        object.__setattr__(self, "cutoff", cutoff)
        object.__setattr__(
            self,
            "retrieved_at",
            _utc_datetime("retrieved_at", self.retrieved_at),
        )
        if self.retrieved_at < cutoff:
            raise ValueError("retrieved_at must not be before cutoff")
        object.__setattr__(self, "horizon", _required_text("horizon", self.horizon))
        facts = tuple(self.facts)
        historical_events = tuple(self.historical_events)
        if any(not isinstance(fact, ContextFact) for fact in facts):
            raise TypeError("facts must contain only ContextFact values")
        if any(not isinstance(event, CanonicalEvent) for event in historical_events):
            raise TypeError(
                "historical_events must contain only CanonicalEvent values"
            )
        object.__setattr__(self, "facts", facts)
        object.__setattr__(self, "historical_events", historical_events)

        fact_ids = [fact.fact_id for fact in facts]
        if len(set(fact_ids)) != len(fact_ids):
            raise ValueError("context facts must have unique fact_id values")
        historical_event_ids = [event.event_id for event in historical_events]
        if len(set(historical_event_ids)) != len(historical_event_ids):
            raise ValueError("historical events must have unique event_id values")
        if self.event.event_id in historical_event_ids:
            raise ValueError("the primary event cannot also be historical context")
        if any(
            event.occurred_at >= self.event.occurred_at
            for event in historical_events
        ):
            raise ValueError("historical events must predate the primary event")

        events = (self.event, *historical_events)
        if any(event.observed_at > cutoff for event in events):
            raise ValueError("context contains an event observed after cutoff")
        if any(fact.observed_at > cutoff for fact in facts):
            raise ValueError("context contains a fact observed after cutoff")
        valid_at = min(self.event.occurred_at, cutoff)
        if any(not fact.is_valid_at(valid_at) for fact in facts):
            raise ValueError("context contains a fact that is not valid at event time")
        if any(
            source.observed_at > cutoff
            for event in events
            for source in event.provenance
        ):
            raise ValueError("context contains event provenance observed after cutoff")
        if any(
            source.observed_at > cutoff
            for fact in facts
            for source in fact.provenance
        ):
            raise ValueError("context contains fact provenance observed after cutoff")

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        ids = [fact.fact_id for fact in self.facts]
        ids.extend(event.event_id for event in self.historical_events)
        ids.extend(
            source.evidence_id
            for item in (self.event, *self.historical_events)
            for source in item.provenance
        )
        ids.extend(
            source.evidence_id
            for fact in self.facts
            for source in fact.provenance
        )
        return tuple(dict.fromkeys(ids))


@dataclass(frozen=True, slots=True)
class ImpactAssessment:
    """Public, case- and horizon-specific output of the economy model."""

    event_id: str
    case_id: str
    horizon: str
    score: float
    assessed_at: datetime
    context_cutoff: datetime
    model_version: str
    feature_schema_version: str
    input_hash: str
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    confidence: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _required_text("event_id", self.event_id))
        object.__setattr__(self, "case_id", _required_text("case_id", self.case_id))
        object.__setattr__(self, "horizon", _required_text("horizon", self.horizon))
        score = _finite_number("score", self.score)
        if score is None:
            raise TypeError("score must be a number")
        if not -1.0 <= score <= 1.0:
            raise ValueError("score must be between -1 and 1")
        object.__setattr__(self, "score", score)
        assessed_at = _utc_datetime("assessed_at", self.assessed_at)
        context_cutoff = _utc_datetime("context_cutoff", self.context_cutoff)
        if assessed_at < context_cutoff:
            raise ValueError("assessed_at must not be before context_cutoff")
        object.__setattr__(self, "assessed_at", assessed_at)
        object.__setattr__(self, "context_cutoff", context_cutoff)
        object.__setattr__(
            self,
            "model_version",
            _required_text("model_version", self.model_version),
        )
        object.__setattr__(
            self,
            "feature_schema_version",
            _required_text("feature_schema_version", self.feature_schema_version),
        )
        input_hash = _required_text("input_hash", self.input_hash).casefold()
        if re.fullmatch(r"[0-9a-f]{64}", input_hash) is None:
            raise ValueError("input_hash must be a SHA-256 hex digest")
        object.__setattr__(self, "input_hash", input_hash)
        object.__setattr__(
            self,
            "evidence_ids",
            _normalised_ids("evidence_ids", self.evidence_ids),
        )
        object.__setattr__(
            self,
            "confidence",
            _confidence("confidence", self.confidence),
        )

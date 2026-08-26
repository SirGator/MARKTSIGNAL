"""V1 orchestration from canonical event to one economic impact score."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Callable, Protocol

from src.domain import CanonicalEvent, CaseRef, ContextBundle, ImpactAssessment
from src.economy import EconomyModel, EconomyScore
from src.extraction import EventExtractor, SourceDocument


class EventMemoryPort(Protocol):
    def add(self, event: CanonicalEvent) -> tuple[CanonicalEvent, bool]: ...

    def get(
        self,
        event_id: str,
        *,
        cutoff: datetime | None = None,
    ) -> CanonicalEvent | None: ...


class ContextRetrieverPort(Protocol):
    def retrieve(
        self,
        event: CanonicalEvent,
        case: CaseRef,
        *,
        cutoff: datetime,
        horizon: str,
        retrieved_at: datetime | None = None,
        fact_limit: int | None = None,
        historical_event_limit: int | None = None,
    ) -> ContextBundle: ...


class EventUnavailableError(LookupError):
    """The requested event did not exist at the assessment cutoff."""


class InsufficientContextError(ValueError):
    """The pipeline cannot produce an auditable, case-specific assessment."""


@dataclass(frozen=True, slots=True)
class IngestionResult:
    source_event_id: str
    canonical_event: CanonicalEvent
    was_created: bool


class EconomyPipeline:
    """Coordinate extraction, memory, retrieval and economic interpretation.

    The pipeline intentionally ends at :class:`ImpactAssessment`; there is no
    World State update in V1.
    """

    def __init__(
        self,
        *,
        event_memory: EventMemoryPort,
        context_retriever: ContextRetrieverPort,
        economy_model: EconomyModel,
        event_extractor: EventExtractor | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._event_memory = event_memory
        self._context_retriever = context_retriever
        self._economy_model = economy_model
        self._event_extractor = event_extractor
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def ingest_document(self, document: SourceDocument) -> tuple[IngestionResult, ...]:
        """Extract and conservatively deduplicate all events in one document."""

        if self._event_extractor is None:
            raise RuntimeError("no event_extractor is configured")
        extracted = self._event_extractor.extract(document)
        return self.ingest_events(
            self._attach_provenance(event, document) for event in extracted
        )

    def ingest_events(
        self,
        events: Iterable[CanonicalEvent],
    ) -> tuple[IngestionResult, ...]:
        """Store already canonicalized events through the deduplication boundary."""

        results: list[IngestionResult] = []
        try:
            iterator = iter(events)
        except TypeError as error:
            raise TypeError("events must be an iterable of CanonicalEvent values") from error

        validated_events = tuple(iterator)
        if any(not isinstance(event, CanonicalEvent) for event in validated_events):
            raise TypeError("events must contain only CanonicalEvent values")

        for event in validated_events:
            canonical, was_created = self._event_memory.add(event)
            results.append(
                IngestionResult(
                    source_event_id=event.event_id,
                    canonical_event=canonical,
                    was_created=was_created,
                )
            )
        return tuple(results)

    def assess(
        self,
        event_id: str,
        case: CaseRef,
        *,
        cutoff: datetime,
        horizon: str,
    ) -> ImpactAssessment:
        """Return the only public V1 result: one score in ``[-1, +1]``."""

        if not isinstance(case, CaseRef):
            raise TypeError("case must be a CaseRef")
        if not isinstance(horizon, str) or not horizon.strip():
            raise ValueError("horizon must be a non-empty string")
        normalized_horizon = horizon.strip()
        if not isinstance(cutoff, datetime):
            raise TypeError("cutoff must be a datetime")
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            raise ValueError("cutoff must be timezone-aware")
        requested_cutoff = cutoff.astimezone(timezone.utc)

        operation_time = self._clock()
        if not isinstance(operation_time, datetime):
            raise TypeError("clock must return a datetime")
        if operation_time.tzinfo is None or operation_time.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        operation_time = operation_time.astimezone(timezone.utc)

        event = self._event_memory.get(event_id, cutoff=requested_cutoff)
        if event is None:
            raise EventUnavailableError(
                f"event {event_id!r} was not available at the requested cutoff"
            )
        if operation_time < event.observed_at or operation_time < requested_cutoff:
            raise ValueError("cannot assess with a cutoff later than the current clock")

        context = self._context_retriever.retrieve(
            event,
            case,
            cutoff=requested_cutoff,
            horizon=normalized_horizon,
            retrieved_at=operation_time,
        )
        self._validate_retrieval_binding(
            context,
            event=event,
            case=case,
            cutoff=requested_cutoff,
            horizon=normalized_horizon,
            operation_time=operation_time,
        )
        self._require_case_specific_context(context)

        model_score = self._economy_model.predict(context, horizon=normalized_horizon)
        if not isinstance(model_score, EconomyScore):
            raise TypeError("economy_model.predict must return EconomyScore")
        unknown_evidence = set(model_score.evidence_ids) - set(context.evidence_ids)
        if unknown_evidence:
            raise ValueError(
                "economy model reported evidence outside the retrieved context: "
                f"{sorted(unknown_evidence)}"
            )

        return ImpactAssessment(
            event_id=context.event.event_id,
            case_id=context.case.case_id,
            horizon=normalized_horizon,
            score=model_score.score,
            assessed_at=operation_time,
            context_cutoff=context.cutoff,
            model_version=model_score.model_version,
            feature_schema_version=model_score.feature_schema_version,
            input_hash=model_score.input_hash,
            evidence_ids=model_score.evidence_ids,
            confidence=model_score.confidence,
        )

    @staticmethod
    def _attach_provenance(
        event: CanonicalEvent,
        document: SourceDocument,
    ) -> CanonicalEvent:
        if not isinstance(event, CanonicalEvent):
            raise TypeError("event extractor must return CanonicalEvent values")
        if event.observed_at < document.provenance.observed_at:
            raise ValueError(
                "an extracted event cannot predate the source document observation"
            )
        if document.provenance in event.provenance:
            return event
        return replace(event, provenance=(*event.provenance, document.provenance))

    @staticmethod
    def _require_case_specific_context(context: ContextBundle) -> None:
        case_entity = context.case.entity_id
        event_entities = {*context.event.subject_ids, *context.event.object_ids}
        direct_link = case_entity in event_entities
        graph_link = EconomyPipeline._has_entity_path(
            start=case_entity,
            targets=event_entities,
            context=context,
        )
        if not (direct_link or graph_link):
            raise InsufficientContextError(
                "no point-in-time relation path links the event to the requested case"
            )
        if not context.event.provenance:
            raise InsufficientContextError("the primary event has no source provenance")
        source_less_facts = [fact.fact_id for fact in context.facts if not fact.provenance]
        source_less_history = [
            event.event_id
            for event in context.historical_events
            if not event.provenance
        ]
        if source_less_facts or source_less_history:
            raise InsufficientContextError(
                "retrieved context lacks source provenance: "
                f"facts={source_less_facts}, history={source_less_history}"
            )

    @staticmethod
    def _has_entity_path(
        *,
        start: str,
        targets: set[str],
        context: ContextBundle,
    ) -> bool:
        if not targets:
            return False
        adjacency: dict[str, set[str]] = defaultdict(set)
        for fact in context.facts:
            if fact.object_id is None:
                continue
            adjacency[fact.subject_id].add(fact.object_id)
            adjacency[fact.object_id].add(fact.subject_id)

        queue = deque((start,))
        visited = {start}
        while queue:
            entity = queue.popleft()
            for other in adjacency.get(entity, ()):
                if other in targets:
                    return True
                if other not in visited:
                    visited.add(other)
                    queue.append(other)
        return False

    @staticmethod
    def _validate_retrieval_binding(
        context: ContextBundle,
        *,
        event: CanonicalEvent,
        case: CaseRef,
        cutoff: datetime,
        horizon: str,
        operation_time: datetime,
    ) -> None:
        if not isinstance(context, ContextBundle):
            raise TypeError("context_retriever.retrieve must return ContextBundle")
        if context.event.event_id != event.event_id:
            raise ValueError("retriever returned a different event")
        if context.case != case:
            raise ValueError("retriever returned a different case")
        if context.cutoff != cutoff:
            raise ValueError("retriever returned a different cutoff")
        if context.horizon != horizon:
            raise ValueError("retriever returned a different horizon")
        if context.retrieved_at != operation_time:
            raise ValueError("retriever returned an unbound retrieval timestamp")

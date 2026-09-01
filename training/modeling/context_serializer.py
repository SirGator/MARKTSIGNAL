"""Deterministic ContextBundle serialization for EconomyEncoder.

Output format (single text sequence):

    [CLS]
    [EVENT] type=<type>; summary=<summary>; subjects=<entities>; ...
    [CASE] type=<case_type>; name=[CASE_ENTITY]
    [HORIZON] <horizon>
    [CONTEXT] subject=<entity>; predicate=<predicate>; ...
    [HISTORICAL_EVENT] ...
    [SEP]

The target company identity is masked as [CASE_ENTITY] so the model
cannot memorize specific company names.

This serializer is training-local; its contract version is embedded in checkpoints.

The runtime validates this serializer version before loading a checkpoint.
"""

from __future__ import annotations

import re

from training.domain import CanonicalEvent, CaseRef, ContextBundle, ContextFact


NO_SUMMARY = "[NO_SUMMARY]"
SUMMARY_MODE_FULL = "full"
SUMMARY_MODE_NONE = "none"
SUMMARY_MODES = frozenset((SUMMARY_MODE_FULL, SUMMARY_MODE_NONE))
SERIALIZER_CONTRACT_VERSION = "context-serializer-v3"
# v2 checkpoints serialized facts/historical events in caller-supplied order.
# v3 sorts both canonically so two semantically identical ContextBundles always
# produce identical token sequences.  Reading v2 artifacts stays supported;
# saving always writes the current version.
_LEGACY_SERIALIZER_CONTRACT_VERSIONS = frozenset(
    (
        "context-serializer-v2",
        SERIALIZER_CONTRACT_VERSION,
    )
)


def normalize_summary_mode(value: str) -> str:
    """Return one canonical summary mode or reject an unsafe value."""

    if not isinstance(value, str):
        raise TypeError("summary_mode must be a string")
    normalized = value.strip().casefold()
    if normalized not in SUMMARY_MODES:
        raise ValueError(
            f"summary_mode must be one of {sorted(SUMMARY_MODES)}, got {value!r}"
        )
    return normalized


def _fact_sort_key(fact: ContextFact) -> tuple[str, ...]:
    """Stable canonical order: predicate, subject, object, value, fact_id."""

    return (
        fact.predicate.casefold(),
        fact.subject_id.casefold(),
        ("" if fact.object_id is None else fact.object_id.casefold()),
        ("" if fact.value is None else str(fact.value).casefold()),
        fact.fact_id.casefold(),
    )


def _historical_event_sort_key(event: CanonicalEvent) -> tuple[object, ...]:
    """Stable canonical order for retrieved historical comparisons."""

    return (
        event.occurred_at,
        event.event_id.casefold(),
    )


class ContextSerializer:
    """Serialize structured context into one masked text sequence."""

    CASE_MARKER = "[CASE_ENTITY]"

    def __init__(
        self,
        *,
        summary_mode: str = SUMMARY_MODE_FULL,
        allow_no_summary_marker: bool = True,
    ) -> None:
        if not isinstance(allow_no_summary_marker, bool):
            raise TypeError("allow_no_summary_marker must be a boolean")
        self._summary_mode = normalize_summary_mode(summary_mode)
        self._allow_no_summary_marker = allow_no_summary_marker

    @property
    def summary_mode(self) -> str:
        return self._summary_mode

    @property
    def allows_no_summary_marker(self) -> bool:
        return self._allow_no_summary_marker

    def serialize(self, context: ContextBundle, *, horizon: str) -> str:
        if not isinstance(context, ContextBundle):
            raise TypeError("context must be a ContextBundle")
        if not isinstance(horizon, str) or not horizon.strip():
            raise ValueError("horizon must be a non-empty string")
        if context.horizon != horizon.strip():
            raise ValueError("horizon does not match the retrieved ContextBundle")

        lines = ["[CLS]"]
        lines.append("[EVENT] " + self._event_text(context.event, context.case))
        lines.append("[CASE] " + self._case_text(context.case))
        lines.append("[HORIZON] " + self._clean(horizon))
        # Canonical ordering makes serialization independent of the order in
        # which the retriever delivered facts or historical events.  Two
        # semantically identical ContextBundles produce identical token
        # sequences (see SERIALIZER_CONTRACT_VERSION v3).
        for fact in sorted(context.facts, key=_fact_sort_key):
            lines.append("[CONTEXT] " + self._fact_text(fact, context.case))
        for hist in sorted(
            context.historical_events,
            key=_historical_event_sort_key,
        ):
            lines.append("[HISTORICAL_EVENT] " + self._event_text(hist, context.case))
        lines.append("[SEP]")

        return "\n".join(lines)

    def serialize_text(
        self,
        *,
        event_text: str,
        case_type: str,
        horizon: str,
        context_text: str,
    ) -> str:
        """Serialize raw text fields into the same format.

        Kept for raw-text diagnostics and legacy callers.  Structured training
        examples use :meth:`serialize` through ``training.bridge``.
        """
        lines = ["[CLS]"]
        explicit_marker = event_text.strip() == NO_SUMMARY
        if explicit_marker and not self._allow_no_summary_marker:
            raise ValueError(
                "legacy full-summary checkpoints require a real event summary"
            )
        serialized_event = (
            NO_SUMMARY
            if self._summary_mode == SUMMARY_MODE_NONE or explicit_marker
            else self._clean(event_text)
        )
        lines.append(f"[EVENT] {serialized_event}")
        lines.append(f"[CASE] type={self._clean(case_type)} name={self.CASE_MARKER}")
        lines.append(f"[HORIZON] {self._clean(horizon)}")
        if context_text:
            lines.append(f"[CONTEXT] {self._clean(context_text)}")
        lines.append("[SEP]")
        return "\n".join(lines)

    def _event_text(self, event: CanonicalEvent, case: CaseRef) -> str:
        parts = [f"type={self._clean(event.event_type)}"]
        explicit_marker = event.summary.strip() == NO_SUMMARY
        if explicit_marker and not self._allow_no_summary_marker:
            raise ValueError(
                "legacy full-summary checkpoints require real event summaries"
            )
        summary = (
            NO_SUMMARY
            if self._summary_mode == SUMMARY_MODE_NONE or explicit_marker
            else self._mask_text(event.summary, case)
        )
        parts.append(f"summary={summary}")
        parts.append("subjects=" + self._entities(event.subject_ids, case))
        if event.object_ids:
            parts.append("objects=" + self._entities(event.object_ids, case))
        if event.direction is not None:
            parts.append(f"direction={self._clean(event.direction)}")
        if event.magnitude is not None:
            parts.append(f"magnitude={event.magnitude:g}")
        if event.unit is not None:
            parts.append(f"unit={self._clean(event.unit)}")
        parts.append(f"confidence={event.confidence:g}")
        return "; ".join(parts)

    def _case_text(self, case: CaseRef) -> str:
        return f"type={self._clean(case.case_type)} name={self.CASE_MARKER}"

    def _fact_text(self, fact: ContextFact, case: CaseRef) -> str:
        parts = [
            f"subject={self._entity(fact.subject_id, case)}",
            f"predicate={self._clean(fact.predicate)}",
        ]
        if fact.object_id is not None:
            parts.append(f"object={self._entity(fact.object_id, case)}")
        if fact.value is not None:
            parts.append(f"value={self._mask_text(str(fact.value), case)}")
        if fact.unit is not None:
            parts.append(f"unit={self._clean(fact.unit)}")
        parts.append(f"confidence={fact.confidence:g}")
        return "; ".join(parts)

    def _entities(self, entities: tuple[str, ...], case: CaseRef) -> str:
        return ",".join(self._entity(entity, case) for entity in entities)

    def _entity(self, entity: str, case: CaseRef) -> str:
        if entity == case.entity_id:
            return self.CASE_MARKER
        return self._clean(entity)

    def _mask_text(self, value: str, case: CaseRef) -> str:
        masked = self._clean(value)
        for identity in (case.entity_id, case.name):
            if identity:
                masked = re.sub(
                    rf"(?<!\w){re.escape(self._clean(identity))}(?!\w)",
                    self.CASE_MARKER,
                    masked,
                    flags=re.IGNORECASE,
                )
        return masked

    @staticmethod
    def _clean(value: str) -> str:
        escaped = value.replace("[", "(").replace("]", ")").replace(";", ",")
        return " ".join(escaped.split())

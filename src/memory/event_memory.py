"""In-memory canonical event store with conservative exact deduplication."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from threading import RLock
from typing import Iterable

from src.domain import CanonicalEvent, ProvenanceRecord, _utc_datetime


class ExactEventDeduplicator:
    """Deduplicate only events with identical normalized canonical content.

    V1 intentionally performs no fuzzy text, entity, magnitude, or timestamp
    matching.  Similar events can be reviewed as candidates later without the
    memory silently merging distinct economic developments.
    """

    @staticmethod
    def key(event: CanonicalEvent) -> tuple[object, ...]:
        if not isinstance(event, CanonicalEvent):
            raise TypeError("event must be a CanonicalEvent")
        return event.deduplication_key

    def are_duplicates(
        self,
        left: CanonicalEvent,
        right: CanonicalEvent,
    ) -> bool:
        return self.key(left) == self.key(right)


class EventMemory:
    """Thread-safe V1 event memory.

    The implementation is deliberately storage-agnostic at the contract level.
    A persistent adapter can later preserve the same ``add/get/search`` API.
    """

    def __init__(self, *, deduplicator: ExactEventDeduplicator | None = None) -> None:
        self._deduplicator = deduplicator or ExactEventDeduplicator()
        self._events: dict[str, CanonicalEvent] = {}
        self._aliases: dict[str, str] = {}
        self._dedup_index: dict[tuple[object, ...], str] = {}
        self._lock = RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def add(self, event: CanonicalEvent) -> tuple[CanonicalEvent, bool]:
        """Store an event, returning ``(canonical_event, was_created)``.

        Exact duplicates retain the first canonical event ID.  Only provenance
        and the earliest observation time are merged; canonical semantics are
        never replaced by a later extraction.
        """

        if not isinstance(event, CanonicalEvent):
            raise TypeError("event must be a CanonicalEvent")

        key = self._deduplicator.key(event)
        with self._lock:
            existing_id = self._aliases.get(event.event_id, event.event_id)
            same_id_event = self._events.get(existing_id)
            if same_id_event is not None and not self._deduplicator.are_duplicates(
                same_id_event,
                event,
            ):
                raise ValueError(
                    f"event_id {event.event_id!r} already refers to a different event"
                )

            canonical_id = self._dedup_index.get(key)
            if canonical_id is None and same_id_event is None:
                self._events[event.event_id] = event
                self._aliases[event.event_id] = event.event_id
                self._dedup_index[key] = event.event_id
                return event, True

            canonical_id = canonical_id or existing_id
            existing = self._events[canonical_id]
            merged = self._merge_provenance(existing, event)
            self._events[canonical_id] = merged
            self._aliases[event.event_id] = canonical_id
            self._dedup_index[key] = canonical_id
            return merged, False

    def extend(
        self,
        events: Iterable[CanonicalEvent],
    ) -> tuple[tuple[CanonicalEvent, bool], ...]:
        return tuple(self.add(event) for event in events)

    def find_duplicate(self, event: CanonicalEvent) -> CanonicalEvent | None:
        """Return an existing exact duplicate without mutating the memory."""

        key = self._deduplicator.key(event)
        with self._lock:
            event_id = self._dedup_index.get(key)
            return None if event_id is None else self._events[event_id]

    def get(
        self,
        event_id: str,
        *,
        cutoff: datetime | None = None,
    ) -> CanonicalEvent | None:
        """Get an event or a cutoff-safe snapshot of it."""

        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("event_id must be a non-empty string")
        event_id = event_id.strip()
        with self._lock:
            canonical_id = self._aliases.get(event_id, event_id)
            event = self._events.get(canonical_id)
            if event is None or cutoff is None:
                return event
            return event.as_observed_at(cutoff)

    def search(
        self,
        *,
        cutoff: datetime,
        event_type: str | None = None,
        subject_ids: tuple[str, ...] = (),
        object_ids: tuple[str, ...] = (),
        occurred_after: datetime | None = None,
        occurred_before: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[CanonicalEvent, ...]:
        """Search events that were known by ``cutoff``.

        Entity filters use subset semantics: all requested IDs must occur on the
        event.  Occurrence bounds are inclusive and do not replace the mandatory
        observation-time cutoff.
        """

        safe_cutoff = _utc_datetime("cutoff", cutoff)
        after = (
            None
            if occurred_after is None
            else _utc_datetime("occurred_after", occurred_after)
        )
        before = (
            None
            if occurred_before is None
            else _utc_datetime("occurred_before", occurred_before)
        )
        if after is not None and before is not None and after > before:
            raise ValueError("occurred_after must be before or equal to occurred_before")
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise TypeError("limit must be an integer or None")
            if limit < 0:
                raise ValueError("limit must be non-negative")

        required_subjects = self._normalized_filter_ids("subject_ids", subject_ids)
        required_objects = self._normalized_filter_ids("object_ids", object_ids)
        wanted_type = None if event_type is None else event_type.strip().casefold()
        if event_type is not None and not wanted_type:
            raise ValueError("event_type must be non-empty when supplied")

        with self._lock:
            candidates = tuple(self._events.values())

        matches: list[CanonicalEvent] = []
        for event in candidates:
            snapshot = event.as_observed_at(safe_cutoff)
            if snapshot is None:
                continue
            if wanted_type is not None and snapshot.event_type.casefold() != wanted_type:
                continue
            event_subjects = {value.casefold() for value in snapshot.subject_ids}
            if not required_subjects.issubset(event_subjects):
                continue
            event_objects = {value.casefold() for value in snapshot.object_ids}
            if not required_objects.issubset(event_objects):
                continue
            if after is not None and snapshot.occurred_at < after:
                continue
            if before is not None and snapshot.occurred_at > before:
                continue
            matches.append(snapshot)

        matches.sort(
            key=lambda item: (item.occurred_at, item.observed_at, item.event_id),
            reverse=True,
        )
        if limit is not None:
            matches = matches[:limit]
        return tuple(matches)

    @staticmethod
    def _merge_provenance(
        existing: CanonicalEvent,
        duplicate: CanonicalEvent,
    ) -> CanonicalEvent:
        provenance_by_value: dict[ProvenanceRecord, None] = {
            source: None for source in existing.provenance
        }
        provenance_by_value.update({source: None for source in duplicate.provenance})
        provenance = tuple(provenance_by_value)
        if duplicate.observed_at < existing.observed_at:
            cutoff_safe_confidence = duplicate.confidence
        elif duplicate.observed_at > existing.observed_at:
            cutoff_safe_confidence = existing.confidence
        else:
            # Both extractions were available at the same point in time.  The
            # conservative deterministic value avoids insertion-order effects.
            cutoff_safe_confidence = min(existing.confidence, duplicate.confidence)

        return replace(
            existing,
            observed_at=min(existing.observed_at, duplicate.observed_at),
            confidence=cutoff_safe_confidence,
            provenance=provenance,
        )

    @staticmethod
    def _normalized_filter_ids(name: str, values: tuple[str, ...]) -> set[str]:
        if isinstance(values, str):
            raise TypeError(f"{name} must be an iterable of ids, not a string")
        normalized: set[str] = set()
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must contain only non-empty strings")
            normalized.add(value.strip().casefold())
        return normalized

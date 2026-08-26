"""Thread-safe append-only in-memory storage for social source data."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from datetime import datetime
from threading import RLock
from typing import Iterable

from .domain import EngagementSnapshot, SocialObservation, _utc_datetime


@dataclass(frozen=True, slots=True)
class SocialStoreAppendResult:
    """Atomic result of persisting one collected social batch."""

    observations: tuple[tuple[SocialObservation, bool], ...]
    engagement_snapshots: tuple[tuple[EngagementSnapshot, bool], ...]
    ingested_at: datetime


class InMemorySocialObservationStore:
    """Append-only V1 store with deterministic, cutoff-safe reads.

    The source identity is ``(source, external_id)``.  Replaying unchanged
    content is idempotent even with later collection timestamps.  Edited
    content is appended as a new immutable revision; cutoff reads expose only
    the latest revision that was actually available at that cutoff.
    """

    def __init__(
        self,
        observations: Iterable[SocialObservation] = (),
        engagement_snapshots: Iterable[EngagementSnapshot] = (),
    ) -> None:
        self._observations: dict[tuple[str, str, int], SocialObservation] = {}
        self._revisions: dict[tuple[str, str], list[int]] = {}
        self._engagement: dict[tuple[str, str, int], EngagementSnapshot] = {}
        self._lock = RLock()
        for observation in observations:
            self.add_observation(observation)
        for snapshot in engagement_snapshots:
            self.add_engagement_snapshot(snapshot)

    def __len__(self) -> int:
        with self._lock:
            return len(self._observations)

    @property
    def engagement_count(self) -> int:
        with self._lock:
            return len(self._engagement)

    def add_observation(
        self,
        observation: SocialObservation,
    ) -> tuple[SocialObservation, bool]:
        """Append an observation, returning ``(stored, was_created)``."""

        if not isinstance(observation, SocialObservation):
            raise TypeError("observation must be a SocialObservation")
        with self._lock:
            return self._add_observation_locked(observation)

    append_observation = add_observation

    def append_batch(
        self,
        observations: Iterable[SocialObservation],
        engagement_snapshots: Iterable[EngagementSnapshot],
        *,
        ingested_at: datetime,
    ) -> SocialStoreAppendResult:
        """Atomically persist a collected batch at the actual append time.

        The collector's staging timestamps are never trusted as persistence
        timestamps.  Every record receives the same store commit timestamp.
        On any validation/conflict error the complete batch is rolled back.
        """

        persisted_at = _utc_datetime("ingested_at", ingested_at)
        observation_batch = tuple(observations)
        engagement_batch = tuple(engagement_snapshots)
        if any(not isinstance(item, SocialObservation) for item in observation_batch):
            raise TypeError("observations must contain only SocialObservation values")
        if any(not isinstance(item, EngagementSnapshot) for item in engagement_batch):
            raise TypeError(
                "engagement_snapshots must contain only EngagementSnapshot values"
            )
        for item in (*observation_batch, *engagement_batch):
            if item.known_at > persisted_at:
                raise ValueError("ingested_at must not be before any record known_at")

        with self._lock:
            observations_before = dict(self._observations)
            revisions_before = {
                key: list(revisions) for key, revisions in self._revisions.items()
            }
            engagement_before = dict(self._engagement)
            try:
                observation_results = tuple(
                    self._add_observation_locked(
                        replace(item, ingested_at=persisted_at)
                    )
                    for item in observation_batch
                )
                engagement_results = tuple(
                    self._add_engagement_locked(
                        replace(item, ingested_at=persisted_at)
                    )
                    for item in engagement_batch
                )
            except Exception:
                self._observations = observations_before
                self._revisions = revisions_before
                self._engagement = engagement_before
                raise
        return SocialStoreAppendResult(
            observations=observation_results,
            engagement_snapshots=engagement_results,
            ingested_at=persisted_at,
        )

    def add_engagement_snapshot(
        self,
        snapshot: EngagementSnapshot,
    ) -> tuple[EngagementSnapshot, bool]:
        """Append one engagement version without replacing prior counters."""

        if not isinstance(snapshot, EngagementSnapshot):
            raise TypeError("snapshot must be an EngagementSnapshot")
        with self._lock:
            return self._add_engagement_locked(snapshot)

    append_engagement_snapshot = add_engagement_snapshot

    def get_observation(
        self,
        source: str,
        external_id: str,
        *,
        cutoff: datetime | None = None,
    ) -> SocialObservation | None:
        source_key = self._normalised_text("source", source).casefold()
        external_key = self._normalised_text("external_id", external_id)
        with self._lock:
            revisions = tuple(self._revisions.get((source_key, external_key), ()))
            candidates = tuple(
                self._observations[(source_key, external_key, revision)]
                for revision in revisions
            )
        if cutoff is None:
            return None if not candidates else candidates[-1]
        safe_cutoff = _utc_datetime("cutoff", cutoff)
        visible = [item for item in candidates if item.is_available_at(safe_cutoff)]
        return None if not visible else self._latest_visible(visible)

    def query_observations(
        self,
        *,
        cutoff: datetime,
        entity_id: str | None = None,
        sources: tuple[str, ...] = (),
        authored_after: datetime | None = None,
        authored_at_or_before: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[SocialObservation, ...]:
        """Return records visible at ``cutoff`` in deterministic order.

        Authored-time bounds form a right-closed interval:
        ``authored_after < authored_at <= authored_at_or_before``.  Crucially,
        those bounds do not replace the mandatory availability cutoff.
        """

        safe_cutoff = _utc_datetime("cutoff", cutoff)
        after = (
            None
            if authored_after is None
            else _utc_datetime("authored_after", authored_after)
        )
        before = (
            None
            if authored_at_or_before is None
            else _utc_datetime("authored_at_or_before", authored_at_or_before)
        )
        if after is not None and before is not None and after >= before:
            raise ValueError("authored_after must be before authored_at_or_before")
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise TypeError("limit must be an integer or None")
            if limit < 0:
                raise ValueError("limit must be non-negative")

        wanted_entity = (
            None
            if entity_id is None
            else self._normalised_text("entity_id", entity_id).casefold()
        )
        if isinstance(sources, str):
            raise TypeError("sources must be a tuple of source names")
        wanted_sources = {
            self._normalised_text("sources", source).casefold() for source in sources
        }

        with self._lock:
            candidates = tuple(self._observations.values())

        visible_by_source_id: dict[tuple[str, str], list[SocialObservation]] = {}
        for observation in candidates:
            if observation.is_available_at(safe_cutoff):
                visible_by_source_id.setdefault(
                    (observation.source, observation.external_id),
                    [],
                ).append(observation)

        matches: list[SocialObservation] = []
        for revisions in visible_by_source_id.values():
            observation = self._latest_visible(revisions)
            if wanted_entity is not None and wanted_entity not in {
                value.casefold() for value in observation.entity_ids
            }:
                continue
            if wanted_sources and observation.source not in wanted_sources:
                continue
            if after is not None and observation.authored_at <= after:
                continue
            if before is not None and observation.authored_at > before:
                continue
            matches.append(observation)

        matches.sort(
            key=lambda item: (
                item.authored_at,
                item.available_at,
                item.source,
                item.external_id,
            )
        )
        if limit is not None:
            matches = matches[:limit]
        return tuple(matches)

    def revisions_for(
        self,
        source: str,
        external_id: str,
        *,
        cutoff: datetime | None = None,
    ) -> tuple[SocialObservation, ...]:
        """Return append-only content revisions in stable revision order."""

        source_key = self._normalised_text("source", source).casefold()
        external_key = self._normalised_text("external_id", external_id)
        with self._lock:
            revision_ids = tuple(
                self._revisions.get((source_key, external_key), ())
            )
            revisions = tuple(
                self._observations[(source_key, external_key, revision)]
                for revision in revision_ids
            )
        if cutoff is None:
            return revisions
        safe_cutoff = _utc_datetime("cutoff", cutoff)
        return tuple(item for item in revisions if item.is_available_at(safe_cutoff))

    def query_engagement_snapshots(
        self,
        *,
        source: str,
        external_id: str,
        cutoff: datetime,
    ) -> tuple[EngagementSnapshot, ...]:
        """Return every engagement version available by ``cutoff``."""

        source_key = self._normalised_text("source", source).casefold()
        external_key = self._normalised_text("external_id", external_id)
        safe_cutoff = _utc_datetime("cutoff", cutoff)
        with self._lock:
            candidates = tuple(self._engagement.values())
        visible = [
            snapshot
            for snapshot in candidates
            if snapshot.source == source_key
            and snapshot.external_id == external_key
            and snapshot.is_available_at(safe_cutoff)
        ]
        visible.sort(
            key=lambda item: (
                item.known_at,
                item.ingested_at,
                item.version,
            )
        )
        return tuple(visible)

    def latest_engagement_snapshot(
        self,
        *,
        source: str,
        external_id: str,
        cutoff: datetime,
    ) -> EngagementSnapshot | None:
        visible = self.query_engagement_snapshots(
            source=source,
            external_id=external_id,
            cutoff=cutoff,
        )
        return None if not visible else visible[-1]

    @staticmethod
    def _normalised_text(name: str, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _same_except_ingestion(left: object, right: object) -> bool:
        """Compare immutable source payloads while tolerating retry time."""

        return all(
            getattr(left, item.name) == getattr(right, item.name)
            for item in fields(left)
            if item.name != "ingested_at"
        )

    @staticmethod
    def _same_observation_payload(
        left: SocialObservation,
        right: SocialObservation,
    ) -> bool:
        """Compare source content/enrichment, not collection metadata."""

        ignored = {"known_at", "ingested_at", "revision"}
        return all(
            getattr(left, item.name) == getattr(right, item.name)
            for item in fields(left)
            if item.name not in ignored
        )

    def _add_observation_locked(
        self,
        observation: SocialObservation,
    ) -> tuple[SocialObservation, bool]:
        source_key = (observation.source, observation.external_id)
        revision_ids = self._revisions.get(source_key, [])
        if revision_ids:
            latest = self._observations[(*source_key, revision_ids[-1])]
            if self._same_observation_payload(latest, observation):
                return latest, False
            revision = revision_ids[-1] + 1
        else:
            revision = 1
        stored = replace(observation, revision=revision)
        self._observations[(*source_key, revision)] = stored
        self._revisions.setdefault(source_key, []).append(revision)
        return stored, True

    def _add_engagement_locked(
        self,
        snapshot: EngagementSnapshot,
    ) -> tuple[EngagementSnapshot, bool]:
        observation_key = (snapshot.source, snapshot.external_id)
        if not self._revisions.get(observation_key):
            raise KeyError(
                f"engagement references unknown observation "
                f"{snapshot.observation_id!r}"
            )
        earliest = self._observations[(*observation_key, 1)]
        if snapshot.known_at < earliest.known_at:
            raise ValueError("engagement known_at must not predate its observation")
        key = (*observation_key, snapshot.version)
        existing = self._engagement.get(key)
        if existing is None:
            self._engagement[key] = snapshot
            return snapshot, True
        if self._same_except_ingestion(existing, snapshot):
            return existing, False
        raise ValueError(
            f"engagement version {snapshot.evidence_id!r} already exists "
            "with different immutable content"
        )

    @staticmethod
    def _latest_visible(
        revisions: Iterable[SocialObservation],
    ) -> SocialObservation:
        return max(
            revisions,
            key=lambda item: (
                item.known_at,
                item.ingested_at,
                item.revision,
            ),
        )

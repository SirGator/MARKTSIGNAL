"""Immutable point-in-time contracts for social source observations.

``authored_at`` describes when a post was created, ``known_at`` when the
source made it available to the collector, and ``ingested_at`` when this
system actually persisted it.  A historical feature computation must respect
all three and can never treat a later backfill as if it had been present
earlier.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math


SOCIAL_SOURCE_SCHEMA_VERSION = "social-source-v1"


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


def _positive_duration(name: str, value: timedelta) -> timedelta:
    if not isinstance(value, timedelta):
        raise TypeError(f"{name} must be a timedelta")
    if value <= timedelta(0):
        raise ValueError(f"{name} must be positive")
    return value


def _finite_number(name: str, value: float | int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_finite_number(
    name: str,
    value: float | int | None,
) -> float | None:
    if value is None:
        return None
    return _finite_number(name, value)


def _unit_interval(name: str, value: float | int) -> float:
    result = _finite_number(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


def _optional_unit_interval(
    name: str,
    value: float | int | None,
) -> float | None:
    if value is None:
        return None
    return _unit_interval(name, value)


def _signed_unit_interval(name: str, value: float | int) -> float:
    result = _finite_number(name, value)
    if not -1.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between -1 and 1")
    return result


def _optional_signed_unit_interval(
    name: str,
    value: float | int | None,
) -> float | None:
    if value is None:
        return None
    return _signed_unit_interval(name, value)


def _normalised_ids(name: str, values: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{name} must be a tuple of strings, not a string")
    cleaned = tuple(_required_text(name, value) for value in values)
    by_casefold: dict[str, str] = {}
    for value in cleaned:
        key = value.casefold()
        previous = by_casefold.get(key)
        if previous is not None:
            raise ValueError(f"{name} contains duplicate id {value!r}")
        by_casefold[key] = value
    return tuple(sorted(cleaned, key=lambda value: (value.casefold(), value)))


def _counter(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer or None")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True, slots=True)
class SocialObservation:
    """One immutable social post/comment as first seen by the system.

    ``stance`` is an optional directional classification in ``[-1, +1]``:
    ``-1`` is bearish, ``0`` neutral, and ``+1`` bullish.  Missing stance is
    represented by ``None`` and is deliberately different from neutral.
    ``content_hash`` supports cross-source duplicate measurement; observations
    with different external IDs are retained rather than silently merged.
    """

    source: str
    external_id: str
    entity_ids: tuple[str, ...]
    authored_at: datetime
    known_at: datetime
    ingested_at: datetime
    text: str
    content_hash: str
    author_hash: str | None = None
    stance: float | None = None
    stance_confidence: float | None = None
    uri: str | None = None
    revision: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _required_text("source", self.source).casefold())
        object.__setattr__(
            self,
            "external_id",
            _required_text("external_id", self.external_id),
        )
        entity_ids = _normalised_ids("entity_ids", self.entity_ids)
        if not entity_ids:
            raise ValueError("entity_ids must contain at least one entity")
        object.__setattr__(self, "entity_ids", entity_ids)

        authored_at = _utc_datetime("authored_at", self.authored_at)
        known_at = _utc_datetime("known_at", self.known_at)
        ingested_at = _utc_datetime("ingested_at", self.ingested_at)
        if known_at < authored_at:
            raise ValueError("known_at must not be before authored_at")
        if ingested_at < known_at:
            raise ValueError("ingested_at must not be before known_at")
        object.__setattr__(self, "authored_at", authored_at)
        object.__setattr__(self, "known_at", known_at)
        object.__setattr__(self, "ingested_at", ingested_at)

        object.__setattr__(self, "text", _required_text("text", self.text))
        object.__setattr__(
            self,
            "content_hash",
            _required_text("content_hash", self.content_hash),
        )
        object.__setattr__(
            self,
            "author_hash",
            _optional_text("author_hash", self.author_hash),
        )
        stance = _optional_signed_unit_interval("stance", self.stance)
        stance_confidence = _optional_unit_interval(
            "stance_confidence",
            self.stance_confidence,
        )
        if (stance is None) != (stance_confidence is None):
            raise ValueError("stance and stance_confidence must be supplied together")
        object.__setattr__(self, "stance", stance)
        object.__setattr__(self, "stance_confidence", stance_confidence)
        object.__setattr__(self, "uri", _optional_text("uri", self.uri))
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError("revision must be an integer")
        if self.revision < 1:
            raise ValueError("revision must be at least 1")

    @property
    def evidence_id(self) -> str:
        base = f"{self.source}:{self.external_id}"
        return base if self.revision == 1 else f"{base}:revision:{self.revision}"

    @property
    def observation_id(self) -> str:
        """Stable source identity shared by every content revision."""

        return f"{self.source}:{self.external_id}"

    @property
    def provenance(self) -> str:
        """Return the best inspectable provenance reference available."""

        return self.uri or self.evidence_id

    @property
    def available_at(self) -> datetime:
        """Earliest cutoff at which the local system could use the record."""

        return max(self.known_at, self.ingested_at)

    def is_available_at(self, cutoff: datetime) -> bool:
        return self.available_at <= _utc_datetime("cutoff", cutoff)


@dataclass(frozen=True, slots=True)
class EngagementSnapshot:
    """Append-only, versioned engagement counters for one observation.

    Counters are optional because platforms expose different metrics.  A new
    measurement receives a new ``version``; an existing version can only be
    replayed idempotently, never overwritten.
    """

    source: str
    external_id: str
    version: int
    known_at: datetime
    ingested_at: datetime
    score: float | None = None
    likes: int | None = None
    upvotes: int | None = None
    comments: int | None = None
    shares: int | None = None
    views: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _required_text("source", self.source).casefold())
        object.__setattr__(
            self,
            "external_id",
            _required_text("external_id", self.external_id),
        )
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError("version must be an integer")
        if self.version < 1:
            raise ValueError("version must be at least 1")
        known_at = _utc_datetime("known_at", self.known_at)
        ingested_at = _utc_datetime("ingested_at", self.ingested_at)
        if ingested_at < known_at:
            raise ValueError("ingested_at must not be before known_at")
        object.__setattr__(self, "known_at", known_at)
        object.__setattr__(self, "ingested_at", ingested_at)
        object.__setattr__(self, "score", _optional_finite_number("score", self.score))
        for name in ("likes", "upvotes", "comments", "shares", "views"):
            object.__setattr__(self, name, _counter(name, getattr(self, name)))
        if all(
            value is None
            for value in (
                self.score,
                self.likes,
                self.upvotes,
                self.comments,
                self.shares,
                self.views,
            )
        ):
            raise ValueError("an engagement snapshot requires at least one metric")

    @property
    def observation_id(self) -> str:
        return f"{self.source}:{self.external_id}"

    @property
    def evidence_id(self) -> str:
        return f"{self.observation_id}:engagement:v{self.version}"

    @property
    def available_at(self) -> datetime:
        return max(self.known_at, self.ingested_at)

    def is_available_at(self, cutoff: datetime) -> bool:
        return self.available_at <= _utc_datetime("cutoff", cutoff)


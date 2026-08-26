"""Platform-neutral social ingestion and a Reddit JSON V1 adapter.

The module deliberately keeps transport, parsing, entity resolution and
sentiment scoring injectable.  The concrete Reddit collector needs no
credentials, performs no network access at import time, and can be exercised
entirely with an in-memory HTTP transport.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from hashlib import sha256
import html
import json
import math
import re
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urljoin
from urllib.request import Request, urlopen

from .domain import (
    EngagementSnapshot,
    SocialObservation,
    _optional_text,
    _required_text,
    _utc_datetime,
)
from .store import (
    InMemorySocialObservationStore,
    SocialStoreAppendResult,
)


_DELETED_TEXT = frozenset({"[deleted]", "[removed]"})
_REDDIT_KINDS = {"t1": "comment", "t3": "post"}


def _optional_utc_datetime(
    name: str,
    value: datetime | None,
) -> datetime | None:
    if value is None:
        return None
    return _utc_datetime(name, value)


def _normalise_space(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def _stable_hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _content_hash(text: str) -> str:
    return _stable_hash(_normalise_space(text).casefold())


def _author_hash(source: str, author: str | None) -> str | None:
    if author is None:
        return None
    cleaned = _normalise_space(author)
    if not cleaned or cleaned.casefold() in _DELETED_TEXT:
        return None
    # Source scoping avoids creating a cross-platform tracking identifier.
    return _stable_hash(f"{source.casefold()}\0{cleaned.casefold()}")


def _finite_optional_number(name: str, value: object) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _nonnegative_optional_int(name: str, value: object) -> int | None:
    if value is None or isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


@dataclass(frozen=True, slots=True)
class RawEngagement:
    """Platform-neutral engagement values captured in one source response."""

    score: float | None = None
    likes: int | None = None
    upvotes: int | None = None
    comments: int | None = None
    shares: int | None = None
    views: int | None = None

    def __post_init__(self) -> None:
        score = _finite_optional_number("score", self.score)
        if self.score is not None and score is None:
            raise ValueError("score must be a finite number or None")
        object.__setattr__(self, "score", score)
        for name in ("likes", "upvotes", "comments", "shares", "views"):
            value = getattr(self, name)
            cleaned = _nonnegative_optional_int(name, value)
            if value is not None and cleaned is None:
                raise ValueError(f"{name} must be a non-negative integer or None")
            object.__setattr__(self, name, cleaned)

    @property
    def has_metrics(self) -> bool:
        return any(
            value is not None
            for value in (
                self.score,
                self.likes,
                self.upvotes,
                self.comments,
                self.shares,
                self.views,
            )
        )


@dataclass(frozen=True, slots=True)
class RawSocialPost:
    """Platform-neutral post/comment before entity and stance enrichment.

    ``effective_at`` is intentionally optional and is never inferred from a
    Reddit publication time.  It is reserved for a separately extracted time
    at which the claim itself becomes economically effective.
    """

    source: str
    external_id: str
    kind: str
    published_at: datetime
    effective_at: datetime | None
    known_at: datetime
    ingested_at: datetime
    text: str
    author: str | None = None
    uri: str | None = None
    parent_external_id: str | None = None
    engagement: RawEngagement = field(default_factory=RawEngagement)

    def __post_init__(self) -> None:
        source = _required_text("source", self.source).casefold()
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "external_id",
            _required_text("external_id", self.external_id),
        )
        kind = _required_text("kind", self.kind).casefold()
        if kind not in {"post", "comment"}:
            raise ValueError("kind must be 'post' or 'comment'")
        object.__setattr__(self, "kind", kind)
        published_at = _utc_datetime("published_at", self.published_at)
        effective_at = _optional_utc_datetime("effective_at", self.effective_at)
        known_at = _utc_datetime("known_at", self.known_at)
        ingested_at = _utc_datetime("ingested_at", self.ingested_at)
        if known_at < published_at:
            raise ValueError("known_at must not be before published_at")
        if ingested_at < known_at:
            raise ValueError("ingested_at must not be before known_at")
        object.__setattr__(self, "published_at", published_at)
        object.__setattr__(self, "effective_at", effective_at)
        object.__setattr__(self, "known_at", known_at)
        object.__setattr__(self, "ingested_at", ingested_at)
        text = _normalise_space(_required_text("text", self.text))
        if text.casefold() in _DELETED_TEXT:
            raise ValueError("deleted or removed text is not ingestible")
        object.__setattr__(self, "text", text)
        author = _optional_text("author", self.author)
        if author is not None and author.casefold() in _DELETED_TEXT:
            author = None
        object.__setattr__(self, "author", author)
        object.__setattr__(self, "uri", _optional_text("uri", self.uri))
        object.__setattr__(
            self,
            "parent_external_id",
            _optional_text("parent_external_id", self.parent_external_id),
        )
        if not isinstance(self.engagement, RawEngagement):
            raise TypeError("engagement must be a RawEngagement")


@dataclass(frozen=True, slots=True)
class SentimentScore:
    stance: float
    confidence: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.stance, bool)
            or not isinstance(self.stance, (int, float))
            or not math.isfinite(float(self.stance))
            or not -1.0 <= float(self.stance) <= 1.0
        ):
            raise ValueError("stance must be finite and between -1 and 1")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise ValueError("confidence must be finite and between 0 and 1")
        object.__setattr__(self, "stance", float(self.stance))
        object.__setattr__(self, "confidence", float(self.confidence))


@runtime_checkable
class EntityResolver(Protocol):
    def resolve(self, text: str) -> tuple[str, ...]:
        """Return canonical entity IDs mentioned by ``text``."""


@runtime_checkable
class SentimentScorer(Protocol):
    def score(self, text: str) -> SentimentScore | None:
        """Return directional stance, or ``None`` when no stance is known."""


@dataclass(frozen=True, slots=True)
class SocialIngestionIssue:
    external_id: str
    code: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "external_id",
            _required_text("external_id", self.external_id),
        )
        object.__setattr__(self, "code", _required_text("code", self.code))
        object.__setattr__(self, "message", _required_text("message", self.message))


@dataclass(frozen=True, slots=True)
class SocialIngestionBatch:
    raw_posts: tuple[RawSocialPost, ...] = ()
    observations: tuple[SocialObservation, ...] = ()
    engagement_snapshots: tuple[EngagementSnapshot, ...] = ()
    issues: tuple[SocialIngestionIssue, ...] = ()
    after: str | None = None

    def __post_init__(self) -> None:
        contracts = (
            ("raw_posts", self.raw_posts, RawSocialPost),
            ("observations", self.observations, SocialObservation),
            ("engagement_snapshots", self.engagement_snapshots, EngagementSnapshot),
            ("issues", self.issues, SocialIngestionIssue),
        )
        for name, values, expected in contracts:
            if not isinstance(values, tuple) or any(
                not isinstance(value, expected) for value in values
            ):
                raise TypeError(f"{name} must be a tuple of {expected.__name__} values")
        object.__setattr__(self, "after", _optional_text("after", self.after))

    @property
    def skipped_count(self) -> int:
        return len(self.issues)


@runtime_checkable
class SocialPayloadParser(Protocol):
    def parse(
        self,
        payload: object,
        *,
        known_at: datetime,
        ingested_at: datetime,
        engagement_version: int | None = None,
    ) -> SocialIngestionBatch:
        """Parse one source response into immutable social contracts."""


@runtime_checkable
class SocialCollector(Protocol):
    def collect(
        self,
        *,
        max_pages: int = 1,
        start_after: str | None = None,
    ) -> SocialIngestionBatch:
        """Collect and parse one or more source pages."""


class KeywordEntityResolver:
    """Deterministic V1 alias resolver with token-boundary matching."""

    def __init__(self, aliases: Mapping[str, Iterable[str]]) -> None:
        if not isinstance(aliases, Mapping):
            raise TypeError("aliases must be a mapping of entity IDs to aliases")
        patterns: list[tuple[str, tuple[re.Pattern[str], ...]]] = []
        for entity_id, entity_aliases in aliases.items():
            entity = _required_text("entity_id", entity_id)
            if isinstance(entity_aliases, str):
                raise TypeError("each entity alias collection must be an iterable")
            compiled: list[re.Pattern[str]] = []
            seen: set[str] = set()
            for alias in entity_aliases:
                cleaned = _normalise_space(_required_text("alias", alias))
                folded = cleaned.casefold()
                if folded in seen:
                    continue
                seen.add(folded)
                expression = re.escape(cleaned).replace(r"\ ", r"\s+")
                compiled.append(
                    re.compile(rf"(?<!\w){expression}(?!\w)", re.IGNORECASE)
                )
            if not compiled:
                raise ValueError(f"entity {entity!r} requires at least one alias")
            patterns.append((entity, tuple(compiled)))
        self._patterns = tuple(
            sorted(patterns, key=lambda item: (item[0].casefold(), item[0]))
        )

    def resolve(self, text: str) -> tuple[str, ...]:
        cleaned = _required_text("text", text)
        return tuple(
            entity_id
            for entity_id, patterns in self._patterns
            if any(pattern.search(cleaned) for pattern in patterns)
        )


class LexiconSentimentScorer:
    """Small transparent baseline scorer, not a market prediction model."""

    DEFAULT_POSITIVE = (
        "bullish",
        "buy",
        "growth",
        "profit",
        "strong",
        "beats",
        "outperform",
        "steigt",
        "gewinn",
        "wachstum",
        "kaufen",
        "stark",
    )
    DEFAULT_NEGATIVE = (
        "bearish",
        "sell",
        "loss",
        "weak",
        "misses",
        "crash",
        "bankruptcy",
        "fällt",
        "verlust",
        "verkaufen",
        "schwach",
        "insolvenz",
    )

    def __init__(
        self,
        *,
        positive: Iterable[str] = DEFAULT_POSITIVE,
        negative: Iterable[str] = DEFAULT_NEGATIVE,
    ) -> None:
        positive_terms = self._terms("positive", positive)
        negative_terms = self._terms("negative", negative)
        overlap = set(positive_terms).intersection(negative_terms)
        if overlap:
            raise ValueError(f"sentiment terms overlap: {sorted(overlap)!r}")
        self._positive = tuple(self._compile(term) for term in positive_terms)
        self._negative = tuple(self._compile(term) for term in negative_terms)

    def score(self, text: str) -> SentimentScore | None:
        cleaned = _required_text("text", text)
        positive_count = sum(
            len(pattern.findall(cleaned)) for pattern in self._positive
        )
        negative_count = sum(
            len(pattern.findall(cleaned)) for pattern in self._negative
        )
        total = positive_count + negative_count
        if total == 0:
            return None
        stance = (positive_count - negative_count) / total
        # More independent lexical evidence increases confidence, asymptotically.
        confidence = total / (total + 2.0)
        return SentimentScore(stance=stance, confidence=confidence)

    @staticmethod
    def _terms(name: str, values: Iterable[str]) -> tuple[str, ...]:
        if isinstance(values, str):
            raise TypeError(f"{name} terms must be an iterable, not a string")
        result = {
            _normalise_space(_required_text(f"{name} term", value)).casefold()
            for value in values
        }
        if not result:
            raise ValueError(f"{name} terms must not be empty")
        return tuple(sorted(result))

    @staticmethod
    def _compile(term: str) -> re.Pattern[str]:
        expression = re.escape(term).replace(r"\ ", r"\s+")
        return re.compile(rf"(?<!\w){expression}(?!\w)", re.IGNORECASE)


class SocialPayloadError(ValueError):
    """Raised when a source payload cannot be interpreted safely."""


class RedditJsonParser:
    """Normalize Reddit Listing JSON (posts and comments) into social V1."""

    def __init__(
        self,
        *,
        entity_resolver: EntityResolver,
        sentiment_scorer: SentimentScorer | None = None,
        reddit_origin: str = "https://www.reddit.com",
    ) -> None:
        if not isinstance(entity_resolver, EntityResolver):
            raise TypeError("entity_resolver must implement EntityResolver")
        if sentiment_scorer is not None and not isinstance(
            sentiment_scorer,
            SentimentScorer,
        ):
            raise TypeError("sentiment_scorer must implement SentimentScorer")
        self._entity_resolver = entity_resolver
        self._sentiment_scorer = sentiment_scorer
        self._reddit_origin = _required_text("reddit_origin", reddit_origin).rstrip("/") + "/"

    def parse(
        self,
        payload: object,
        *,
        known_at: datetime,
        ingested_at: datetime,
        engagement_version: int | None = None,
    ) -> SocialIngestionBatch:
        known = _utc_datetime("known_at", known_at)
        ingested = _utc_datetime("ingested_at", ingested_at)
        if ingested < known:
            raise ValueError("ingested_at must not be before known_at")
        version = (
            self._timestamp_version(known)
            if engagement_version is None
            else self._validate_version(engagement_version)
        )
        after = self._extract_after(payload)

        raw_posts: list[RawSocialPost] = []
        observations: list[SocialObservation] = []
        snapshots: list[EngagementSnapshot] = []
        issues: list[SocialIngestionIssue] = []
        seen: dict[str, RawSocialPost] = {}

        for index, (kind_code, data) in enumerate(self._iter_children(payload)):
            fallback_id = f"payload-item-{index}"
            external_id = self._external_id(kind_code, data) or fallback_id
            if kind_code == "more":
                child_count = data.get("count")
                suffix = (
                    f" ({child_count} referenced comments)"
                    if isinstance(child_count, int) and not isinstance(child_count, bool)
                    else ""
                )
                issues.append(
                    SocialIngestionIssue(
                        external_id=external_id,
                        code="more_comments_unfetched",
                        message=(
                            "Reddit MoreComments requires a separate API expansion"
                            f"{suffix}"
                        ),
                    )
                )
                continue
            try:
                raw = self._raw_post(
                    kind_code,
                    data,
                    known_at=known,
                    ingested_at=ingested,
                )
            except (TypeError, ValueError, OverflowError, OSError) as exc:
                issues.append(
                    SocialIngestionIssue(
                        external_id=external_id,
                        code="invalid_or_removed_item",
                        message=str(exc) or exc.__class__.__name__,
                    )
                )
                continue

            previous = seen.get(raw.external_id)
            if previous is not None:
                code = "duplicate_item" if previous == raw else "conflicting_duplicate"
                issues.append(
                    SocialIngestionIssue(
                        external_id=raw.external_id,
                        code=code,
                        message="duplicate external ID retained only once",
                    )
                )
                continue
            seen[raw.external_id] = raw
            raw_posts.append(raw)

            resolved = self._entity_resolver.resolve(raw.text)
            if isinstance(resolved, str):
                raise TypeError("EntityResolver.resolve must return entity IDs, not a string")
            entity_ids = tuple(resolved)
            if not entity_ids:
                issues.append(
                    SocialIngestionIssue(
                        external_id=raw.external_id,
                        code="no_entity",
                        message="no canonical entity could be resolved",
                    )
                )
                continue
            sentiment = (
                None
                if self._sentiment_scorer is None
                else self._sentiment_scorer.score(raw.text)
            )
            observation = SocialObservation(
                source=raw.source,
                external_id=raw.external_id,
                entity_ids=entity_ids,
                authored_at=raw.published_at,
                known_at=raw.known_at,
                ingested_at=raw.ingested_at,
                text=raw.text,
                content_hash=_content_hash(raw.text),
                author_hash=_author_hash(raw.source, raw.author),
                stance=None if sentiment is None else sentiment.stance,
                stance_confidence=None if sentiment is None else sentiment.confidence,
                uri=raw.uri,
            )
            observations.append(observation)
            if raw.engagement.has_metrics:
                snapshots.append(
                    EngagementSnapshot(
                        source=raw.source,
                        external_id=raw.external_id,
                        version=version,
                        known_at=known,
                        ingested_at=ingested,
                        score=raw.engagement.score,
                        likes=raw.engagement.likes,
                        upvotes=raw.engagement.upvotes,
                        comments=raw.engagement.comments,
                        shares=raw.engagement.shares,
                        views=raw.engagement.views,
                    )
                )

        return SocialIngestionBatch(
            raw_posts=tuple(raw_posts),
            observations=tuple(observations),
            engagement_snapshots=tuple(snapshots),
            issues=tuple(issues),
            after=after,
        )

    def _raw_post(
        self,
        kind_code: str,
        data: Mapping[str, object],
        *,
        known_at: datetime,
        ingested_at: datetime,
    ) -> RawSocialPost:
        kind = _REDDIT_KINDS.get(kind_code)
        if kind is None:
            raise ValueError(f"unsupported Reddit kind {kind_code!r}")
        external_id = self._external_id(kind_code, data)
        if external_id is None:
            raise ValueError("Reddit item has no id/name")
        text = self._text(kind, data)
        if text is None:
            raise ValueError("Reddit item has only deleted, removed, or empty text")
        published_at = self._published_at(data.get("created_utc"))
        author_value = data.get("author")
        author = author_value if isinstance(author_value, str) else None
        permalink_value = data.get("permalink")
        uri = (
            urljoin(self._reddit_origin, permalink_value)
            if isinstance(permalink_value, str) and permalink_value.strip()
            else None
        )
        parent_value = data.get("parent_id")
        parent_id = parent_value.strip() if isinstance(parent_value, str) else None
        engagement = RawEngagement(
            score=_finite_optional_number("score", data.get("score")),
            upvotes=_nonnegative_optional_int("ups", data.get("ups")),
            comments=(
                _nonnegative_optional_int("num_comments", data.get("num_comments"))
                if kind == "post"
                else None
            ),
        )
        return RawSocialPost(
            source="reddit",
            external_id=external_id,
            kind=kind,
            published_at=published_at,
            # Reddit creation time is publication, not claim effective time.
            effective_at=None,
            known_at=known_at,
            ingested_at=ingested_at,
            text=text,
            author=author,
            uri=uri,
            parent_external_id=parent_id,
            engagement=engagement,
        )

    @staticmethod
    def _text(kind: str, data: Mapping[str, object]) -> str | None:
        fields = ("body",) if kind == "comment" else ("title", "selftext")
        parts: list[str] = []
        for name in fields:
            value = data.get(name)
            if not isinstance(value, str):
                continue
            cleaned = _normalise_space(value)
            if not cleaned or cleaned.casefold() in _DELETED_TEXT:
                continue
            parts.append(cleaned)
        return "\n\n".join(parts) if parts else None

    @staticmethod
    def _published_at(value: object) -> datetime:
        timestamp = _finite_optional_number("created_utc", value)
        if timestamp is None:
            raise ValueError("created_utc must be a finite Unix timestamp")
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)

    @staticmethod
    def _external_id(
        kind_code: str,
        data: Mapping[str, object],
    ) -> str | None:
        name = data.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
        identifier = data.get("id")
        if isinstance(identifier, str) and identifier.strip():
            return f"{kind_code}_{identifier.strip()}"
        return None

    @classmethod
    def _iter_children(
        cls,
        payload: object,
    ) -> Iterable[tuple[str, Mapping[str, object]]]:
        roots: tuple[object, ...]
        if isinstance(payload, list):
            roots = tuple(payload)
        elif isinstance(payload, Mapping):
            roots = (payload,)
        else:
            raise SocialPayloadError("Reddit payload must be a Listing object or list")
        for root in roots:
            yield from cls._walk_listing(root)

    @classmethod
    def _walk_listing(
        cls,
        listing: object,
    ) -> Iterable[tuple[str, Mapping[str, object]]]:
        if not isinstance(listing, Mapping):
            raise SocialPayloadError("Reddit Listing must be an object")
        listing_data = listing.get("data")
        if not isinstance(listing_data, Mapping):
            raise SocialPayloadError("Reddit Listing.data must be an object")
        children = listing_data.get("children")
        if not isinstance(children, list):
            raise SocialPayloadError("Reddit Listing.data.children must be a list")
        for child in children:
            if not isinstance(child, Mapping):
                continue
            kind = child.get("kind")
            data = child.get("data")
            if not isinstance(kind, str) or not isinstance(data, Mapping):
                continue
            if kind in _REDDIT_KINDS or kind == "more":
                yield kind, data
            replies = data.get("replies")
            if isinstance(replies, Mapping):
                yield from cls._walk_listing(replies)

    @staticmethod
    def _extract_after(payload: object) -> str | None:
        root = payload[0] if isinstance(payload, list) and payload else payload
        if not isinstance(root, Mapping):
            raise SocialPayloadError("Reddit payload must be a Listing object or list")
        data = root.get("data")
        if not isinstance(data, Mapping):
            raise SocialPayloadError("Reddit Listing.data must be an object")
        after = data.get("after")
        if after is None:
            return None
        if not isinstance(after, str) or not after.strip():
            raise SocialPayloadError("Reddit pagination cursor must be a string or null")
        return after.strip()

    @staticmethod
    def _timestamp_version(known_at: datetime) -> int:
        return max(1, int(known_at.timestamp() * 1_000_000))

    @staticmethod
    def _validate_version(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("engagement_version must be an integer or None")
        if value < 1:
            raise ValueError("engagement_version must be at least 1")
        return value


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    body: bytes
    headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.status, bool) or not isinstance(self.status, int):
            raise TypeError("status must be an integer")
        if not 100 <= self.status <= 599:
            raise ValueError("status must be a valid HTTP status")
        if not isinstance(self.body, bytes):
            raise TypeError("body must be bytes")
        if not isinstance(self.headers, Mapping):
            raise TypeError("headers must be a mapping")
        normalised = {
            _required_text("header name", str(name)).casefold(): str(value).strip()
            for name, value in self.headers.items()
        }
        object.__setattr__(self, "headers", MappingProxyType(normalised))


@runtime_checkable
class HttpTransport(Protocol):
    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
    ) -> HttpResponse:
        """Perform one GET and return the response without status coercion."""


class UrllibHttpTransport:
    """Small stdlib transport; errors remain explicit response/domain errors."""

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
    ) -> HttpResponse:
        request = Request(url, headers=dict(headers), method="GET")
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310
                return HttpResponse(
                    status=response.status,
                    body=response.read(),
                    headers=dict(response.headers.items()),
                )
        except HTTPError as exc:
            return HttpResponse(
                status=exc.code,
                body=exc.read(),
                headers=dict(exc.headers.items()) if exc.headers is not None else {},
            )
        except URLError as exc:
            raise SocialTransportError(f"Reddit transport failed: {exc.reason}") from exc


class SocialTransportError(RuntimeError):
    pass


class HttpStatusError(SocialTransportError):
    def __init__(self, *, status: int, url: str) -> None:
        self.status = status
        self.url = url
        super().__init__(f"Reddit returned HTTP {status} for {url}")


class RateLimitError(HttpStatusError):
    def __init__(
        self,
        *,
        status: int,
        url: str,
        retry_after: float | None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(status=status, url=url)


class RedditJsonCollector:
    """Paginated anonymous Reddit collector with optional thread comments.

    ``collect`` returns a staging batch.  Use ``SocialIngestionOrchestrator``
    for production ingestion so ``ingested_at`` is stamped by the store at the
    actual append boundary.
    """

    def __init__(
        self,
        *,
        subreddit: str,
        parser: SocialPayloadParser,
        user_agent: str,
        transport: HttpTransport | None = None,
        timeout: float = 10.0,
        page_size: int = 100,
        listing: str = "new",
        base_url: str = "https://www.reddit.com",
        clock: Callable[[], datetime] | None = None,
        include_comments: bool = False,
        max_threads_per_page: int = 10,
        comment_depth: int = 10,
    ) -> None:
        subreddit_value = _required_text("subreddit", subreddit)
        if not re.fullmatch(r"[A-Za-z0-9_]{1,64}", subreddit_value):
            raise ValueError("subreddit contains unsupported characters")
        if not isinstance(parser, SocialPayloadParser):
            raise TypeError("parser must implement SocialPayloadParser")
        user_agent_value = _required_text("user_agent", user_agent)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("timeout must be a number")
        timeout_value = float(timeout)
        if not math.isfinite(timeout_value) or timeout_value <= 0.0:
            raise ValueError("timeout must be finite and positive")
        if isinstance(page_size, bool) or not isinstance(page_size, int):
            raise TypeError("page_size must be an integer")
        if not 1 <= page_size <= 100:
            raise ValueError("page_size must be between 1 and 100")
        listing_value = _required_text("listing", listing).casefold()
        if listing_value not in {"new", "hot", "rising"}:
            raise ValueError("listing must be new, hot, or rising")
        transport_value = transport or UrllibHttpTransport()
        if not isinstance(transport_value, HttpTransport):
            raise TypeError("transport must implement HttpTransport")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        if not isinstance(include_comments, bool):
            raise TypeError("include_comments must be a boolean")
        for name, value, minimum, maximum in (
            ("max_threads_per_page", max_threads_per_page, 1, 100),
            ("comment_depth", comment_depth, 1, 20),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}")

        self._subreddit = subreddit_value
        self._parser = parser
        self._user_agent = user_agent_value
        self._transport = transport_value
        self._timeout = timeout_value
        self._page_size = page_size
        self._listing = listing_value
        self._base_url = _required_text("base_url", base_url).rstrip("/")
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._include_comments = include_comments
        self._max_threads_per_page = max_threads_per_page
        self._comment_depth = comment_depth

    def collect(
        self,
        *,
        max_pages: int = 1,
        start_after: str | None = None,
    ) -> SocialIngestionBatch:
        if isinstance(max_pages, bool) or not isinstance(max_pages, int):
            raise TypeError("max_pages must be an integer")
        if max_pages < 1:
            raise ValueError("max_pages must be at least 1")

        if start_after is not None:
            start_after = _required_text("start_after", start_after)
        batches: list[SocialIngestionBatch] = []
        after = start_after
        seen_cursors: set[str] = set(() if after is None else (after,))
        for _ in range(max_pages):
            url = self._url(after)
            page_batch = self._request_batch(url)
            batches.append(page_batch)
            if self._include_comments:
                posts = [item for item in page_batch.raw_posts if item.kind == "post"]
                for post_index, raw_post in enumerate(posts):
                    if post_index >= self._max_threads_per_page:
                        batches.append(
                            SocialIngestionBatch(
                                issues=(
                                    SocialIngestionIssue(
                                        external_id=raw_post.external_id,
                                        code="thread_fetch_limit",
                                        message=(
                                            "thread comments were not fetched because "
                                            "max_threads_per_page was reached"
                                        ),
                                    ),
                                )
                            )
                        )
                        continue
                    batches.append(
                        self._request_batch(self._thread_url(raw_post.external_id))
                    )
            after = page_batch.after
            if after is None:
                break
            if after in seen_cursors:
                raise SocialPayloadError("Reddit pagination cursor repeated")
            seen_cursors.add(after)

        return self._merge_batches(batches, after=after)

    def _request_batch(self, url: str) -> SocialIngestionBatch:
        response = self._transport.get(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": self._user_agent,
            },
            timeout=self._timeout,
        )
        self._raise_for_status(response, url)
        try:
            payload = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SocialPayloadError("Reddit response is not valid UTF-8 JSON") from exc

        known_at = _utc_datetime("clock known_at", self._clock())
        # Staging objects need a valid domain timestamp but are not considered
        # persisted.  SocialIngestionOrchestrator replaces this value atomically
        # at the store append boundary.
        return self._parser.parse(
            payload,
            known_at=known_at,
            ingested_at=known_at,
        )

    def _url(self, after: str | None) -> str:
        query: dict[str, object] = {
            "raw_json": 1,
            "limit": self._page_size,
        }
        if after is not None:
            query["after"] = after
        path = f"/r/{quote(self._subreddit, safe='')}/{self._listing}.json"
        return f"{self._base_url}{path}?{urlencode(query)}"

    def _thread_url(self, external_id: str) -> str:
        identifier = external_id[3:] if external_id.startswith("t3_") else external_id
        path = f"/comments/{quote(identifier, safe='')}.json"
        query = urlencode(
            {
                "raw_json": 1,
                "limit": 100,
                "depth": self._comment_depth,
                "sort": "new",
            }
        )
        return f"{self._base_url}{path}?{query}"

    @staticmethod
    def _raise_for_status(response: HttpResponse, url: str) -> None:
        if response.status == 429:
            raw_retry = response.headers.get("retry-after")
            retry_after: float | None = None
            if raw_retry is not None:
                try:
                    parsed = float(raw_retry)
                    retry_after = parsed if math.isfinite(parsed) and parsed >= 0.0 else None
                except ValueError:
                    retry_after = None
            raise RateLimitError(
                status=response.status,
                url=url,
                retry_after=retry_after,
            )
        if response.status >= 400:
            raise HttpStatusError(status=response.status, url=url)

    @staticmethod
    def _merge_batches(
        batches: Iterable[SocialIngestionBatch],
        *,
        after: str | None,
    ) -> SocialIngestionBatch:
        raw_by_id: dict[str, RawSocialPost] = {}
        observation_by_id: dict[str, SocialObservation] = {}
        snapshot_by_id: dict[str, EngagementSnapshot] = {}
        issues: list[SocialIngestionIssue] = []

        for batch in batches:
            issues.extend(batch.issues)
            for raw in batch.raw_posts:
                if raw.external_id in raw_by_id:
                    issues.append(
                        SocialIngestionIssue(
                            external_id=raw.external_id,
                            code="duplicate_across_pages",
                            message="pagination overlap retained only once",
                        )
                    )
                    continue
                raw_by_id[raw.external_id] = raw
            for observation in batch.observations:
                observation_by_id.setdefault(observation.evidence_id, observation)
            for snapshot in batch.engagement_snapshots:
                # Only the snapshot belonging to the first observation seen in
                # overlapping pages is retained in this collection batch.
                snapshot_by_id.setdefault(snapshot.observation_id, snapshot)

        observations = tuple(
            sorted(
                observation_by_id.values(),
                key=lambda item: (
                    item.authored_at,
                    item.source,
                    item.external_id,
                ),
            )
        )
        observation_ids = {item.evidence_id for item in observations}
        snapshots = tuple(
            sorted(
                (
                    item
                    for item in snapshot_by_id.values()
                    if item.observation_id in observation_ids
                ),
                key=lambda item: (item.source, item.external_id, item.version),
            )
        )
        return SocialIngestionBatch(
            raw_posts=tuple(
                sorted(
                    raw_by_id.values(),
                    key=lambda item: (
                        item.published_at,
                        item.source,
                        item.external_id,
                    ),
                )
            ),
            observations=observations,
            engagement_snapshots=snapshots,
            issues=tuple(issues),
            after=after,
        )


@dataclass(frozen=True, slots=True)
class PersistedSocialIngestion:
    """One collected batch and its atomic store-append result."""

    batch: SocialIngestionBatch
    store_result: SocialStoreAppendResult


class SocialIngestionOrchestrator:
    """Collect social data and commit it with truthful persistence time."""

    def __init__(
        self,
        *,
        collector: SocialCollector,
        store: InMemorySocialObservationStore,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(collector, SocialCollector):
            raise TypeError("collector must implement SocialCollector")
        if not isinstance(store, InMemorySocialObservationStore):
            raise TypeError("store must be an InMemorySocialObservationStore")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        self._collector = collector
        self._store = store
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def collect_and_persist(
        self,
        *,
        max_pages: int = 1,
        start_after: str | None = None,
    ) -> PersistedSocialIngestion:
        staged = self._collector.collect(
            max_pages=max_pages,
            start_after=start_after,
        )
        persisted_at = _utc_datetime("persistence clock", self._clock())
        store_result = self._store.append_batch(
            staged.observations,
            staged.engagement_snapshots,
            ingested_at=persisted_at,
        )
        persisted_batch = SocialIngestionBatch(
            raw_posts=tuple(
                replace(item, ingested_at=persisted_at) for item in staged.raw_posts
            ),
            observations=tuple(item for item, _created in store_result.observations),
            engagement_snapshots=tuple(
                item for item, _created in store_result.engagement_snapshots
            ),
            issues=staged.issues,
            after=staged.after,
        )
        return PersistedSocialIngestion(
            batch=persisted_batch,
            store_result=store_result,
        )


__all__ = [
    "EntityResolver",
    "HttpResponse",
    "HttpStatusError",
    "HttpTransport",
    "KeywordEntityResolver",
    "LexiconSentimentScorer",
    "PersistedSocialIngestion",
    "RateLimitError",
    "RawEngagement",
    "RawSocialPost",
    "RedditJsonCollector",
    "RedditJsonParser",
    "SentimentScore",
    "SentimentScorer",
    "SocialCollector",
    "SocialIngestionBatch",
    "SocialIngestionIssue",
    "SocialIngestionOrchestrator",
    "SocialPayloadError",
    "SocialPayloadParser",
    "SocialTransportError",
    "UrllibHttpTransport",
]

"""Retrieve auditable company facts and historical events for one case."""

from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Protocol

from src.domain import CanonicalEvent, CaseRef, ContextBundle, ContextFact
from src.graph.world_graph import InMemoryWorldGraph


class EventMemoryReader(Protocol):
    """Structural subset of EventMemory required by context retrieval."""

    def search(
        self,
        *,
        cutoff: datetime,
        event_type: str | None = None,
        subject_ids: Sequence[str] = (),
        object_ids: Sequence[str] = (),
        occurred_after: datetime | None = None,
        occurred_before: datetime | None = None,
        limit: int | None = None,
    ) -> Sequence[CanonicalEvent]: ...


_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_ECONOMIC_PREDICATES = frozenset(
    {
        "competeswith",
        "consumes",
        "costshare",
        "customerof",
        "derivedfrom",
        "dependson",
        "exposedto",
        "fuelcostshare",
        "hedgecoverage",
        "hedgedby",
        "inputcostshare",
        "operatingcostshare",
        "produces",
        "revenueexposure",
        "sellsto",
        "supplies",
        "supplierof",
    }
)
_GENERIC_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "change",
        "company",
        "commodity",
        "down",
        "event",
        "for",
        "in",
        "of",
        "percent",
        "price",
        "report",
        "the",
        "to",
        "up",
    }
)


class ContextRetriever:
    """Build a temporally safe, deterministic context bundle.

    Ranking combines graph distance from the case, exact event-entity links,
    economic relationship types, lexical overlap, and confidence.  It does not
    attempt to infer impact direction; that remains the economy model's job.
    """

    def __init__(
        self,
        world_graph: InMemoryWorldGraph,
        event_memory: EventMemoryReader | None = None,
        *,
        fact_limit: int = 8,
        historical_event_limit: int = 4,
        graph_hops: int = 2,
    ) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (fact_limit, historical_event_limit, graph_hops)
        ):
            raise TypeError("retrieval limits and graph_hops must be integers")
        if fact_limit < 0:
            raise ValueError("fact_limit must be non-negative")
        if historical_event_limit < 0:
            raise ValueError("historical_event_limit must be non-negative")
        if graph_hops < 1:
            raise ValueError("graph_hops must be at least 1")
        self._world_graph = world_graph
        self._event_memory = event_memory
        self._fact_limit = fact_limit
        self._historical_event_limit = historical_event_limit
        self._graph_hops = graph_hops

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
    ) -> ContextBundle:
        """Return the best available context for an event/case combination."""

        resolved_fact_limit = self._fact_limit if fact_limit is None else fact_limit
        resolved_event_limit = (
            self._historical_event_limit
            if historical_event_limit is None
            else historical_event_limit
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (resolved_fact_limit, resolved_event_limit)
        ):
            raise TypeError("retrieval limits must be integers")
        if resolved_fact_limit < 0:
            raise ValueError("fact_limit must be non-negative")
        if resolved_event_limit < 0:
            raise ValueError("historical_event_limit must be non-negative")
        if not isinstance(horizon, str) or not horizon.strip():
            raise ValueError("horizon must be a non-empty string")

        safe_event = event.as_observed_at(cutoff)
        if safe_event is None:
            raise ValueError("event was not yet available at cutoff")

        # Facts describe the world at the event time.  Availability is checked
        # separately against cutoff, so facts learned later cannot leak in.
        valid_at = min(safe_event.occurred_at, cutoff)
        neighborhood = self._world_graph.neighborhood_with_distance(
            (case.entity_id,),
            cutoff=cutoff,
            valid_at=valid_at,
            max_hops=self._graph_hops,
        )

        peer_safe_neighborhood = tuple(
            item
            for item in neighborhood
            if not self._touches_peer_case(item[0], case)
        )
        facts, has_case_event_link = self._select_linked_facts(
            peer_safe_neighborhood,
            event=safe_event,
            case=case,
            limit=resolved_fact_limit,
        )

        related_entities = {case.entity_id}
        for fact in facts:
            related_entities.add(fact.subject_id)
            if fact.object_id is not None:
                related_entities.add(fact.object_id)

        historical_events = (
            self._retrieve_historical_events(
                event=safe_event,
                case=case,
                cutoff=cutoff,
                related_entities=frozenset(related_entities),
                limit=resolved_event_limit,
            )
            if has_case_event_link
            else ()
        )

        return ContextBundle(
            event=safe_event,
            case=case,
            cutoff=cutoff,
            retrieved_at=retrieved_at or max(datetime.now(timezone.utc), cutoff),
            horizon=horizon.strip(),
            facts=facts,
            historical_events=historical_events,
        )

    def _retrieve_historical_events(
        self,
        *,
        event: CanonicalEvent,
        case: CaseRef,
        cutoff: datetime,
        related_entities: frozenset[str],
        limit: int,
    ) -> tuple[CanonicalEvent, ...]:
        if self._event_memory is None or limit == 0:
            return ()

        candidates = self._event_memory.search(
            cutoff=cutoff,
            occurred_before=event.occurred_at,
            # V1 uses an in-memory store, so rank every eligible event.  A
            # pre-ranking cap could otherwise miss the true deterministic top-k.
            limit=None,
        )
        time_safe = (
            candidate
            for candidate in candidates
            if candidate.event_id != event.event_id
            and candidate.observed_at <= cutoff
            and candidate.occurred_at < event.occurred_at
        )
        eligible = (
            candidate
            for candidate in time_safe
            if self._is_relevant_historical_event(
                candidate=candidate,
                event=event,
                case=case,
                related_entities=related_entities,
            )
        )
        ranked = sorted(
            eligible,
            key=lambda candidate: self._historical_event_order_key(
                candidate=candidate,
                event=event,
                case=case,
                related_entities=related_entities,
            ),
        )

        return tuple(ranked[:limit])

    @classmethod
    def _select_linked_facts(
        cls,
        neighborhood: Sequence[tuple[ContextFact, int]],
        *,
        event: CanonicalEvent,
        case: CaseRef,
        limit: int,
    ) -> tuple[tuple[ContextFact, ...], bool]:
        """Select complete case-to-event paths plus economic case qualifiers."""

        event_entities = frozenset((*event.subject_ids, *event.object_ids))
        direct_event_link = case.entity_id in event_entities
        paths = cls._entity_paths(
            neighborhood,
            start=case.entity_id,
            targets=event_entities,
        )
        anchor_paths = tuple(
            sorted(
                paths,
                key=lambda path: (len(path), tuple(fact.fact_id for fact in path)),
            )
        )

        if limit == 0:
            return (), direct_event_link

        selected: list[ContextFact] = []
        selected_ids: set[str] = set()
        has_complete_anchor = direct_event_link

        for path in anchor_paths:
            additions = [fact for fact in path if fact.fact_id not in selected_ids]
            if len(selected) + len(additions) > limit:
                continue
            selected.extend(additions)
            selected_ids.update(fact.fact_id for fact in additions)
            has_complete_anchor = True

        if not has_complete_anchor:
            return (), False

        qualifiers = (
            (fact, distance)
            for fact, distance in neighborhood
            if fact.fact_id not in selected_ids
            and cls._touches_case(fact, case)
            and cls._relation_key(fact.predicate) in _ECONOMIC_PREDICATES
        )
        for fact, _distance in sorted(
            qualifiers,
            key=lambda item: cls._fact_order_key(
                fact=item[0],
                distance=item[1],
                event=event,
                case=case,
            ),
        ):
            if len(selected) >= limit:
                break
            selected.append(fact)
            selected_ids.add(fact.fact_id)

        # Preserve path order for auditability, then deterministic qualifier rank.
        return tuple(selected), True

    @classmethod
    def _entity_paths(
        cls,
        neighborhood: Sequence[tuple[ContextFact, int]],
        *,
        start: str,
        targets: frozenset[str],
    ) -> tuple[tuple[ContextFact, ...], ...]:
        if not targets or start in targets:
            return ()

        adjacency: dict[str, list[tuple[str, ContextFact]]] = defaultdict(list)
        for fact, _distance in neighborhood:
            if fact.object_id is None:
                continue
            adjacency[fact.subject_id].append((fact.object_id, fact))
            adjacency[fact.object_id].append((fact.subject_id, fact))
        for edges in adjacency.values():
            edges.sort(key=lambda item: (item[1].fact_id, item[0]))

        queue = deque([(start, (), frozenset((start,)))])
        best_depth: dict[str, int] = {start: 0}
        found: dict[str, tuple[ContextFact, ...]] = {}
        while queue:
            entity, path, visited = queue.popleft()
            for other, fact in adjacency.get(entity, ()):
                if other in visited:
                    continue
                next_path = (*path, fact)
                if other in targets:
                    previous = found.get(other)
                    if previous is None or (
                        len(next_path), tuple(item.fact_id for item in next_path)
                    ) < (len(previous), tuple(item.fact_id for item in previous)):
                        found[other] = next_path
                    continue
                depth = len(next_path)
                if depth <= best_depth.get(other, depth):
                    best_depth[other] = depth
                    queue.append((other, next_path, visited | {other}))
        return tuple(found[target] for target in sorted(found))

    @classmethod
    def _is_relevant_historical_event(
        cls,
        *,
        candidate: CanonicalEvent,
        event: CanonicalEvent,
        case: CaseRef,
        related_entities: frozenset[str],
    ) -> bool:
        current_entities = frozenset((*event.subject_ids, *event.object_ids))
        candidate_entities = frozenset((*candidate.subject_ids, *candidate.object_ids))
        if current_entities.intersection(candidate_entities):
            return True
        if candidate.event_type.casefold() == event.event_type.casefold():
            return True

        shared_related = related_entities.intersection(candidate_entities)
        if not shared_related:
            return False
        if any(entity != case.entity_id for entity in shared_related):
            return True

        # A case-only match needs substantive topical overlap; an arbitrary old
        # event of the same company is not an analogue for the current event.
        context_tokens = cls._meaningful_tokens(
            cls._tokens((event.summary, *(related_entities - {case.entity_id})))
        )
        candidate_tokens = cls._meaningful_tokens(cls._tokens((candidate.summary,)))
        return bool(context_tokens.intersection(candidate_tokens))

    @staticmethod
    def _touches_case(fact: ContextFact, case: CaseRef) -> bool:
        return fact.subject_id == case.entity_id or fact.object_id == case.entity_id

    @staticmethod
    def _touches_peer_case(fact: ContextFact, case: CaseRef) -> bool:
        if case.case_type.casefold() != "company" or ":" not in case.entity_id:
            return False
        namespace = case.entity_id.split(":", 1)[0] + ":"
        endpoints = (fact.subject_id, fact.object_id)
        return any(
            endpoint is not None
            and endpoint != case.entity_id
            and endpoint.startswith(namespace)
            for endpoint in endpoints
        )

    @classmethod
    def _fact_order_key(
        cls,
        *,
        fact: ContextFact,
        distance: int,
        event: CanonicalEvent,
        case: CaseRef,
    ) -> tuple[float, int, float, str]:
        score = cls._fact_score(fact=fact, distance=distance, event=event, case=case)
        return (-score, distance, -fact.observed_at.timestamp(), fact.fact_id)

    @classmethod
    def _fact_score(
        cls,
        *,
        fact: ContextFact,
        distance: int,
        event: CanonicalEvent,
        case: CaseRef,
    ) -> float:
        event_entities = frozenset((*event.subject_ids, *event.object_ids))
        fact_entities = {fact.subject_id}
        if fact.object_id is not None:
            fact_entities.add(fact.object_id)

        score = max(0.0, 5.0 - 1.5 * (distance - 1))
        if fact.subject_id == case.entity_id:
            score += 6.0
        if fact.object_id == case.entity_id:
            score += 4.0

        event_entity_matches = fact_entities.intersection(event_entities)
        score += 8.0 * len(event_entity_matches)
        if (
            fact.subject_id == case.entity_id
            and fact.object_id is not None
            and fact.object_id in event_entities
        ):
            score += 7.0
        if fact.object_id == case.entity_id and fact.subject_id in event_entities:
            score += 6.0

        predicate_key = cls._relation_key(fact.predicate)
        if predicate_key in _ECONOMIC_PREDICATES:
            score += 2.0

        event_tokens = cls._event_tokens(event)
        fact_tokens = cls._fact_tokens(fact)
        score += 5.0 * cls._overlap(event_tokens, fact_tokens)
        score += 0.25 * fact.confidence
        return score

    @classmethod
    def _historical_event_order_key(
        cls,
        *,
        candidate: CanonicalEvent,
        event: CanonicalEvent,
        case: CaseRef,
        related_entities: frozenset[str],
    ) -> tuple[float, float, str]:
        score = cls._historical_event_score(
            candidate=candidate,
            event=event,
            case=case,
            related_entities=related_entities,
        )
        return (-score, -candidate.occurred_at.timestamp(), candidate.event_id)

    @classmethod
    def _historical_event_score(
        cls,
        *,
        candidate: CanonicalEvent,
        event: CanonicalEvent,
        case: CaseRef,
        related_entities: frozenset[str],
    ) -> float:
        current_entities = frozenset((*event.subject_ids, *event.object_ids))
        candidate_entities = frozenset((*candidate.subject_ids, *candidate.object_ids))

        score = 0.0
        if candidate.event_type.casefold() == event.event_type.casefold():
            score += 6.0
        score += 8.0 * len(current_entities.intersection(candidate_entities))
        score += 3.0 * len(related_entities.intersection(candidate_entities))
        if case.entity_id in candidate_entities:
            score += 5.0
        if candidate.direction is not None and candidate.direction == event.direction:
            score += 1.0
        if candidate.unit is not None and candidate.unit == event.unit:
            score += 0.5
        score += 5.0 * cls._overlap(cls._event_tokens(event), cls._event_tokens(candidate))
        if score == 0.0:
            return 0.0
        return score + 0.25 * candidate.confidence

    @classmethod
    def _event_tokens(cls, event: CanonicalEvent) -> frozenset[str]:
        values: list[object] = [
            event.event_type,
            event.summary,
            event.direction,
            event.unit,
            *event.subject_ids,
            *event.object_ids,
        ]
        return cls._tokens(values)

    @classmethod
    def _fact_tokens(cls, fact: ContextFact) -> frozenset[str]:
        return cls._tokens(
            (
                fact.subject_id,
                fact.predicate,
                fact.object_id,
                fact.value,
                fact.unit,
            )
        )

    @classmethod
    def _tokens(cls, values: Sequence[object | None]) -> frozenset[str]:
        tokens: set[str] = set()
        for value in values:
            if value is None:
                continue
            normalized = str(value).replace("_", " ").replace("-", " ").casefold()
            tokens.update(_TOKEN_RE.findall(normalized))
        return frozenset(tokens)

    @staticmethod
    def _meaningful_tokens(tokens: frozenset[str]) -> frozenset[str]:
        return frozenset(
            token
            for token in tokens
            if len(token) > 1 and token not in _GENERIC_TOKENS
        )

    @staticmethod
    def _relation_key(predicate: str) -> str:
        return "".join(character for character in predicate.casefold() if character.isalnum())

    @staticmethod
    def _overlap(left: frozenset[str], right: frozenset[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left.intersection(right)) / math.sqrt(len(left) * len(right))

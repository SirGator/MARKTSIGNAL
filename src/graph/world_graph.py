"""A small, deterministic in-memory world graph.

The graph deliberately stores :class:`~src.domain.ContextFact` objects without
rewriting them.  This keeps validity information and source provenance attached
to every fact that is later handed to the economy model.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from threading import RLock

from src.domain import ContextFact


class InMemoryWorldGraph:
    """Store typed context facts and traverse their entity relationships.

    Facts are immutable and keyed by ``fact_id``.  Re-adding the same fact is
    idempotent; reusing an id for different content raises ``ValueError``.  A
    changed real-world assertion therefore needs a new fact/revision id, which
    preserves older knowledge for point-in-time retrieval.  All read methods
    return tuples in a stable order.
    """

    def __init__(self, facts: Iterable[ContextFact] = ()) -> None:
        self._facts: dict[str, ContextFact] = {}
        self._by_subject: dict[str, set[str]] = defaultdict(set)
        self._by_object: dict[str, set[str]] = defaultdict(set)
        self._by_predicate: dict[str, set[str]] = defaultdict(set)
        self._lock = RLock()
        self.add_facts(facts)

    def __len__(self) -> int:
        with self._lock:
            return len(self._facts)

    def add_fact(self, fact: ContextFact) -> None:
        """Insert a fact idempotently while retaining complete provenance."""

        if not isinstance(fact, ContextFact):
            raise TypeError("fact must be a ContextFact")

        with self._lock:
            previous = self._facts.get(fact.fact_id)
            if previous is not None:
                if previous == fact:
                    return
                raise ValueError(
                    f"fact_id {fact.fact_id!r} already refers to a different fact; "
                    "use a new fact_id for a revision"
                )

            self._facts[fact.fact_id] = fact
            self._add_to_indexes(fact)

    # Convenience alias matching EventMemory.add; it has the same immutable,
    # idempotent semantics as add_fact (it is deliberately not an upsert).
    add = add_fact

    def add_facts(self, facts: Iterable[ContextFact]) -> None:
        for fact in facts:
            self.add_fact(fact)

    def get_fact(self, fact_id: str) -> ContextFact | None:
        with self._lock:
            return self._facts.get(fact_id)

    def all_facts(
        self,
        *,
        cutoff: datetime,
        valid_at: datetime | None = None,
    ) -> tuple[ContextFact, ...]:
        """Return every fact known by ``cutoff`` and valid at ``valid_at``."""

        with self._lock:
            candidates = tuple(self._facts.values())
        return self._visible_sorted(candidates, cutoff=cutoff, valid_at=valid_at)

    def search(
        self,
        *,
        cutoff: datetime,
        valid_at: datetime | None = None,
        subject_ids: Iterable[str] = (),
        object_ids: Iterable[str] = (),
        predicates: Iterable[str] = (),
        entity_ids: Iterable[str] = (),
        limit: int | None = None,
    ) -> tuple[ContextFact, ...]:
        """Filter visible facts using deterministic AND semantics.

        Values inside one filter are ORed.  Different non-empty filters are
        ANDed.  ``entity_ids`` matches either end of a fact and is useful when
        the edge direction is not known by the caller.
        """

        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise TypeError("limit must be an integer or None")
            if limit < 0:
                raise ValueError("limit must be non-negative or None")

        subjects = frozenset(subject_ids)
        objects = frozenset(object_ids)
        predicate_keys = frozenset(self._predicate_key(item) for item in predicates)
        entities = frozenset(entity_ids)

        with self._lock:
            candidate_ids: set[str] | None = None
            indexes: list[set[str]] = []

            if subjects:
                indexes.append(set().union(*(self._by_subject.get(item, set()) for item in subjects)))
            if objects:
                indexes.append(set().union(*(self._by_object.get(item, set()) for item in objects)))
            if predicate_keys:
                indexes.append(
                    set().union(*(self._by_predicate.get(item, set()) for item in predicate_keys))
                )
            if entities:
                entity_matches: set[str] = set()
                for entity_id in entities:
                    entity_matches.update(self._by_subject.get(entity_id, set()))
                    entity_matches.update(self._by_object.get(entity_id, set()))
                indexes.append(entity_matches)

            for index_matches in indexes:
                candidate_ids = (
                    index_matches
                    if candidate_ids is None
                    else candidate_ids.intersection(index_matches)
                )

            if candidate_ids is None:
                candidates = tuple(self._facts.values())
            else:
                candidates = tuple(self._facts[fact_id] for fact_id in candidate_ids)

        visible = self._visible_sorted(candidates, cutoff=cutoff, valid_at=valid_at)
        return visible if limit is None else visible[:limit]

    def neighborhood(
        self,
        entity_ids: Iterable[str],
        *,
        cutoff: datetime,
        valid_at: datetime | None = None,
        max_hops: int = 1,
    ) -> tuple[ContextFact, ...]:
        """Return the visible fact neighborhood around the supplied entities."""

        return tuple(
            fact
            for fact, _distance in self.neighborhood_with_distance(
                entity_ids,
                cutoff=cutoff,
                valid_at=valid_at,
                max_hops=max_hops,
            )
        )

    def neighborhood_with_distance(
        self,
        entity_ids: Iterable[str],
        *,
        cutoff: datetime,
        valid_at: datetime | None = None,
        max_hops: int = 1,
    ) -> tuple[tuple[ContextFact, int], ...]:
        """Breadth-first traversal returning each fact and its minimum hop.

        Invisible or no-longer-valid facts cannot be used as traversal edges,
        preventing future knowledge from leaking into an otherwise valid fact.
        """

        if isinstance(max_hops, bool) or not isinstance(max_hops, int):
            raise TypeError("max_hops must be an integer")
        if max_hops < 0:
            raise ValueError("max_hops must be non-negative")
        if max_hops == 0:
            return ()

        frontier = set(entity_ids)
        visited_entities = set(frontier)
        seen_fact_ids: set[str] = set()
        found: list[tuple[ContextFact, int]] = []

        for distance in range(1, max_hops + 1):
            if not frontier:
                break

            with self._lock:
                candidate_ids: set[str] = set()
                for entity_id in frontier:
                    candidate_ids.update(self._by_subject.get(entity_id, set()))
                    candidate_ids.update(self._by_object.get(entity_id, set()))
                candidates = tuple(
                    self._facts[fact_id]
                    for fact_id in sorted(candidate_ids)
                    if fact_id not in seen_fact_ids
                )

            visible = self._visible_sorted(candidates, cutoff=cutoff, valid_at=valid_at)
            next_frontier: set[str] = set()
            for fact in visible:
                seen_fact_ids.add(fact.fact_id)
                found.append((fact, distance))
                if fact.subject_id not in visited_entities:
                    next_frontier.add(fact.subject_id)
                if fact.object_id is not None and fact.object_id not in visited_entities:
                    next_frontier.add(fact.object_id)

            visited_entities.update(next_frontier)
            frontier = next_frontier

        return tuple(sorted(found, key=lambda item: (item[1], self._fact_sort_key(item[0]))))

    def _add_to_indexes(self, fact: ContextFact) -> None:
        self._by_subject[fact.subject_id].add(fact.fact_id)
        if fact.object_id is not None:
            self._by_object[fact.object_id].add(fact.fact_id)
        self._by_predicate[self._predicate_key(fact.predicate)].add(fact.fact_id)

    @staticmethod
    def _predicate_key(predicate: str) -> str:
        return predicate.strip().casefold()

    @classmethod
    def _visible_sorted(
        cls,
        facts: Iterable[ContextFact],
        *,
        cutoff: datetime,
        valid_at: datetime | None,
    ) -> tuple[ContextFact, ...]:
        visible = (
            snapshot
            for fact in facts
            if (snapshot := cls._as_available_at(fact, cutoff)) is not None
            and (valid_at is None or snapshot.is_valid_at(valid_at))
        )
        return tuple(sorted(visible, key=cls._fact_sort_key))

    @staticmethod
    def _as_available_at(fact: ContextFact, cutoff: datetime) -> ContextFact | None:
        """Return a snapshot that cannot expose later source provenance."""

        return fact.as_observed_at(cutoff)

    @staticmethod
    def _fact_sort_key(fact: ContextFact) -> tuple[float, str]:
        # Newer observations are preferred; id makes ties stable across runs.
        return (-fact.observed_at.timestamp(), fact.fact_id)

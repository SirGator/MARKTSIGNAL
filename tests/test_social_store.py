from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import unittest

from src.sources.social import (
    EngagementSnapshot,
    InMemorySocialObservationStore,
    SocialObservation,
)


UTC = timezone.utc
BASE = datetime(2026, 1, 1, 12, tzinfo=UTC)


def observation(
    external_id: str,
    *,
    source: str = "reddit",
    entity_ids: tuple[str, ...] = ("company:x",),
    authored_minute: int = 0,
    known_minute: int | None = None,
    ingested_minute: int | None = None,
    text: str | None = None,
) -> SocialObservation:
    known = authored_minute + 1 if known_minute is None else known_minute
    ingested = known + 1 if ingested_minute is None else ingested_minute
    return SocialObservation(
        source=source,
        external_id=external_id,
        entity_ids=entity_ids,
        authored_at=BASE + timedelta(minutes=authored_minute),
        known_at=BASE + timedelta(minutes=known),
        ingested_at=BASE + timedelta(minutes=ingested),
        text=text or f"Post {external_id}",
        content_hash=f"hash-{external_id}",
        author_hash=f"author-{external_id}",
        stance=0.5,
        stance_confidence=0.8,
        uri=f"https://example.invalid/{external_id}",
    )


def engagement(
    version: int,
    *,
    known_minute: int,
    ingested_minute: int,
    likes: int,
) -> EngagementSnapshot:
    return EngagementSnapshot(
        source="reddit",
        external_id="post-1",
        version=version,
        known_at=BASE + timedelta(minutes=known_minute),
        ingested_at=BASE + timedelta(minutes=ingested_minute),
        likes=likes,
        comments=version,
    )


class SocialObservationStoreTests(unittest.TestCase):
    def test_append_is_idempotent_and_edits_become_revisions(self) -> None:
        store = InMemorySocialObservationStore()
        item = observation("post-1")

        stored, created = store.add_observation(item)
        replayed, replay_created = store.add_observation(item)
        later_retry, retry_created = store.add_observation(
            replace(item, ingested_at=item.ingested_at + timedelta(minutes=5))
        )

        self.assertTrue(created)
        self.assertFalse(replay_created)
        self.assertFalse(retry_created)
        self.assertIs(stored, replayed)
        self.assertIs(stored, later_retry)
        self.assertEqual(len(store), 1)
        self.assertEqual(later_retry.ingested_at, item.ingested_at)

        edited, edit_created = store.add_observation(
            replace(
                item,
                known_at=item.known_at + timedelta(minutes=10),
                ingested_at=item.ingested_at + timedelta(minutes=10),
                text="Edited source content",
                content_hash="hash-post-1-edited",
            )
        )
        self.assertTrue(edit_created)
        self.assertEqual(edited.revision, 2)
        self.assertEqual(edited.evidence_id, "reddit:post-1:revision:2")
        self.assertEqual(len(store), 2)
        self.assertEqual(
            store.get_observation(
                "reddit",
                "post-1",
                cutoff=item.ingested_at + timedelta(minutes=5),
            ),
            item,
        )
        self.assertEqual(store.get_observation("reddit", "post-1"), edited)
        self.assertEqual(store.revisions_for("reddit", "post-1"), (item, edited))

    def test_atomic_batch_stamps_actual_persistence_time(self) -> None:
        staged = observation("post-1", ingested_minute=2)
        staged_snapshot = engagement(
            1,
            known_minute=3,
            ingested_minute=4,
            likes=10,
        )
        store = InMemorySocialObservationStore()
        persisted_at = BASE + timedelta(minutes=20)

        result = store.append_batch(
            (staged,),
            (staged_snapshot,),
            ingested_at=persisted_at,
        )

        persisted_observation = result.observations[0][0]
        persisted_snapshot = result.engagement_snapshots[0][0]
        self.assertEqual(result.ingested_at, persisted_at)
        self.assertEqual(persisted_observation.ingested_at, persisted_at)
        self.assertEqual(persisted_snapshot.ingested_at, persisted_at)
        self.assertIsNone(
            store.get_observation(
                "reddit",
                "post-1",
                cutoff=persisted_at - timedelta(microseconds=1),
            )
        )

    def test_cutoff_requires_both_known_and_ingested_availability(self) -> None:
        early = observation("early", authored_minute=0, ingested_minute=2)
        backfilled_late = observation(
            "late-backfill",
            authored_minute=0,
            known_minute=1,
            ingested_minute=20,
        )
        future_known = observation(
            "future-known",
            authored_minute=8,
            known_minute=15,
            ingested_minute=16,
        )
        store = InMemorySocialObservationStore(
            (future_known, backfilled_late, early)
        )

        visible = store.query_observations(
            cutoff=BASE + timedelta(minutes=10),
            entity_id="COMPANY:X",
        )

        self.assertEqual([item.external_id for item in visible], ["early"])
        self.assertIsNone(
            store.get_observation(
                "reddit",
                "late-backfill",
                cutoff=BASE + timedelta(minutes=10),
            )
        )
        # Uncut storage access still returns the complete append-only record.
        self.assertEqual(
            store.get_observation("reddit", "late-backfill"),
            backfilled_late,
        )

    def test_filters_boundaries_limit_and_order_are_deterministic(self) -> None:
        records = (
            observation("z", authored_minute=5, source="stocktwits"),
            observation("b", authored_minute=5),
            observation("a", authored_minute=5),
            observation(
                "other-entity",
                authored_minute=6,
                entity_ids=("company:y",),
            ),
            observation("left-boundary", authored_minute=0),
        )
        store = InMemorySocialObservationStore(reversed(records))

        visible = store.query_observations(
            cutoff=BASE + timedelta(minutes=30),
            entity_id="company:x",
            sources=("REDDIT",),
            authored_after=BASE,
            authored_at_or_before=BASE + timedelta(minutes=5),
            limit=2,
        )

        self.assertEqual([item.external_id for item in visible], ["a", "b"])

    def test_engagement_versions_are_append_only_and_cutoff_safe(self) -> None:
        post = observation("post-1", ingested_minute=2)
        store = InMemorySocialObservationStore((post,))
        version_one = engagement(1, known_minute=3, ingested_minute=4, likes=10)
        version_two = engagement(2, known_minute=8, ingested_minute=12, likes=25)

        store.add_engagement_snapshot(version_two)
        stored, created = store.add_engagement_snapshot(version_one)
        replayed, replay_created = store.add_engagement_snapshot(
            replace(version_one, ingested_at=BASE + timedelta(minutes=5))
        )

        self.assertTrue(created)
        self.assertFalse(replay_created)
        self.assertIs(stored, replayed)
        self.assertEqual(store.engagement_count, 2)
        at_ten = store.query_engagement_snapshots(
            source="REDDIT",
            external_id="post-1",
            cutoff=BASE + timedelta(minutes=10),
        )
        self.assertEqual([item.version for item in at_ten], [1])
        self.assertEqual(
            store.latest_engagement_snapshot(
                source="reddit",
                external_id="post-1",
                cutoff=BASE + timedelta(minutes=20),
            ),
            version_two,
        )

        with self.assertRaisesRegex(ValueError, "different immutable content"):
            store.add_engagement_snapshot(replace(version_one, likes=11))

    def test_engagement_requires_an_existing_observation(self) -> None:
        store = InMemorySocialObservationStore()
        with self.assertRaisesRegex(KeyError, "unknown observation"):
            store.add_engagement_snapshot(
                engagement(1, known_minute=3, ingested_minute=4, likes=1)
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
import json
from typing import Mapping
import unittest

from src.sources.social import (
    HttpResponse,
    HttpStatusError,
    KeywordEntityResolver,
    LexiconSentimentScorer,
    RateLimitError,
    RawEngagement,
    RawSocialPost,
    RedditJsonCollector,
    RedditJsonParser,
    SocialIngestionOrchestrator,
    InMemorySocialObservationStore,
    SocialPayloadError,
)


UTC = timezone.utc
BASE = datetime(2026, 1, 10, 12, tzinfo=UTC)


def epoch(at: datetime) -> float:
    return at.timestamp()


def listing(children: list[object], *, after: str | None = None) -> dict[str, object]:
    return {
        "kind": "Listing",
        "data": {
            "after": after,
            "children": children,
        },
    }


def post(
    identifier: str,
    text: str,
    *,
    created_at: datetime | None = None,
    author: str = "investor",
    selftext: str = "",
    score: object = 10,
    ups: object = 9,
    comments: object = 2,
    replies: object = "",
) -> dict[str, object]:
    return {
        "kind": "t3",
        "data": {
            "id": identifier,
            "name": f"t3_{identifier}",
            "title": text,
            "selftext": selftext,
            "created_utc": epoch(created_at or BASE - timedelta(minutes=10)),
            "author": author,
            "permalink": f"/r/markets/comments/{identifier}/example/",
            "score": score,
            "ups": ups,
            "num_comments": comments,
            "replies": replies,
        },
    }


def comment(
    identifier: str,
    text: str,
    *,
    created_at: datetime | None = None,
    author: str = "commenter",
    score: object = 3,
    ups: object = 3,
    replies: object = "",
) -> dict[str, object]:
    return {
        "kind": "t1",
        "data": {
            "id": identifier,
            "name": f"t1_{identifier}",
            "body": text,
            "created_utc": epoch(created_at or BASE - timedelta(minutes=5)),
            "author": author,
            "parent_id": "t3_parent",
            "permalink": f"/r/markets/comments/parent/example/{identifier}/",
            "score": score,
            "ups": ups,
            "replies": replies,
        },
    }


def parser() -> RedditJsonParser:
    return RedditJsonParser(
        entity_resolver=KeywordEntityResolver(
            {
                "company:apple": ("Apple", "AAPL", "$AAPL"),
                "company:tesla": ("Tesla", "TSLA", "$TSLA"),
            }
        ),
        sentiment_scorer=LexiconSentimentScorer(),
    )


class RawAndEnrichmentContractTests(unittest.TestCase):
    def test_raw_post_keeps_all_time_semantics_separate(self) -> None:
        published = BASE - timedelta(hours=1)
        effective = BASE - timedelta(days=2)
        raw = RawSocialPost(
            source="Reddit",
            external_id="t3_x",
            kind="POST",
            published_at=published,
            effective_at=effective,
            known_at=BASE,
            ingested_at=BASE + timedelta(seconds=1),
            text="Apple announces a product",
            author="alice",
            engagement=RawEngagement(score=-1, comments=4),
        )

        self.assertEqual(raw.source, "reddit")
        self.assertEqual(raw.published_at, published)
        self.assertEqual(raw.effective_at, effective)
        self.assertNotEqual(raw.effective_at, raw.published_at)
        with self.assertRaises(FrozenInstanceError):
            raw.text = "changed"  # type: ignore[misc]

    def test_keyword_resolution_respects_boundaries(self) -> None:
        resolver = KeywordEntityResolver(
            {
                "company:apple": ("Apple", "$AAPL"),
                "company:cat": ("CAT",),
            }
        )

        self.assertEqual(
            resolver.resolve("Buy $AAPL; Apple is strong"),
            ("company:apple",),
        )
        self.assertEqual(resolver.resolve("concatenate strings"), ())
        self.assertEqual(resolver.resolve("CAT rises"), ("company:cat",))

    def test_lexicon_sentiment_is_transparent_and_can_be_missing(self) -> None:
        scorer = LexiconSentimentScorer(
            positive=("buy", "strong"),
            negative=("sell", "weak"),
        )
        bullish = scorer.score("BUY because results are strong")
        balanced = scorer.score("buy or sell")

        self.assertIsNotNone(bullish)
        assert bullish is not None
        self.assertEqual(bullish.stance, 1.0)
        self.assertEqual(bullish.confidence, 0.5)
        self.assertIsNotNone(balanced)
        assert balanced is not None
        self.assertEqual(balanced.stance, 0.0)
        self.assertIsNone(scorer.score("No directional vocabulary here"))


class RedditJsonParserTests(unittest.TestCase):
    def test_posts_nested_comments_and_engagement_are_normalised(self) -> None:
        nested = listing(
            [
                comment(
                    "nested",
                    "Tesla is weak, sell before a crash",
                    author="[deleted]",
                    score=-2,
                    ups=0,
                )
            ]
        )
        payload = [
            listing(
                [
                    post(
                        "apple-post",
                        "Apple beats expectations",
                        selftext="AAPL reports strong profit growth",
                        score=42,
                        ups=41,
                        comments=3,
                    )
                ]
            ),
            listing(
                [
                    comment(
                        "tesla-comment",
                        "Tesla looks weak",
                        replies=nested,
                    )
                ]
            ),
        ]

        result = parser().parse(
            payload,
            known_at=BASE,
            ingested_at=BASE + timedelta(seconds=1),
            engagement_version=7,
        )

        self.assertEqual(
            [item.external_id for item in result.observations],
            ["t3_apple-post", "t1_tesla-comment", "t1_nested"],
        )
        apple = result.observations[0]
        self.assertEqual(apple.entity_ids, ("company:apple",))
        self.assertEqual(apple.authored_at, BASE - timedelta(minutes=10))
        self.assertEqual(apple.known_at, BASE)
        self.assertEqual(apple.ingested_at, BASE + timedelta(seconds=1))
        self.assertGreater(apple.stance or 0.0, 0.0)
        self.assertEqual(len(apple.content_hash), 64)
        self.assertEqual(len(apple.author_hash or ""), 64)
        self.assertEqual(
            apple.uri,
            "https://www.reddit.com/r/markets/comments/apple-post/example/",
        )

        nested_observation = result.observations[2]
        self.assertIsNone(nested_observation.author_hash)
        self.assertLess(nested_observation.stance or 0.0, 0.0)
        self.assertEqual(result.raw_posts[0].effective_at, None)
        self.assertEqual(result.raw_posts[1].parent_external_id, "t3_parent")

        snapshots = {item.external_id: item for item in result.engagement_snapshots}
        self.assertEqual(snapshots["t3_apple-post"].version, 7)
        self.assertEqual(snapshots["t3_apple-post"].score, 42.0)
        self.assertEqual(snapshots["t3_apple-post"].upvotes, 41)
        self.assertEqual(snapshots["t3_apple-post"].comments, 3)
        self.assertEqual(snapshots["t1_nested"].score, -2.0)
        self.assertIsNone(snapshots["t1_nested"].comments)
        self.assertEqual(result.issues, ())

    def test_removed_items_missing_entities_and_duplicate_ids_are_safe(self) -> None:
        same = post("duplicate", "Apple is strong")
        payload = listing(
            [
                same,
                same,
                comment("removed", "[removed]"),
                post("deleted", "[deleted]", selftext="[removed]"),
                post("no-entity", "General market discussion"),
            ]
        )

        result = parser().parse(
            payload,
            known_at=BASE,
            ingested_at=BASE,
            engagement_version=1,
        )

        self.assertEqual(
            [item.external_id for item in result.observations],
            ["t3_duplicate"],
        )
        # The unresolved raw post remains auditable, but cannot enter signals.
        self.assertEqual(
            [item.external_id for item in result.raw_posts],
            ["t3_duplicate", "t3_no-entity"],
        )
        self.assertEqual(
            [issue.code for issue in result.issues],
            (
                [
                    "duplicate_item",
                    "invalid_or_removed_item",
                    "invalid_or_removed_item",
                    "no_entity",
                ]
            ),
        )

    def test_same_content_with_different_ids_is_retained_for_duplicate_share(self) -> None:
        payload = listing(
            [
                post("copy-a", "  APPLE   is STRONG "),
                post("copy-b", "apple is strong"),
            ]
        )
        result = parser().parse(
            payload,
            known_at=BASE,
            ingested_at=BASE,
            engagement_version=1,
        )

        self.assertEqual(len(result.observations), 2)
        self.assertEqual(
            result.observations[0].content_hash,
            result.observations[1].content_hash,
        )
        self.assertNotEqual(
            result.observations[0].external_id,
            result.observations[1].external_id,
        )

    def test_future_publication_is_not_clamped_into_history(self) -> None:
        payload = listing(
            [post("future", "Apple is strong", created_at=BASE + timedelta(minutes=1))]
        )

        result = parser().parse(
            payload,
            known_at=BASE,
            ingested_at=BASE,
            engagement_version=1,
        )

        self.assertEqual(result.observations, ())
        self.assertEqual(result.raw_posts, ())
        self.assertEqual(result.issues[0].code, "invalid_or_removed_item")
        self.assertIn("known_at", result.issues[0].message)
        with self.assertRaisesRegex(ValueError, "ingested_at"):
            parser().parse(
                listing([]),
                known_at=BASE,
                ingested_at=BASE - timedelta(seconds=1),
            )

    def test_bad_optional_engagement_does_not_drop_a_valid_post(self) -> None:
        payload = listing(
            [
                post(
                    "bad-metrics",
                    "Apple is strong",
                    score=float("nan"),
                    ups=-1,
                    comments="ten",
                )
            ]
        )
        result = parser().parse(
            payload,
            known_at=BASE,
            ingested_at=BASE,
            engagement_version=1,
        )

        self.assertEqual(len(result.observations), 1)
        self.assertEqual(result.engagement_snapshots, ())


class FakeTransport:
    def __init__(self, responses: list[HttpResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, str], float]] = []

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: float,
    ) -> HttpResponse:
        self.calls.append((url, dict(headers), timeout))
        if not self._responses:
            raise AssertionError("unexpected real/additional HTTP request")
        return self._responses.pop(0)


class FixedClock:
    def __init__(self, values: list[datetime]) -> None:
        self._values = iter(values)

    def __call__(self) -> datetime:
        return next(self._values)


def response(payload: object, *, status: int = 200) -> HttpResponse:
    return HttpResponse(
        status=status,
        body=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
    )


class RedditJsonCollectorTests(unittest.TestCase):
    def test_pagination_user_agent_timeout_and_clock_are_injected(self) -> None:
        first_payload = listing(
            [post("first", "Apple is strong")],
            after="t3_next cursor",
        )
        second_payload = listing(
            [post("second", "Tesla is weak")],
            after=None,
        )
        transport = FakeTransport([response(first_payload), response(second_payload)])
        times = [
            BASE,
            BASE + timedelta(seconds=1),
            BASE + timedelta(seconds=2),
            BASE + timedelta(seconds=3),
        ]
        collector = RedditJsonCollector(
            subreddit="wallstreetbetsGER",
            parser=parser(),
            user_agent="market-context-research/0.1 contact@example.invalid",
            transport=transport,
            timeout=4.5,
            page_size=25,
            clock=FixedClock(times),
        )

        result = collector.collect(max_pages=3)

        self.assertEqual(len(transport.calls), 2)
        first_url, first_headers, first_timeout = transport.calls[0]
        second_url = transport.calls[1][0]
        self.assertIn("/r/wallstreetbetsGER/new.json?", first_url)
        self.assertIn("limit=25", first_url)
        self.assertNotIn("after=", first_url)
        self.assertIn("after=t3_next+cursor", second_url)
        self.assertEqual(
            first_headers["User-Agent"],
            "market-context-research/0.1 contact@example.invalid",
        )
        self.assertEqual(first_timeout, 4.5)
        self.assertEqual(
            [item.external_id for item in result.observations],
            ["t3_first", "t3_second"],
        )
        self.assertEqual(result.observations[0].known_at, BASE)
        self.assertEqual(result.observations[1].known_at, BASE + timedelta(seconds=1))
        self.assertIsNone(result.after)

    def test_start_after_resumes_from_a_checkpoint(self) -> None:
        transport = FakeTransport(
            [response(listing([post("resumed", "Apple is strong")], after="t3_next"))]
        )
        collector = RedditJsonCollector(
            subreddit="markets",
            parser=parser(),
            user_agent="test-agent",
            transport=transport,
            clock=FixedClock([BASE]),
        )

        result = collector.collect(max_pages=1, start_after="t3_checkpoint")

        self.assertIn("after=t3_checkpoint", transport.calls[0][0])
        self.assertEqual(result.after, "t3_next")
        self.assertEqual(result.observations[0].external_id, "t3_resumed")

    def test_optional_thread_fetch_collects_nested_comments_and_reports_more(self) -> None:
        nested = listing([comment("nested", "Tesla is weak")])
        more = {
            "kind": "more",
            "data": {
                "id": "remaining",
                "name": "more_remaining",
                "count": 17,
                "children": ["a", "b"],
            },
        }
        thread_payload = [
            listing([post("thread", "Apple is strong")]),
            listing(
                [
                    comment(
                        "top-comment",
                        "Apple is weak",
                        replies=nested,
                    ),
                    more,
                ]
            ),
        ]
        transport = FakeTransport(
            [
                response(listing([post("thread", "Apple is strong")], after=None)),
                response(thread_payload),
            ]
        )
        collector = RedditJsonCollector(
            subreddit="markets",
            parser=parser(),
            user_agent="test-agent",
            transport=transport,
            clock=FixedClock([BASE, BASE + timedelta(seconds=1)]),
            include_comments=True,
        )

        result = collector.collect()

        self.assertEqual(len(transport.calls), 2)
        self.assertIn("/comments/thread.json?", transport.calls[1][0])
        self.assertEqual(
            {item.external_id for item in result.observations},
            {"t3_thread", "t1_top-comment", "t1_nested"},
        )
        self.assertIn("more_comments_unfetched", {item.code for item in result.issues})
        self.assertIn("duplicate_across_pages", {item.code for item in result.issues})

    def test_collect_persist_repolls_and_edits_are_append_only(self) -> None:
        transport = FakeTransport(
            [
                response(listing([post("tracked", "Apple is strong", score=10)])),
                response(listing([post("tracked", "Apple is strong", score=20)])),
                response(listing([post("tracked", "Apple is weak", score=5)])),
            ]
        )
        collection_times = [
            BASE,
            BASE + timedelta(minutes=1),
            BASE + timedelta(minutes=2),
        ]
        persistence_times = [
            BASE + timedelta(seconds=10),
            BASE + timedelta(minutes=1, seconds=10),
            BASE + timedelta(minutes=2, seconds=10),
        ]
        store = InMemorySocialObservationStore()
        collector = RedditJsonCollector(
            subreddit="markets",
            parser=parser(),
            user_agent="test-agent",
            transport=transport,
            clock=FixedClock(collection_times),
        )
        orchestrator = SocialIngestionOrchestrator(
            collector=collector,
            store=store,
            clock=FixedClock(persistence_times),
        )

        first = orchestrator.collect_and_persist()
        second = orchestrator.collect_and_persist()
        third = orchestrator.collect_and_persist()

        self.assertTrue(first.store_result.observations[0][1])
        self.assertFalse(second.store_result.observations[0][1])
        self.assertTrue(third.store_result.observations[0][1])
        self.assertEqual(len(store), 2)
        self.assertEqual(store.engagement_count, 3)
        revisions = store.revisions_for("reddit", "t3_tracked")
        self.assertEqual([item.revision for item in revisions], [1, 2])
        self.assertEqual(revisions[0].text, "Apple is strong")
        self.assertEqual(revisions[1].text, "Apple is weak")
        self.assertEqual(revisions[0].ingested_at, persistence_times[0])
        self.assertEqual(revisions[1].ingested_at, persistence_times[2])
        self.assertEqual(first.batch.raw_posts[0].ingested_at, persistence_times[0])
        self.assertEqual(second.batch.observations[0], revisions[0])
        self.assertEqual(
            store.get_observation(
                "reddit",
                "t3_tracked",
                cutoff=persistence_times[1],
            ),
            revisions[0],
        )
        self.assertEqual(
            store.get_observation(
                "reddit",
                "t3_tracked",
                cutoff=persistence_times[2],
            ),
            revisions[1],
        )
        snapshots = store.query_engagement_snapshots(
            source="reddit",
            external_id="t3_tracked",
            cutoff=persistence_times[2],
        )
        self.assertEqual([item.score for item in snapshots], [10.0, 20.0, 5.0])

    def test_rate_limit_and_http_errors_are_explicit(self) -> None:
        limited_transport = FakeTransport(
            [
                HttpResponse(
                    status=429,
                    body=b"{}",
                    headers={"Retry-After": "12.5"},
                )
            ]
        )
        limited = RedditJsonCollector(
            subreddit="markets",
            parser=parser(),
            user_agent="test-agent",
            transport=limited_transport,
        )
        with self.assertRaises(RateLimitError) as caught:
            limited.collect()
        self.assertEqual(caught.exception.status, 429)
        self.assertEqual(caught.exception.retry_after, 12.5)

        unavailable = RedditJsonCollector(
            subreddit="markets",
            parser=parser(),
            user_agent="test-agent",
            transport=FakeTransport([HttpResponse(status=503, body=b"down")]),
        )
        with self.assertRaises(HttpStatusError) as unavailable_error:
            unavailable.collect()
        self.assertEqual(unavailable_error.exception.status, 503)

    def test_invalid_json_and_repeated_pagination_are_rejected(self) -> None:
        bad_json = RedditJsonCollector(
            subreddit="markets",
            parser=parser(),
            user_agent="test-agent",
            transport=FakeTransport([HttpResponse(status=200, body=b"not json")]),
        )
        with self.assertRaisesRegex(SocialPayloadError, "UTF-8 JSON"):
            bad_json.collect()

        repeated_payload = listing([], after="same-cursor")
        repeated = RedditJsonCollector(
            subreddit="markets",
            parser=parser(),
            user_agent="test-agent",
            transport=FakeTransport(
                [response(repeated_payload), response(repeated_payload)]
            ),
            clock=FixedClock(
                [
                    BASE,
                    BASE,
                    BASE + timedelta(seconds=1),
                    BASE + timedelta(seconds=1),
                ]
            ),
        )
        with self.assertRaisesRegex(SocialPayloadError, "cursor repeated"):
            repeated.collect(max_pages=3)

    def test_collector_validates_scope_without_making_a_request(self) -> None:
        transport = FakeTransport([])
        with self.assertRaisesRegex(ValueError, "subreddit"):
            RedditJsonCollector(
                subreddit="../secrets",
                parser=parser(),
                user_agent="test-agent",
                transport=transport,
            )
        collector = RedditJsonCollector(
            subreddit="markets",
            parser=parser(),
            user_agent="test-agent",
            transport=transport,
        )
        with self.assertRaisesRegex(ValueError, "at least 1"):
            collector.collect(max_pages=0)
        self.assertEqual(transport.calls, [])


if __name__ == "__main__":
    unittest.main()

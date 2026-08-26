from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from src.sources.social import SocialObservation, social_observation_to_document


UTC = timezone.utc
AUTHORED_AT = datetime(2026, 8, 22, 10, tzinfo=UTC)


def observation(*, ingested_at: datetime = AUTHORED_AT + timedelta(minutes=2)) -> SocialObservation:
    return SocialObservation(
        source="reddit",
        external_id="t3_airx",
        entity_ids=("company:airline-x",),
        authored_at=AUTHORED_AT,
        known_at=AUTHORED_AT + timedelta(minutes=1),
        ingested_at=ingested_at,
        text="Airline X may lose a customer.",
        content_hash="content-hash",
        uri="https://example.invalid/t3_airx",
    )


class SocialSourceAdapterTests(unittest.TestCase):
    def test_social_post_becomes_a_generic_source_document(self) -> None:
        document = social_observation_to_document(observation())

        self.assertEqual(document.content, "Airline X may lose a customer.")
        self.assertEqual(document.provenance.source_type, "social")
        self.assertEqual(document.provenance.source_id, "social:reddit")
        self.assertEqual(document.provenance.external_id, "t3_airx")
        self.assertEqual(document.provenance.published_at, AUTHORED_AT)
        self.assertEqual(
            document.provenance.observed_at,
            AUTHORED_AT + timedelta(minutes=2),
        )

    def test_future_social_post_cannot_be_adapted_for_an_earlier_cutoff(self) -> None:
        with self.assertRaisesRegex(ValueError, "unavailable"):
            social_observation_to_document(
                observation(),
                cutoff=AUTHORED_AT + timedelta(minutes=1),
            )


if __name__ == "__main__":
    unittest.main()

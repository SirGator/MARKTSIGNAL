"""Inbound source adapters for unstructured information.

Sources are deliberately kept before event extraction.  They collect and
normalise external material but do not assign an economic impact score.
"""

from .social import (
    InMemorySocialObservationStore,
    KeywordEntityResolver,
    RedditJsonCollector,
    RedditJsonParser,
    SocialIngestionOrchestrator,
    SocialObservation,
    social_observation_to_document,
)

__all__ = [
    "InMemorySocialObservationStore",
    "KeywordEntityResolver",
    "RedditJsonCollector",
    "RedditJsonParser",
    "SocialIngestionOrchestrator",
    "SocialObservation",
    "social_observation_to_document",
]

"""Bridge persisted social material into the generic extraction boundary."""

from __future__ import annotations

from datetime import datetime

from src.domain import ProvenanceRecord, _utc_datetime
from src.extraction import SourceDocument

from .domain import SocialObservation


def social_observation_to_document(
    observation: SocialObservation,
    *,
    cutoff: datetime | None = None,
) -> SourceDocument:
    """Expose one visible post/comment as untrusted source material.

    A social post has no privileged scoring path.  Once persisted, it becomes
    the same ``SourceDocument`` contract used for news, filings or political
    statements and must therefore pass through event extraction and RAG before
    it can influence an Economy Impact score.
    """

    if not isinstance(observation, SocialObservation):
        raise TypeError("observation must be a SocialObservation")
    if cutoff is not None:
        safe_cutoff = _utc_datetime("cutoff", cutoff)
        if not observation.is_available_at(safe_cutoff):
            raise ValueError("social observation was unavailable at the cutoff")
    return SourceDocument(
        content=observation.text,
        provenance=ProvenanceRecord(
            source_id=f"social:{observation.source}",
            source_type="social",
            observed_at=observation.available_at,
            published_at=observation.authored_at,
            uri=observation.uri,
            external_id=observation.external_id,
            content_hash=observation.content_hash,
        ),
    )

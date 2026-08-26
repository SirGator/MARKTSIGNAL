"""Contracts for source-specific event extractors.

V1 deliberately injects an extractor instead of coupling the pipeline to one
LLM, feed parser, or market-data adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from src.domain import CanonicalEvent, ProvenanceRecord


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """Untrusted source content plus immutable provenance."""

    content: str
    provenance: ProvenanceRecord

    def __post_init__(self) -> None:
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("content must be a non-empty string")
        if not isinstance(self.provenance, ProvenanceRecord):
            raise TypeError("provenance must be a ProvenanceRecord")
        object.__setattr__(self, "content", self.content.strip())


class EventExtractor(Protocol):
    """Convert one source document into zero or more canonical events."""

    def extract(self, document: SourceDocument) -> Sequence[CanonicalEvent]: ...

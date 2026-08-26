"""Storage- and framework-independent economy-model contract."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Protocol

from src.domain import ContextBundle


@dataclass(frozen=True, slots=True)
class EconomyScore:
    """Internal economic score before audit metadata is attached.

    ``score`` is the expected incremental economic impact relative to the
    no-event baseline for the requested horizon.  It is not a probability or a
    confidence value.  An optional confidence remains separate.
    """

    score: float
    model_version: str
    feature_schema_version: str
    input_hash: str
    evidence_ids: tuple[str, ...]
    confidence: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("score must be a number")
        score = float(self.score)
        if not math.isfinite(score) or not -1.0 <= score <= 1.0:
            raise ValueError("score must be finite and between -1 and 1")
        object.__setattr__(self, "score", score)

        if not isinstance(self.model_version, str) or not self.model_version.strip():
            raise ValueError("model_version must be a non-empty string")
        object.__setattr__(self, "model_version", self.model_version.strip())

        if not isinstance(self.feature_schema_version, str) or not self.feature_schema_version.strip():
            raise ValueError("feature_schema_version must be a non-empty string")
        object.__setattr__(
            self,
            "feature_schema_version",
            self.feature_schema_version.strip(),
        )

        if not isinstance(self.input_hash, str) or re.fullmatch(
            r"[0-9a-fA-F]{64}", self.input_hash.strip()
        ) is None:
            raise ValueError("input_hash must be a SHA-256 hex digest")
        object.__setattr__(self, "input_hash", self.input_hash.strip().casefold())

        if isinstance(self.evidence_ids, str):
            raise TypeError("evidence_ids must be a tuple of strings")
        evidence_ids = tuple(dict.fromkeys(self.evidence_ids))
        if not evidence_ids or any(
            not isinstance(item, str) or not item.strip() for item in evidence_ids
        ):
            raise ValueError("evidence_ids must contain non-empty strings")
        object.__setattr__(
            self,
            "evidence_ids",
            tuple(item.strip() for item in evidence_ids),
        )

        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(
                self.confidence, (int, float)
            ):
                raise TypeError("confidence must be a number or None")
            confidence = float(self.confidence)
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be finite and between 0 and 1")
            object.__setattr__(self, "confidence", confidence)


class EconomyModel(Protocol):
    """Interpret an immutable event/case/context bundle economically."""

    def predict(
        self,
        context: ContextBundle,
        *,
        horizon: str,
    ) -> EconomyScore: ...

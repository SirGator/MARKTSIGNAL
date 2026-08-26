"""Concrete, versioned tensor encoding for a retrieved ContextBundle.

Uses the BPETokenizer (src/models/tokenizer.py) — the same tokenizer
used by training. This guarantees identical token IDs between training
and inference.
"""

from __future__ import annotations

import hashlib
import json

import torch

from src.domain import ContextBundle
from src.models.adapter import EconomyTensorInput
from src.models.context_serializer import (
    SERIALIZER_CONTRACT_VERSION,
    SUMMARY_MODE_FULL,
    ContextSerializer,
    normalize_summary_mode,
)
from src.models.tokenizer import BPETokenizer


class ContextTensorEncoder:
    """Map serialized context text to token_ids + attention_mask.

    Uses BPETokenizer for encoding — identical to training.
    """

    def __init__(
        self,
        *,
        tokenizer: BPETokenizer,
        feature_schema_version: str,
        serializer: ContextSerializer | None = None,
        summary_mode: str | None = None,
        max_seq_len: int = 512,
    ) -> None:
        if not isinstance(feature_schema_version, str) or not feature_schema_version.strip():
            raise ValueError("feature_schema_version must be a non-empty string")
        if max_seq_len < 16:
            raise ValueError("max_seq_len must be at least 16")
        if not isinstance(tokenizer, BPETokenizer):
            raise TypeError("tokenizer must be a BPETokenizer")

        if serializer is not None and not isinstance(serializer, ContextSerializer):
            raise TypeError("serializer must be a ContextSerializer or None")
        if summary_mode is not None:
            normalized_summary_mode = normalize_summary_mode(summary_mode)
            if (
                serializer is not None
                and serializer.summary_mode != normalized_summary_mode
            ):
                raise ValueError(
                    "serializer.summary_mode does not match summary_mode"
                )
        elif serializer is not None:
            normalized_summary_mode = serializer.summary_mode
        else:
            normalized_summary_mode = SUMMARY_MODE_FULL

        self._tokenizer = tokenizer
        self._feature_schema_version = feature_schema_version.strip()
        self._serializer = serializer or ContextSerializer(
            summary_mode=normalized_summary_mode
        )
        self._summary_mode = normalized_summary_mode
        self._max_seq_len = max_seq_len

    @property
    def feature_schema_version(self) -> str:
        return self._feature_schema_version

    @property
    def summary_mode(self) -> str:
        return self._summary_mode

    def encode(
        self,
        context: ContextBundle,
        *,
        horizon: str,
    ) -> EconomyTensorInput:
        serialized = self._serializer.serialize(context, horizon=horizon)
        token_ids = self._tokenizer.encode(serialized)

        if len(token_ids) > self._max_seq_len:
            token_ids = token_ids[: self._max_seq_len]

        token_tensor = torch.tensor([token_ids], dtype=torch.long)
        attention_tensor = torch.ones((1, len(token_ids)), dtype=torch.long)

        canonical_input = json.dumps(
            {
                "feature_schema_version": self._feature_schema_version,
                "horizon": horizon.strip(),
                "serializer_contract_version": SERIALIZER_CONTRACT_VERSION,
                "summary_mode": self._summary_mode,
                "tokenizer_hash": self._tokenizer.fingerprint,
                "token_ids": token_tensor.tolist(),
                "attention_mask": attention_tensor.tolist(),
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        input_hash = hashlib.sha256(canonical_input).hexdigest()

        evidence_ids = context.evidence_ids
        if not evidence_ids:
            raise ValueError("context must contain auditable evidence")

        return EconomyTensorInput(
            token_ids=token_tensor,
            attention_mask=attention_tensor,
            feature_schema_version=self._feature_schema_version,
            input_hash=input_hash,
            evidence_ids=evidence_ids,
        )

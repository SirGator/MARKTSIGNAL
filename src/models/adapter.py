"""Adapter from ContextBundle to the EconomyEncoder text input."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import torch
from torch import Tensor

from src.domain import ContextBundle
from src.economy import EconomyScore


@dataclass(frozen=True, slots=True)
class EconomyTensorInput:
    token_ids: Tensor
    attention_mask: Tensor
    feature_schema_version: str
    input_hash: str
    evidence_ids: tuple[str, ...]


class EconomyInputEncoder(Protocol):
    """Versioned tokenizer adapter supplied by training."""

    @property
    def feature_schema_version(self) -> str: ...

    def encode(
        self,
        context: ContextBundle,
        *,
        horizon: str,
    ) -> EconomyTensorInput: ...


class TorchEconomyModel:
    """Expose EconomyEncoder behind the framework-neutral model port."""

    def __init__(
        self,
        model: torch.nn.Module,
        input_encoder: EconomyInputEncoder,
        *,
        model_version: str,
    ) -> None:
        if not isinstance(model_version, str) or not model_version.strip():
            raise ValueError("model_version must be a non-empty string")
        self._model = model.eval()
        self._input_encoder = input_encoder
        self._model_version = model_version.strip()

    def predict(
        self,
        context: ContextBundle,
        *,
        horizon: str,
    ) -> EconomyScore:
        encoded = self._input_encoder.encode(context, horizon=horizon)
        if not isinstance(encoded, EconomyTensorInput):
            raise TypeError("input_encoder.encode must return EconomyTensorInput")

        with torch.inference_mode():
            output = self._model(
                token_ids=encoded.token_ids,
                attention_mask=encoded.attention_mask,
            )

        if not isinstance(output, dict) or "score" not in output:
            raise TypeError("economy model output must contain a score tensor")
        score_tensor = output["score"]
        if not isinstance(score_tensor, Tensor) or score_tensor.numel() != 1:
            raise ValueError("economy model must return exactly one score value")
        score = float(score_tensor.detach().cpu().item())
        if not math.isfinite(score):
            raise ValueError("economy model returned a non-finite score")

        if score < -1.000001 or score > 1.000001:
            raise ValueError("economy model returned score outside [-1, 1]")
        score = min(1.0, max(-1.0, score))
        if encoded.feature_schema_version != self._input_encoder.feature_schema_version:
            raise ValueError("encoded feature schema version does not match encoder")
        return EconomyScore(
            score=score,
            model_version=self._model_version,
            feature_schema_version=encoded.feature_schema_version,
            input_hash=encoded.input_hash,
            evidence_ids=encoded.evidence_ids,
        )
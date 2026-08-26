"""Production checkpoint loader for EconomyEncoder.

This module lives in src/models/ so production never imports from training/.
It loads a checkpoint and returns a ready-to-use ContextTensorEncoder +
TorchEconomyModel pair.

max_seq_len and summary_mode are read from the checkpoint and applied to the
encoder so training and inference cannot silently use different contracts.
Legacy checkpoints without summary_mode fail closed.  A verified legacy
full-summary artifact can be loaded with an explicit override; legacy
no-summary artifacts require retraining because their marker format differed.
"""

from __future__ import annotations

from collections.abc import Mapping
import hmac
from pathlib import Path
from typing import TYPE_CHECKING

from src.models.context_serializer import (
    SERIALIZER_CONTRACT_VERSION,
    SUMMARY_MODE_FULL,
    SUMMARY_MODES,
    ContextSerializer,
    normalize_summary_mode,
)

if TYPE_CHECKING:
    from src.models.adapter import TorchEconomyModel
    from src.models.context_encoder import ContextTensorEncoder
    from src.models.tokenizer import BPETokenizer


CHECKPOINT_FORMAT_VERSION = 2


def summary_mode_from_checkpoint_config(
    config: Mapping[str, object],
    *,
    legacy_summary_mode: str | None = None,
) -> str:
    """Read and validate the checkpoint's serialization contract.

    A mode stored by a new checkpoint is authoritative.  Legacy artifacts are
    ambiguous and therefore require an explicit declaration.  Only legacy
    ``full`` is compatible: old no-summary training escaped the marker as
    ``(NO_SUMMARY)``, while the V2 contract intentionally emits the exact
    ``[NO_SUMMARY]`` token.  Such a model must be retrained, not relabelled.
    """

    if not isinstance(config, Mapping):
        raise TypeError("checkpoint config must be a mapping")
    if "summary_mode" in config:
        raw_mode = config["summary_mode"]
    else:
        if legacy_summary_mode is None:
            raise ValueError(
                "legacy checkpoint has no config.summary_mode; pass "
                "legacy_summary_mode='full' for a verified full-summary "
                "artifact, or retrain an old no-summary artifact"
            )
        try:
            legacy_mode = normalize_summary_mode(legacy_summary_mode)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid legacy_summary_mode: {legacy_summary_mode!r}"
            ) from error
        if legacy_mode != SUMMARY_MODE_FULL:
            raise ValueError(
                "legacy no-summary checkpoints used the escaped marker "
                "'(NO_SUMMARY)' and are incompatible with the V2 exact "
                "'[NO_SUMMARY]' contract; retrain the model"
            )
        raw_mode = legacy_mode
    try:
        return normalize_summary_mode(raw_mode)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"invalid checkpoint config.summary_mode: {raw_mode!r}"
        ) from error


def validate_checkpoint_metadata(
    checkpoint: Mapping[str, object],
    config: Mapping[str, object],
) -> None:
    """Validate the versioned container around a non-legacy checkpoint."""

    has_v2_mode = "summary_mode" in config
    format_version = checkpoint.get("checkpoint_format_version")
    serializer_version = checkpoint.get("serializer_contract_version")

    if not has_v2_mode:
        if format_version is not None or serializer_version is not None:
            raise ValueError(
                "checkpoint metadata is incomplete: V2 requires "
                "config.summary_mode"
            )
        return
    if format_version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "unsupported or missing checkpoint_format_version: "
            f"{format_version!r}"
        )
    if serializer_version != SERIALIZER_CONTRACT_VERSION:
        raise ValueError(
            "unsupported or missing serializer_contract_version: "
            f"{serializer_version!r}"
        )


def _configured_version(
    config: Mapping[str, object],
    key: str,
) -> str:
    raw = config.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"checkpoint config.{key} must be a non-empty string")
    return raw.strip()


def validate_checkpoint_token_ids(
    config: Mapping[str, object],
    vocab: Mapping[str, int],
    *,
    require_explicit: bool,
) -> None:
    """Keep model padding/boundary IDs identical to the tokenizer contract."""

    for config_key, token in (
        ("pad_idx", "[pad]"),
        ("cls_idx", "[cls]"),
        ("sep_idx", "[sep]"),
    ):
        if token not in vocab:
            raise ValueError(f"tokenizer vocabulary is missing {token}")
        if require_explicit and config_key not in config:
            raise ValueError(f"V2 checkpoint config.{config_key} is required")
        actual = config.get(config_key, vocab[token])
        if isinstance(actual, bool) or actual != vocab[token]:
            raise ValueError(
                f"checkpoint config.{config_key} does not match tokenizer {token}"
            )


def validate_checkpoint_versions(
    config: Mapping[str, object],
    summary_mode: str,
) -> None:
    """Require public model/schema versions to expose the input mode."""

    mode = normalize_summary_mode(summary_mode)
    expected_tag = f"summary-{mode}"
    for key in ("model_version", "feature_schema_version"):
        version = _configured_version(config, key)
        tags = {
            f"summary-{candidate}"
            for candidate in SUMMARY_MODES
            if f"summary-{candidate}" in version.casefold()
        }
        if expected_tag not in tags or len(tags) != 1:
            raise ValueError(
                f"checkpoint config.{key} must contain only {expected_tag!r}"
            )


def validate_checkpoint_sequence_length(config: Mapping[str, object]) -> int:
    """Return a production-safe maximum sequence length."""

    value = config.get("max_seq_len")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("checkpoint config.max_seq_len must be an integer")
    if value < 16:
        raise ValueError("checkpoint config.max_seq_len must be at least 16")
    return value


def load_economy_model(
    checkpoint_path: Path,
    *,
    legacy_summary_mode: str | None = None,
) -> tuple["TorchEconomyModel", "ContextTensorEncoder", "BPETokenizer"]:
    """Load a checkpoint and return (model_adapter, encoder, tokenizer).

    The returned ContextTensorEncoder uses the same max_seq_len and summary
    contract as the checkpoint.  Model and feature-schema versions are also
    checkpoint-authoritative.  ``legacy_summary_mode`` is consulted only for
    checkpoints whose config predates the summary_mode field.  It must be
    explicitly set to ``full`` for a verified legacy full-summary artifact.
    """
    # Keep torch-dependent imports inside the loader so serializer-only
    # production code remains usable without the optional ML dependency.
    import torch

    from src.models.adapter import TorchEconomyModel
    from src.models.context_encoder import ContextTensorEncoder
    from src.models.model import EconomyEncoder
    from src.models.tokenizer import BPETokenizer

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must contain a mapping")
    cfg = checkpoint["config"]
    if not isinstance(cfg, Mapping):
        raise TypeError("checkpoint config must be a mapping")
    validate_checkpoint_metadata(checkpoint, cfg)
    summary_mode = summary_mode_from_checkpoint_config(
        cfg,
        legacy_summary_mode=legacy_summary_mode,
    )
    is_v2 = "summary_mode" in cfg
    max_seq_len = validate_checkpoint_sequence_length(cfg)
    if is_v2:
        validate_checkpoint_versions(cfg, summary_mode)
    vocab = checkpoint["vocab"]
    merges = [tuple(m) for m in checkpoint.get("merges", [])]

    tokenizer = BPETokenizer(vocab=vocab, merges=merges)
    tokenizer.validate_special_tokens(include_no_summary=is_v2)
    validate_checkpoint_token_ids(cfg, tokenizer.vocab, require_explicit=is_v2)
    if cfg.get("vocab_size") != tokenizer.vocab_size:
        raise ValueError("checkpoint config.vocab_size does not match vocab")
    expected_tokenizer_hash = cfg.get("tokenizer_hash")
    if is_v2:
        if not isinstance(expected_tokenizer_hash, str):
            raise ValueError("V2 checkpoint config.tokenizer_hash is required")
        if not hmac.compare_digest(expected_tokenizer_hash, tokenizer.fingerprint):
            raise ValueError("checkpoint tokenizer_hash does not match vocab/merges")

    checkpoint_feature_schema_version = _configured_version(
        cfg,
        "feature_schema_version",
    )
    checkpoint_model_version = _configured_version(
        cfg,
        "model_version",
    )
    if not is_v2:
        # The runtime now hashes the explicit summary contract.  Preserve the
        # stored artifact identity while making that legacy execution path
        # visible in every exported score.
        checkpoint_feature_schema_version += "+legacy-summary-full-runtime-v2"
        checkpoint_model_version += "+legacy-summary-full-runtime-v2"

    model = EconomyEncoder(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        ff_dim=cfg["ff_dim"],
        max_seq_len=max_seq_len,
        dropout=cfg.get("dropout", 0.0),
        pad_idx=cfg.get("pad_idx", vocab.get("[pad]", 0)),
        cls_idx=cfg.get("cls_idx", vocab.get("[cls]", 1)),
        sep_idx=cfg.get("sep_idx", vocab.get("[sep]", 2)),
    )
    model.load_state_dict(checkpoint["model_state"])

    encoder = ContextTensorEncoder(
        tokenizer=tokenizer,
        feature_schema_version=checkpoint_feature_schema_version,
        serializer=ContextSerializer(
            summary_mode=summary_mode,
            allow_no_summary_marker=is_v2,
        ),
        summary_mode=summary_mode,
        max_seq_len=max_seq_len,
    )

    adapter = TorchEconomyModel(
        model,
        encoder,
        model_version=checkpoint_model_version,
    )

    return adapter, encoder, tokenizer

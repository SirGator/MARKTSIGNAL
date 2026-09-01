"""Training-local validation for the portable checkpoint format."""

from __future__ import annotations

from collections.abc import Mapping

from .context_serializer import (
    _LEGACY_SERIALIZER_CONTRACT_VERSIONS,
    SERIALIZER_CONTRACT_VERSION,
    SUMMARY_MODE_FULL,
    SUMMARY_MODES,
    normalize_summary_mode,
)


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
    if serializer_version not in _LEGACY_SERIALIZER_CONTRACT_VERSIONS:
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


TRAINING_STATE_VERSION = 1
TRAINING_PHASES = frozenset(("pretrain", "score"))


def training_state_from_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    expected_phase: str | None = None,
) -> dict[str, object]:
    """Return a validated resumable state or reject inference-only artifacts."""

    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must contain a mapping")
    if checkpoint.get("training_state_version") != TRAINING_STATE_VERSION:
        if "training_state" not in checkpoint:
            raise ValueError("checkpoint is inference-only and cannot be resumed")
        raise ValueError(
            "unsupported or missing training_state_version: "
            f"{checkpoint.get('training_state_version')!r}"
        )
    raw_state = checkpoint.get("training_state")
    if not isinstance(raw_state, Mapping):
        raise TypeError("checkpoint training_state must be a mapping")
    state = dict(raw_state)

    phase = state.get("phase")
    if phase not in TRAINING_PHASES:
        raise ValueError(f"invalid training phase: {phase!r}")
    if expected_phase is not None and phase != expected_phase:
        raise ValueError(
            f"training checkpoint phase {phase!r} does not match {expected_phase!r}"
        )
    for key in (
        "completed_epochs",
        "global_step",
        "steps_per_epoch",
        "target_epochs",
    ):
        value = state.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"training_state.{key} must be an integer")
        minimum = 1 if key in {"steps_per_epoch", "target_epochs"} else 0
        if value < minimum:
            raise ValueError(f"training_state.{key} must be at least {minimum}")
    if state["global_step"] != (
        state["completed_epochs"] * state["steps_per_epoch"]
    ):
        raise ValueError(
            "training_state.global_step does not match completed epochs"
        )
    if state["completed_epochs"] > state["target_epochs"]:
        raise ValueError("training_state.completed_epochs exceeds target_epochs")

    for key in (
        "optimizer_state",
        "scheduler_state",
        "rng_state",
        "run_contract",
        "phase_state",
    ):
        if not isinstance(state.get(key), Mapping):
            raise TypeError(f"training_state.{key} must be a mapping")
    if state.get("loader_generator_state") is None:
        raise TypeError("training_state.loader_generator_state is required")
    losses = state.get("losses")
    if not isinstance(losses, list) or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in losses
    ):
        raise TypeError("training_state.losses must be a list of numbers")
    if phase == "pretrain" and "mlm_head_state" not in state["phase_state"]:
        raise ValueError(
            "pretrain training_state.phase_state.mlm_head_state is required"
        )
    return state



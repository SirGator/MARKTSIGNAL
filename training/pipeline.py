"""Training pipeline for EconomyEncoder V1.

Two phases (per MODEL_SPEC.md):
    Phase 1: Pretraining (MLM on economy text)
    Phase 2: Score training (SmoothL1Loss on labeled examples)

Both phases use the same EconomyEncoder architecture.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hmac
import hashlib
import math
from pathlib import Path
import random
import struct
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .modeling.context_serializer import (
    SUMMARY_MODE_FULL,
    SUMMARY_MODES,
    normalize_summary_mode,
)
from .modeling.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    SERIALIZER_CONTRACT_VERSION,
    TRAINING_STATE_VERSION,
    training_state_from_checkpoint,
    summary_mode_from_checkpoint_config,
    validate_checkpoint_metadata,
    validate_checkpoint_sequence_length,
    validate_checkpoint_token_ids,
    validate_checkpoint_versions,
)
from .modeling.model import EconomyEncoder

from .bridge import serialize_scenario
from .scenarios import EconomicScenario


PAD_IDX = 0
CheckpointCallback = Callable[[str, int], None]
BestCheckpointCallback = Callable[[int, float], None]
ResumeCheckpointCallback = Callable[[str, int, Mapping[str, object]], None]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Configuration for one training run."""

    model_dir: Path
    vocab_size: int = 24_000
    d_model: int = 384
    num_heads: int = 6
    num_layers: int = 8
    ff_dim: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.1
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_epochs: int = 10
    log_every: int = 50
    model_version: str | None = None
    feature_schema_version: str | None = None
    summary_mode: str = SUMMARY_MODE_FULL
    seed: int = 42
    dataset_version: str | None = None
    dataset_schema_version: str | None = None
    dataset_generator_version: str | None = None
    dataset_seed: int | None = None
    runtime_noise_ratio: float = 0.0
    runtime_noise_profile_version: str = "none"

    def __post_init__(self) -> None:
        mode = normalize_summary_mode(self.summary_mode)
        object.__setattr__(self, "summary_mode", mode)

        def bind_mode(value: str | None, *, fallback: str, name: str) -> str:
            version = fallback if value is None else value
            if not isinstance(version, str) or not version.strip():
                raise ValueError(f"{name} must be a non-empty string")
            version = version.strip()
            expected_tag = f"summary-{mode}"
            present_tags = {
                f"summary-{candidate}"
                for candidate in SUMMARY_MODES
                if f"summary-{candidate}" in version.casefold()
            }
            if present_tags and (
                expected_tag not in present_tags or len(present_tags) != 1
            ):
                raise ValueError(f"{name} conflicts with summary_mode={mode!r}")
            if not present_tags:
                version = f"{version}+{expected_tag}"
            return version

        model_version = bind_mode(
            self.model_version,
            fallback="economy-encoder-v1",
            name="model_version",
        )
        feature_schema_version = bind_mode(
            self.feature_schema_version,
            fallback="economy-context-v2",
            name="feature_schema_version",
        )
        object.__setattr__(self, "model_version", model_version.strip())
        object.__setattr__(
            self,
            "feature_schema_version",
            feature_schema_version.strip(),
        )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if isinstance(self.runtime_noise_ratio, bool) or not isinstance(
            self.runtime_noise_ratio,
            (int, float),
        ):
            raise TypeError("runtime_noise_ratio must be a number")
        noise_ratio = float(self.runtime_noise_ratio)
        if not math.isfinite(noise_ratio) or not 0.0 <= noise_ratio <= 1.0:
            raise ValueError("runtime_noise_ratio must be between 0 and 1")
        object.__setattr__(self, "runtime_noise_ratio", noise_ratio)
        if (
            not isinstance(self.runtime_noise_profile_version, str)
            or not self.runtime_noise_profile_version.strip()
        ):
            raise ValueError(
                "runtime_noise_profile_version must be a non-empty string"
            )
        object.__setattr__(
            self,
            "runtime_noise_profile_version",
            self.runtime_noise_profile_version.strip(),
        )
        dataset_fields = (
            self.dataset_version,
            self.dataset_schema_version,
            self.dataset_generator_version,
            self.dataset_seed,
        )
        if any(value is not None for value in dataset_fields):
            if any(value is None for value in dataset_fields):
                raise ValueError(
                    "dataset version, schema, generator, and seed must be set together"
                )
            for name in (
                "dataset_version",
                "dataset_schema_version",
                "dataset_generator_version",
            ):
                value = getattr(self, name)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{name} must be a non-empty string")
                object.__setattr__(self, name, value.strip())
            if isinstance(self.dataset_seed, bool) or not isinstance(
                self.dataset_seed,
                int,
            ):
                raise TypeError("dataset_seed must be an integer")
        if isinstance(self.max_seq_len, bool) or not isinstance(
            self.max_seq_len,
            int,
        ):
            raise TypeError("max_seq_len must be an integer")
        if self.max_seq_len < 16:
            raise ValueError("max_seq_len must be at least 16")

        def require_int(name: str, *, minimum: int) -> None:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")

        def require_float(
            name: str,
            *,
            minimum: float,
            maximum: float | None = None,
            exclusive_minimum: bool = False,
        ) -> None:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{name} must be finite")
            lower_ok = number > minimum if exclusive_minimum else number >= minimum
            if not lower_ok or (
                maximum is not None and number > maximum
            ):
                bound = (
                    f"greater than {minimum:g}"
                    if exclusive_minimum
                    else f"at least {minimum:g}"
                )
                if maximum is not None:
                    bound += f" and at most {maximum:g}"
                raise ValueError(f"{name} must be {bound}")

        require_int("vocab_size", minimum=12)
        if self.vocab_size > 1_000_000:
            raise ValueError(
                "vocab_size must not exceed 1,000,000 (embedding matrix guard)"
            )
        for name, maximum in (
            ("d_model", 16_384),
            ("num_heads", 256),
            ("num_layers", 256),
            ("ff_dim", 131_072),
        ):
            if getattr(self, name) > maximum:
                raise ValueError(
                    f"{name} must not exceed {maximum} (memory guard)"
                )
        require_int("d_model", minimum=8)
        require_int("num_heads", minimum=1)
        require_int("num_layers", minimum=1)
        require_int("ff_dim", minimum=1)
        require_int("batch_size", minimum=1)
        require_int("warmup_steps", minimum=0)
        require_int("max_epochs", minimum=1)
        require_int("log_every", minimum=1)
        require_float("dropout", minimum=0.0, maximum=1.0)
        if self.dropout == 1.0:
            raise ValueError(
                "dropout=1.0 would zero every activation and prevents "
                "learning entirely"
            )
        require_float(
            "learning_rate",
            minimum=0.0,
            exclusive_minimum=True,
        )
        require_float("weight_decay", minimum=0.0)
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads "
                f"({self.d_model} % {self.num_heads})"
            )
        if self.model_dir is None or not isinstance(self.model_dir, Path):
            raise TypeError("model_dir must be a Path")


class TextDataset(Dataset):
    """Simple text dataset for MLM pretraining.

    Tokenization happens once in the constructor; repeated epochs reuse the
    cached token ids instead of re-running BPE for every sample.
    """

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: "BPETokenizer",
        max_seq_len: int,
    ) -> None:
        self._cls_idx = tokenizer.vocab.get("[cls]", 1)
        self._unk_idx = tokenizer.vocab["[unk]"]
        self._cache = [self._encode(text, tokenizer, max_seq_len) for text in texts]

    def _encode(
        self,
        text: str,
        tokenizer: "BPETokenizer",
        max_seq_len: int,
    ) -> list[int]:
        token_ids = tokenizer.encode(text)
        if len(token_ids) > max_seq_len:
            token_ids = token_ids[:max_seq_len]
        if not token_ids:
            token_ids = [self._unk_idx]
        return token_ids

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        token_ids = self._cache[idx]
        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "length": len(token_ids),
        }


class ScoreDataset(Dataset):
    """Dataset for score training: serialized text + score label.

    Tokenization happens once in the constructor; repeated epochs and the
    per-epoch validation loop reuse the cached token ids.
    """

    def __init__(
        self,
        examples: Sequence,
        tokenizer: "BPETokenizer",
        max_seq_len: int,
        *,
        summary_mode: str = SUMMARY_MODE_FULL,
    ) -> None:
        self._cls_idx = tokenizer.vocab.get("[cls]", 1)
        self._unk_idx = tokenizer.vocab["[unk]"]
        self._scores = [
            torch.tensor(float(example.score), dtype=torch.float32)
            for example in examples
        ]
        self._cache = [
            self._encode(example, tokenizer, max_seq_len, summary_mode=summary_mode)
            for example in examples
        ]

    def _encode(
        self,
        example,
        tokenizer: "BPETokenizer",
        max_seq_len: int,
        *,
        summary_mode: str,
    ) -> list[int]:
        text = serialize_training_example(example, summary_mode=summary_mode)
        token_ids = tokenizer.encode(text)
        if len(token_ids) > max_seq_len:
            token_ids = token_ids[:max_seq_len]
        if not token_ids:
            token_ids = [self._unk_idx]
        if token_ids[0] != self._cls_idx:
            token_ids = [self._cls_idx] + token_ids[: max_seq_len - 1]
        return token_ids

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "token_ids": torch.tensor(self._cache[idx], dtype=torch.long),
            "score": self._scores[idx],
        }


def serialize_training_example(example, *, summary_mode: str) -> str:
    """Serialize one labeled example under the run-wide summary contract.

    EconomicScenario is the active training format and goes through the same
    ContextBundle bridge as evaluation.  The unstructured legacy
    TrainingExample format remains supported only for full-summary runs;
    structured-only runs fail closed instead of dropping economic fields.

    DatasetRecord values (from a frozen, versioned dataset) are routed through
    the same ContextBundle path via :func:`training.bridge.serialize_record`.
    """

    mode = normalize_summary_mode(summary_mode)
    if isinstance(example, EconomicScenario):
        return serialize_scenario(example, summary_mode=mode)

    # Avoid importing dataset.* when torch-only environments do not need it.
    try:
        from dataset.schema import DatasetRecord
    except ModuleNotFoundError:
        DatasetRecord = None  # type: ignore[assignment]
    if DatasetRecord is not None and isinstance(example, DatasetRecord):
        from .bridge import serialize_record
        return serialize_record(example, summary_mode=mode)

    serialized = example.to_serialized()
    if mode == SUMMARY_MODE_FULL:
        return serialized
    raise ValueError(
        "summary_mode='none' requires EconomicScenario or DatasetRecord "
        "structured fields; legacy TrainingExample would lose event "
        "type/direction/magnitude"
    )


def collate_pad(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad sequences to the same length within a batch."""
    max_len = max(item["token_ids"].shape[0] for item in batch)
    batch_size = len(batch)

    token_ids = torch.full((batch_size, max_len), PAD_IDX, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["token_ids"].shape[0]
        token_ids[i, :seq_len] = item["token_ids"]
        attention_mask[i, :seq_len] = 1

    result: dict[str, torch.Tensor] = {
        "token_ids": token_ids,
        "attention_mask": attention_mask,
    }
    if "score" in batch[0]:
        result["score"] = torch.stack([item["score"] for item in batch])
    return result


def predict_scores(
    model: nn.Module,
    examples: Sequence,
    tokenizer: "BPETokenizer",
    *,
    max_seq_len: int,
    summary_mode: str = SUMMARY_MODE_FULL,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> list[float]:
    """Predict a score for every example using the training input contract."""

    if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int):
        raise TypeError("max_seq_len must be an integer")
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if device is None:
        device = torch.device("cpu")

    dataset = ScoreDataset(
        examples,
        tokenizer,
        max_seq_len,
        summary_mode=summary_mode,
    )
    if not dataset:
        return []
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_pad,
    )

    previous_training_mode = getattr(model, "training", None)
    model.eval()
    predictions: list[float] = []
    try:
        with torch.inference_mode():
            for batch in loader:
                output = model(
                    token_ids=batch["token_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                predictions.extend(
                    float(value)
                    for value in output["score"].detach().cpu().reshape(-1).tolist()
                )
    finally:
        if isinstance(previous_training_mode, bool):
            model.train(previous_training_mode)
    return predictions


def apply_mlm(
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_prob: float = 0.15,
    *,
    cls_idx: int = 1,
    sep_idx: int = 2,
    mask_idx: int = 4,
    protected_idx: tuple[int, ...] = (),
    random_replace_prob: float = 0.1,
    keep_original_prob: float = 0.1,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply masked-language-model masking (BERT-style 80/10/10).

    Of the selected positions, 80% become ``[MASK]``, 10% a random vocabulary
    id, and 10% stay unchanged.  Structural tokens ([CLS]/[SEP] and every id
    in ``protected_idx``, e.g. [EVENT]/[CASE]/[CONTEXT]/[HORIZON]) are never
    selected, so the model always knows the input scaffold.

    Returns: (masked_ids, labels, mlm_mask)
    labels is -100 where no masking occurred (ignored by loss).
    """
    if isinstance(mask_prob, bool) or not isinstance(mask_prob, (int, float)):
        raise TypeError("mask_prob must be a number")
    if not math.isfinite(float(mask_prob)) or not 0.0 < float(mask_prob) <= 1.0:
        raise ValueError("mask_prob must be in (0, 1]")
    for name, prob in (
        ("random_replace_prob", random_replace_prob),
        ("keep_original_prob", keep_original_prob),
    ):
        if isinstance(prob, bool) or not isinstance(prob, (int, float)):
            raise TypeError(f"{name} must be a number")
        if not math.isfinite(float(prob)) or not 0.0 <= float(prob) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if random_replace_prob + keep_original_prob > 1.0:
        raise ValueError(
            "random_replace_prob + keep_original_prob must not exceed 1"
        )

    labels = torch.full_like(token_ids, -100)
    mlm_mask = torch.zeros_like(token_ids, dtype=torch.bool)

    structural = token_ids.new_tensor(
        (cls_idx, sep_idx, *protected_idx), dtype=torch.long
    )
    is_structural = (
        (token_ids.unsqueeze(-1) == structural.view(1, 1, -1)).any(dim=-1)
        if structural.numel()
        else torch.zeros_like(token_ids, dtype=torch.bool)
    )

    mask_candidates = attention_mask.bool() & ~is_structural

    rand = torch.rand_like(token_ids.float(), generator=generator)
    selected = (rand < mask_prob) & mask_candidates

    if not selected.any():
        # Zero masked positions make cross_entropy over ignore_index=-100
        # degenerate to NaN.  Fall back to exactly one maskable position per
        # affected sample instead of producing an undefined loss.
        fallback = torch.rand_like(token_ids.float(), generator=generator)
        fallback_scores = fallback.masked_fill(~mask_candidates, float("-inf"))
        best = fallback_scores.argmax(dim=1, keepdim=True)
        per_sample_ok = mask_candidates.any(dim=1)
        selected = torch.zeros_like(selected).scatter(
            1,
            best,
            per_sample_ok.unsqueeze(1),
        )

    labels[selected] = token_ids[selected]
    mlm_mask = selected

    decision = torch.rand_like(token_ids.float(), generator=generator)
    mask_threshold = 1.0 - random_replace_prob - keep_original_prob
    random_threshold = 1.0 - keep_original_prob
    replace_with_mask = selected & (decision < mask_threshold)
    replace_randomly = (
        selected
        & (decision >= mask_threshold)
        & (decision < random_threshold)
    )

    masked_ids = token_ids.clone()
    masked_ids[replace_with_mask] = mask_idx
    if replace_randomly.any():
        vocab_floor = 0
        vocab_ceil = int(token_ids.max().item()) + 1
        if vocab_ceil > 1:
            random_ids = torch.randint(
                vocab_floor,
                vocab_ceil,
                tuple(replace_randomly.shape),
                generator=generator,
                device=token_ids.device,
            )
            masked_ids[replace_randomly] = random_ids[replace_randomly]

    return masked_ids, labels, mlm_mask


class MLMHead(nn.Module):
    """Language modeling head for pretraining."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        from .modeling.model import RMSNorm

        self.norm = RMSNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


def build_vocabulary(
    texts: Sequence[str],
    vocab_size: int = 24_000,
    verbose: bool = False,
) -> dict[str, int]:
    """Build a BPE vocabulary.

    Uses the BPETokenizer to learn subword merges from the training text.
    Returns the vocab dict (token -> id) for compatibility with the rest
    of the pipeline.
    """
    from .modeling.tokenizer import BPETokenizer

    tokenizer = BPETokenizer.train(texts, vocab_size=vocab_size, verbose=verbose)
    return tokenizer.vocab


def build_tokenizer(
    texts: Sequence[str],
    vocab_size: int = 24_000,
    verbose: bool = False,
) -> "BPETokenizer":
    """Train and return a BPE tokenizer."""
    from .modeling.tokenizer import BPETokenizer

    return BPETokenizer.train(texts, vocab_size=vocab_size, verbose=verbose)


def save_checkpoint(
    model: nn.Module,
    tokenizer: "BPETokenizer",
    config: TrainingConfig,
    path: Path,
    *,
    training_state: Mapping[str, object] | None = None,
) -> None:
    """Save model, tokenizer and config as a checkpoint."""
    tokenizer.validate_special_tokens(include_no_summary=True)
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError("config.vocab_size does not match tokenizer vocabulary")
    checkpoint_config = {
        "vocab_size": config.vocab_size,
        "d_model": config.d_model,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "ff_dim": config.ff_dim,
        "max_seq_len": config.max_seq_len,
        "dropout": config.dropout,
        "pad_idx": model.pad_idx,
        "cls_idx": model.cls_idx,
        "sep_idx": model.sep_idx,
        "model_version": config.model_version,
        "feature_schema_version": config.feature_schema_version,
        "summary_mode": config.summary_mode,
        "tokenizer_hash": tokenizer.fingerprint,
        "seed": config.seed,
        "runtime_noise_ratio": config.runtime_noise_ratio,
        "runtime_noise_profile_version": config.runtime_noise_profile_version,
    }
    if config.dataset_version is not None:
        checkpoint_config.update(
            {
                "dataset_version": config.dataset_version,
                "dataset_schema_version": config.dataset_schema_version,
                "dataset_generator_version": config.dataset_generator_version,
                "dataset_seed": config.dataset_seed,
            }
        )
        from .input_contract import dataset_projection_metadata

        checkpoint_config.update(dataset_projection_metadata())

    validate_checkpoint_token_ids(
        checkpoint_config,
        tokenizer.vocab,
        require_explicit=True,
    )
    validate_checkpoint_versions(checkpoint_config, config.summary_mode)
    validate_checkpoint_sequence_length(checkpoint_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "artifact_kind": "resume" if training_state is not None else "inference",
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "serializer_contract_version": SERIALIZER_CONTRACT_VERSION,
        "model_state": model.state_dict(),
        "vocab": tokenizer.vocab,
        "merges": tokenizer.merges,
        "config": checkpoint_config,
    }
    if training_state is not None:
        payload["training_state_version"] = TRAINING_STATE_VERSION
        payload["training_state"] = dict(training_state)
        training_state_from_checkpoint(payload)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def load_checkpoint(
    path: Path,
    *,
    legacy_summary_mode: str | None = None,
    legacy_projection: bool = False,
) -> tuple[EconomyEncoder, "BPETokenizer", dict[str, object]]:
    """Load a checkpoint and return (model, tokenizer, config).

    Legacy artifacts fail closed unless a verified full-summary contract is
    declared explicitly.  Old no-summary artifacts require retraining because
    they used a different serialized marker.
    """
    from .modeling.tokenizer import BPETokenizer

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must contain a mapping")
    raw_cfg = checkpoint["config"]
    if not isinstance(raw_cfg, Mapping):
        raise TypeError("checkpoint config must be a mapping")
    validate_checkpoint_metadata(checkpoint, raw_cfg)
    cfg = dict(raw_cfg)
    cfg["summary_mode"] = summary_mode_from_checkpoint_config(
        cfg,
        legacy_summary_mode=legacy_summary_mode,
    )
    is_v2 = "summary_mode" in raw_cfg
    if is_v2:
        validate_checkpoint_versions(cfg, cfg["summary_mode"])
    validate_checkpoint_sequence_length(cfg)
    if "dataset_schema_version" in cfg:
        from .input_contract import validate_dataset_projection_metadata

        validate_dataset_projection_metadata(
            cfg,
            legacy_projection=legacy_projection,
        )
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
    model = EconomyEncoder(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        ff_dim=cfg["ff_dim"],
        max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"],
        pad_idx=cfg.get("pad_idx", vocab.get("[pad]", 0)),
        cls_idx=cfg.get("cls_idx", vocab.get("[cls]", 1)),
        sep_idx=cfg.get("sep_idx", vocab.get("[sep]", 2)),
    )
    model.load_state_dict(checkpoint["model_state"])
    return model, tokenizer, cfg


def load_training_checkpoint(
    path: Path,
    *,
    expected_phase: str | None = None,
    legacy_summary_mode: str | None = None,
) -> tuple[
    EconomyEncoder,
    "BPETokenizer",
    dict[str, object],
    dict[str, object],
]:
    """Load both the portable inference artifact and its resumable state."""

    model, tokenizer, config = load_checkpoint(
        path,
        legacy_summary_mode=legacy_summary_mode,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = training_state_from_checkpoint(
        checkpoint,
        expected_phase=expected_phase,
    )
    return model, tokenizer, config, state


def create_causal_lr_schedule(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup followed by cosine decay."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)




_TRAINING_ALGORITHM_VERSION = "economy-trainer-v1"


def _training_data_fingerprint(
    phase: str,
    items: Sequence,
    *,
    summary_mode: str,
) -> str:
    """Hash ordered model inputs and score labels for strict resume checks."""

    if phase not in {"pretrain", "score"}:
        raise ValueError(f"unknown training phase: {phase!r}")
    digest = hashlib.sha256()
    for item in items:
        text = (
            str(item)
            if phase == "pretrain"
            else serialize_training_example(item, summary_mode=summary_mode)
        )
        payload = text.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        if phase == "score":
            digest.update(struct.pack("!f", float(item.score)))
    return digest.hexdigest()


def _training_run_contract(
    *,
    phase: str,
    config: TrainingConfig,
    tokenizer: "BPETokenizer",
    data_fingerprint: str,
    steps_per_epoch: int,
    device: torch.device,
) -> dict[str, object]:
    """Describe every setting that must match for an exact resume."""

    return {
        "algorithm_version": _TRAINING_ALGORITHM_VERSION,
        "phase": phase,
        "tokenizer_hash": tokenizer.fingerprint,
        "data_fingerprint": data_fingerprint,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_steps": config.warmup_steps,
        "target_epochs": config.max_epochs,
        "steps_per_epoch": steps_per_epoch,
        "d_model": config.d_model,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "ff_dim": config.ff_dim,
        "max_seq_len": config.max_seq_len,
        "dropout": config.dropout,
        "summary_mode": config.summary_mode,
        "seed": config.seed,
        "runtime_noise_ratio": config.runtime_noise_ratio,
        "runtime_noise_profile_version": config.runtime_noise_profile_version,
        "dataset_version": config.dataset_version,
        "dataset_schema_version": config.dataset_schema_version,
        "dataset_generator_version": config.dataset_generator_version,
        "dataset_seed": config.dataset_seed,
        "device_type": device.type,
        "torch_version": str(torch.__version__),
    }


def _validate_training_run_contract(
    state: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    actual = state.get("run_contract")
    if not isinstance(actual, Mapping):
        raise TypeError("training_state.run_contract must be a mapping")
    differing = sorted(
        key
        for key in set(actual) | set(expected)
        if actual.get(key) != expected.get(key)
    )
    if differing:
        raise ValueError(
            "resume run contract differs for: " + ", ".join(differing)
        )


def _capture_rng_state(device: torch.device) -> dict[str, object]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if device.type == "cuda" else []
        ),
    }


def _restore_rng_state(
    state: Mapping[str, object],
    device: torch.device,
) -> None:
    raw_rng = state.get("rng_state")
    if not isinstance(raw_rng, Mapping):
        raise TypeError("training_state.rng_state must be a mapping")
    try:
        random.setstate(raw_rng["python"])
        torch.set_rng_state(raw_rng["torch_cpu"])
        if device.type == "cuda":
            cuda_state = raw_rng.get("torch_cuda")
            if not isinstance(cuda_state, list) or not cuda_state:
                raise ValueError("CUDA resume checkpoint has no CUDA RNG state")
            torch.cuda.set_rng_state_all(cuda_state)
    except KeyError as error:
        raise ValueError(
            f"training_state.rng_state is missing {error.args[0]!r}"
        ) from error


def _build_training_state(
    *,
    phase: str,
    completed_epochs: int,
    global_step: int,
    steps_per_epoch: int,
    config: TrainingConfig,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loader_generator: torch.Generator,
    run_contract: Mapping[str, object],
    phase_state: Mapping[str, object],
    losses: list[float],
    device: torch.device,
) -> dict[str, object]:
    return {
        "phase": phase,
        "completed_epochs": completed_epochs,
        "global_step": global_step,
        "steps_per_epoch": steps_per_epoch,
        "target_epochs": config.max_epochs,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "loader_generator_state": loader_generator.get_state(),
        "rng_state": _capture_rng_state(device),
        "run_contract": dict(run_contract),
        "phase_state": dict(phase_state),
        "losses": list(losses),
    }
def pretrain(
    model: EconomyEncoder,
    texts: Sequence[str],
    tokenizer: "BPETokenizer",
    config: TrainingConfig,
    device: torch.device | None = None,
    *,
    checkpoint_every_epochs: int = 0,
    checkpoint_callback: CheckpointCallback | None = None,
    resume_checkpoint_callback: ResumeCheckpointCallback | None = None,
    resume_state: Mapping[str, object] | None = None,
) -> dict[str, list[float]]:
    """Phase 1: Pretrain EconomyEncoder with MLM.

    Returns loss history for logging.
    """
    _validate_checkpoint_callback(
        checkpoint_every_epochs,
        checkpoint_callback,
        resume_checkpoint_callback,
    )

    if device is None:
        device = torch.device("cpu")

    torch.manual_seed(config.seed + 1_000)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(config.seed + 1_001)
    model = model.to(device)
    dataset = TextDataset(texts, tokenizer, config.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_pad,
        generator=loader_generator,
    )

    mlm_head = MLMHead(config.d_model, config.vocab_size).to(device)

    params = list(model.parameters()) + list(mlm_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)

    total_steps = config.max_epochs * len(loader)
    scheduler = create_causal_lr_schedule(optimizer, config.warmup_steps, total_steps)
    data_fingerprint = _training_data_fingerprint(
        "pretrain",
        texts,
        summary_mode=config.summary_mode,
    )
    run_contract = _training_run_contract(
        phase="pretrain",
        config=config,
        tokenizer=tokenizer,
        data_fingerprint=data_fingerprint,
        steps_per_epoch=len(loader),
        device=device,
    )


    cls_idx = model.cls_idx
    sep_idx = model.sep_idx
    mask_idx = tokenizer.vocab.get("[mask]", 4)
    protected_idx = tuple(
        tokenizer.vocab[token]
        for token in (
            "[event]",
            "[case]",
            "[context]",
            "[horizon]",
            "[historical_event]",
            "[case_entity]",
            "[no_summary]",
        )
        if token in tokenizer.vocab
    )

    model.train()
    mlm_head.train()
    losses: list[float] = []
    start_epoch = 0
    step = 0
    if resume_state is not None:
        state = training_state_from_checkpoint(
            {
                "training_state_version": TRAINING_STATE_VERSION,
                "training_state": dict(resume_state),
            },
            expected_phase="pretrain",
        )
        _validate_training_run_contract(state, run_contract)
        phase_state = state["phase_state"]
        mlm_head.load_state_dict(phase_state["mlm_head_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        loader_generator.set_state(state["loader_generator_state"])
        losses = [float(value) for value in state["losses"]]
        start_epoch = int(state["completed_epochs"])
        step = int(state["global_step"])
        _restore_rng_state(state, device)
        print(f"[pretrain] resume after epoch {start_epoch}")

    for epoch in range(start_epoch, config.max_epochs):
        for batch in loader:
            token_ids = batch["token_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            masked_ids, labels, _ = apply_mlm(
                token_ids,
                attention_mask,
                cls_idx=cls_idx,
                sep_idx=sep_idx,
                mask_idx=mask_idx,
                protected_idx=protected_idx,
            )

            optimizer.zero_grad()

            if not (labels != -100).any():
                raise ValueError(
                    "MLM batch contains no maskable tokens; pretraining "
                    "requires at least one non-structural token per batch"
                )

            embeddings = _forward_embeddings(model, masked_ids, attention_mask)
            logits = mlm_head(embeddings)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % config.log_every == 0:
                losses.append(loss.item())
                print(f"[pretrain] epoch={epoch} step={step} loss={loss.item():.4f}")

        completed_epoch = epoch + 1
        resume_snapshot = None
        if (
            resume_checkpoint_callback is not None
            and checkpoint_every_epochs > 0
            and completed_epoch % checkpoint_every_epochs == 0
        ):
            resume_snapshot = _build_training_state(
                phase="pretrain",
                completed_epochs=completed_epoch,
                global_step=step,
                steps_per_epoch=len(loader),
                config=config,
                optimizer=optimizer,
                scheduler=scheduler,
                loader_generator=loader_generator,
                run_contract=run_contract,
                phase_state={"mlm_head_state": mlm_head.state_dict()},
                losses=losses,
                device=device,
            )

        _save_periodic_checkpoint_if_due(
            phase="pretrain",
            completed_epochs=completed_epoch,
            checkpoint_every_epochs=checkpoint_every_epochs,
            checkpoint_callback=checkpoint_callback,
            resume_checkpoint_callback=resume_checkpoint_callback,
            training_state=resume_snapshot,
        )

    return {"losses": losses}


def train_scores(
    model: EconomyEncoder,
    examples: Sequence,
    tokenizer: "BPETokenizer",
    config: TrainingConfig,
    device: torch.device | None = None,
    *,
    checkpoint_every_epochs: int = 0,
    checkpoint_callback: CheckpointCallback | None = None,
    validation_examples: Sequence | None = None,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 1e-4,
    evaluation_batch_size: int | None = None,
    best_checkpoint_callback: BestCheckpointCallback | None = None,
    resume_checkpoint_callback: ResumeCheckpointCallback | None = None,
    resume_state: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Phase 2: Train EconomyEncoder on score labels with SmoothL1Loss.

    Optionally validates after every epoch, restores the best model and stops
    after a configured number of non-improving epochs.
    """
    _validate_checkpoint_callback(
        checkpoint_every_epochs,
        checkpoint_callback,
        resume_checkpoint_callback,
    )

    if isinstance(early_stopping_patience, bool) or not isinstance(
        early_stopping_patience,
        int,
    ):
        raise TypeError("early_stopping_patience must be an integer")
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative")
    if isinstance(early_stopping_min_delta, bool) or not isinstance(
        early_stopping_min_delta,
        (int, float),
    ):
        raise TypeError("early_stopping_min_delta must be a number")
    min_delta = float(early_stopping_min_delta)
    if not math.isfinite(min_delta) or min_delta < 0.0:
        raise ValueError(
            "early_stopping_min_delta must be finite and non-negative"
        )
    if best_checkpoint_callback is not None and not callable(
        best_checkpoint_callback
    ):
        raise TypeError("best_checkpoint_callback must be callable")

    eval_batch_size = (
        config.batch_size
        if evaluation_batch_size is None
        else evaluation_batch_size
    )
    if isinstance(eval_batch_size, bool) or not isinstance(eval_batch_size, int):
        raise TypeError("evaluation_batch_size must be an integer")
    if eval_batch_size < 1:
        raise ValueError("evaluation_batch_size must be positive")
    validation = tuple(validation_examples or ())
    if not validation and early_stopping_patience:
        print(
            "[score] no validation examples; early stopping is disabled "
            "for this run"
        )

    if device is None:
        device = torch.device("cpu")

    torch.manual_seed(config.seed + 2_000)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(config.seed + 2_001)
    model = model.to(device)
    dataset = ScoreDataset(
        examples,
        tokenizer,
        config.max_seq_len,
        summary_mode=config.summary_mode,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_pad,
        generator=loader_generator,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_steps = config.max_epochs * len(loader)
    scheduler = create_causal_lr_schedule(optimizer, config.warmup_steps, total_steps)
    data_fingerprint = _training_data_fingerprint(
        "score",
        examples,
        summary_mode=config.summary_mode,
    )
    run_contract = _training_run_contract(
        phase="score",
        config=config,
        tokenizer=tokenizer,
        data_fingerprint=data_fingerprint,
        steps_per_epoch=len(loader),
        device=device,
    )
    run_contract.update(
        {
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": min_delta,
            "validation_fingerprint": (
                _training_data_fingerprint(
                    "score",
                    validation,
                    summary_mode=config.summary_mode,
                )
                if validation
                else None
            ),
        }
    )


    loss_fn = nn.SmoothL1Loss()
    model.train()
    losses: list[float] = []
    validation_mae: list[float] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_mae = math.inf
    best_epoch: int | None = None
    epochs_without_improvement = 0
    epochs_completed = 0
    stopped_early = False
    start_epoch = 0
    step = 0

    if resume_state is not None:
        state = training_state_from_checkpoint(
            {
                "training_state_version": TRAINING_STATE_VERSION,
                "training_state": dict(resume_state),
            },
            expected_phase="score",
        )
        _validate_training_run_contract(state, run_contract)
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        loader_generator.set_state(state["loader_generator_state"])
        losses = [float(value) for value in state["losses"]]
        phase_state = state["phase_state"]
        raw_validation_mae = phase_state.get("validation_mae", [])
        if not isinstance(raw_validation_mae, list):
            raise TypeError("score phase_state.validation_mae must be a list")
        validation_mae = [float(value) for value in raw_validation_mae]
        raw_best_state = phase_state.get("best_model_state", {})
        if not isinstance(raw_best_state, Mapping):
            raise TypeError("score phase_state.best_model_state must be a mapping")
        best_state = dict(raw_best_state) or None
        raw_best_mae = phase_state.get("best_mae")
        best_mae = math.inf if raw_best_mae is None else float(raw_best_mae)
        raw_best_epoch = phase_state.get("best_epoch")
        best_epoch = None if raw_best_epoch is None else int(raw_best_epoch)
        epochs_without_improvement = int(
            phase_state.get("epochs_without_improvement", 0)
        )
        start_epoch = int(state["completed_epochs"])
        epochs_completed = start_epoch
        step = int(state["global_step"])
        stopped_early = bool(
            validation
            and early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        )
        _restore_rng_state(state, device)
        print(f"[score] resume after epoch {start_epoch}")

    end_epoch = start_epoch if stopped_early else config.max_epochs
    for epoch in range(start_epoch, end_epoch):
        for batch in loader:
            token_ids = batch["token_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            scores = batch["score"].to(device)

            optimizer.zero_grad()

            output = model(token_ids=token_ids, attention_mask=attention_mask)
            predicted = output["score"].squeeze(-1)

            loss = loss_fn(predicted, scores)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % config.log_every == 0:
                losses.append(loss.item())
                print(f"[score] epoch={epoch} step={step} loss={loss.item():.4f}")

        completed_epoch = epoch + 1
        epochs_completed = completed_epoch
        if validation:
            from .metrics import compute_score_metrics

            predictions = predict_scores(
                model,
                validation,
                tokenizer,
                max_seq_len=config.max_seq_len,
                summary_mode=config.summary_mode,
                batch_size=eval_batch_size,
                device=device,
            )
            targets = [float(example.score) for example in validation]
            metrics = compute_score_metrics(targets, predictions)
            validation_mae.append(metrics.mae)
            print(
                f"[score] epoch={epoch} validation_mae={metrics.mae:.4f} "
                f"sign_accuracy={metrics.sign_accuracy:.3f}"
            )

            if metrics.mae < best_mae - min_delta:
                best_mae = metrics.mae
                best_epoch = completed_epoch
                epochs_without_improvement = 0
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                if best_checkpoint_callback is not None:
                    best_checkpoint_callback(completed_epoch, metrics.mae)
            else:
                epochs_without_improvement += 1

        resume_snapshot = None
        if (
            resume_checkpoint_callback is not None
            and checkpoint_every_epochs > 0
            and completed_epoch % checkpoint_every_epochs == 0
        ):
            resume_snapshot = _build_training_state(
                phase="score",
                completed_epochs=completed_epoch,
                global_step=step,
                steps_per_epoch=len(loader),
                config=config,
                optimizer=optimizer,
                scheduler=scheduler,
                loader_generator=loader_generator,
                run_contract=run_contract,
                phase_state={
                    "validation_mae": list(validation_mae),
                    "best_model_state": best_state or {},
                    "best_mae": None if best_epoch is None else best_mae,
                    "best_epoch": best_epoch,
                    "epochs_without_improvement": epochs_without_improvement,
                },
                losses=losses,
                device=device,
            )

        _save_periodic_checkpoint_if_due(
            phase="score",
            completed_epochs=completed_epoch,
            checkpoint_every_epochs=checkpoint_every_epochs,
            checkpoint_callback=checkpoint_callback,
            resume_checkpoint_callback=resume_checkpoint_callback,
            training_state=resume_snapshot,
        )

        if (
            validation
            and early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            stopped_early = True
            print(
                "[score] early stopping after "
                f"{completed_epoch} epochs; best epoch={best_epoch}"
            )
            break


    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        model.train()

    return {
        "losses": losses,
        "validation_mae": validation_mae,
        "best_epoch": best_epoch,
        "best_mae": None if best_epoch is None else best_mae,
        "epochs_completed": epochs_completed,
        "stopped_early": stopped_early,
    }


def _validate_checkpoint_callback(
    checkpoint_every_epochs: int,
    checkpoint_callback: CheckpointCallback | None,
    resume_checkpoint_callback: ResumeCheckpointCallback | None = None,
) -> None:
    """Validate optional inference and resumable epoch-boundary hooks."""

    if isinstance(checkpoint_every_epochs, bool) or not isinstance(
        checkpoint_every_epochs,
        int,
    ):
        raise TypeError("checkpoint_every_epochs must be an integer")
    if checkpoint_every_epochs < 0:
        raise ValueError("checkpoint_every_epochs must be non-negative")
    if checkpoint_callback is not None and not callable(checkpoint_callback):
        raise TypeError("checkpoint_callback must be callable")
    if resume_checkpoint_callback is not None and not callable(
        resume_checkpoint_callback
    ):
        raise TypeError("resume_checkpoint_callback must be callable")
    if (
        checkpoint_every_epochs > 0
        and checkpoint_callback is None
        and resume_checkpoint_callback is None
    ):
        raise ValueError(
            "a checkpoint callback is required when checkpoint_every_epochs "
            "is positive"
        )


def _save_periodic_checkpoint_if_due(
    *,
    phase: str,
    completed_epochs: int,
    checkpoint_every_epochs: int,
    checkpoint_callback: CheckpointCallback | None,
    resume_checkpoint_callback: ResumeCheckpointCallback | None = None,
    training_state: Mapping[str, object] | None = None,
) -> None:
    """Invoke checkpoint hooks after a configured completed epoch."""

    if (
        checkpoint_every_epochs <= 0
        or completed_epochs % checkpoint_every_epochs != 0
    ):
        return
    if checkpoint_callback is not None:
        checkpoint_callback(phase, completed_epochs)
    if resume_checkpoint_callback is not None:
        if training_state is None:
            raise ValueError("training_state is required for resume checkpoints")
        resume_checkpoint_callback(phase, completed_epochs, training_state)


def _forward_embeddings(
    model: EconomyEncoder,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run the encoder and return the full hidden states (not just [CLS])."""
    x = model.token_embedding(token_ids)
    key_padding_mask = ~attention_mask.bool()
    for block in model.blocks:
        x = block(x, key_padding_mask=key_padding_mask)
    return model.final_norm(x)

"""Training pipeline for EconomyEncoder V1.

Two phases (per MODEL_SPEC.md):
    Phase 1: Pretraining (MLM on economy text)
    Phase 2: Score training (SmoothL1Loss on labeled examples)

Both phases use the same EconomyEncoder architecture.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hmac
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.models.context_serializer import (
    SUMMARY_MODE_FULL,
    SUMMARY_MODES,
    normalize_summary_mode,
)
from src.models.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    SERIALIZER_CONTRACT_VERSION,
    summary_mode_from_checkpoint_config,
    validate_checkpoint_metadata,
    validate_checkpoint_sequence_length,
    validate_checkpoint_token_ids,
    validate_checkpoint_versions,
)
from src.models.model import EconomyEncoder

from .bridge import serialize_scenario
from .scenarios import EconomicScenario


PAD_IDX = 0


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
        if isinstance(self.max_seq_len, bool) or not isinstance(
            self.max_seq_len,
            int,
        ):
            raise TypeError("max_seq_len must be an integer")
        if self.max_seq_len < 16:
            raise ValueError("max_seq_len must be at least 16")


class TextDataset(Dataset):
    """Simple text dataset for MLM pretraining."""

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: "BPETokenizer",
        max_seq_len: int,
    ) -> None:
        self._texts = list(texts)
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self._texts[idx]
        token_ids = self._tokenizer.encode(text)

        if len(token_ids) > self._max_seq_len:
            token_ids = token_ids[: self._max_seq_len]

        if not token_ids:
            token_ids = [self._tokenizer.vocab["[unk]"]]

        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "length": len(token_ids),
        }


class ScoreDataset(Dataset):
    """Dataset for score training: serialized text + score label."""

    def __init__(
        self,
        examples: Sequence,
        tokenizer: "BPETokenizer",
        max_seq_len: int,
        *,
        summary_mode: str = SUMMARY_MODE_FULL,
    ) -> None:
        self._examples = list(examples)
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len
        self._cls_idx = tokenizer.vocab.get("[cls]", 1)
        self._summary_mode = normalize_summary_mode(summary_mode)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self._examples[idx]
        text = serialize_training_example(
            example,
            summary_mode=self._summary_mode,
        )
        token_ids = self._tokenizer.encode(text)

        if len(token_ids) > self._max_seq_len:
            token_ids = token_ids[: self._max_seq_len]

        if not token_ids:
            token_ids = [self._tokenizer.vocab["[unk]"]]

        if token_ids[0] != self._cls_idx:
            token_ids = [self._cls_idx] + token_ids[: self._max_seq_len - 1]

        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "score": torch.tensor(example.score, dtype=torch.float32),
        }


def serialize_training_example(example, *, summary_mode: str) -> str:
    """Serialize one labeled example under the run-wide summary contract.

    EconomicScenario is the active training format and goes through the same
    ContextBundle bridge as evaluation.  The unstructured legacy
    TrainingExample format remains supported only for full-summary runs;
    structured-only runs fail closed instead of dropping economic fields.
    """
    mode = normalize_summary_mode(summary_mode)
    if isinstance(example, EconomicScenario):
        return serialize_scenario(example, summary_mode=mode)

    serialized = example.to_serialized()
    if mode == SUMMARY_MODE_FULL:
        return serialized
    raise ValueError(
        "summary_mode='none' requires EconomicScenario structured fields; "
        "legacy TrainingExample would lose event type/direction/magnitude"
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


def apply_mlm(
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_prob: float = 0.15,
    *,
    cls_idx: int = 1,
    sep_idx: int = 2,
    mask_idx: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply masked-language-model masking.

    Returns: (masked_ids, labels, mlm_mask)
    labels is -100 where no masking occurred (ignored by loss).
    """
    labels = torch.full_like(token_ids, -100)
    mlm_mask = torch.zeros_like(token_ids, dtype=torch.bool)

    mask_candidates = attention_mask.bool() & (token_ids != cls_idx) & (token_ids != sep_idx)

    rand = torch.rand_like(token_ids.float())
    selected = (rand < mask_prob) & mask_candidates

    labels[selected] = token_ids[selected]
    mlm_mask = selected

    masked_ids = token_ids.clone()
    masked_ids[selected] = mask_idx

    return masked_ids, labels, mlm_mask


class MLMHead(nn.Module):
    """Language modeling head for pretraining."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
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
    from src.models.tokenizer import BPETokenizer

    tokenizer = BPETokenizer.train(texts, vocab_size=vocab_size, verbose=verbose)
    return tokenizer.vocab


def build_tokenizer(
    texts: Sequence[str],
    vocab_size: int = 24_000,
    verbose: bool = False,
) -> "BPETokenizer":
    """Train and return a BPE tokenizer."""
    from src.models.tokenizer import BPETokenizer

    return BPETokenizer.train(texts, vocab_size=vocab_size, verbose=verbose)


def save_checkpoint(
    model: nn.Module,
    tokenizer: "BPETokenizer",
    config: TrainingConfig,
    path: Path,
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
    }
    validate_checkpoint_token_ids(
        checkpoint_config,
        tokenizer.vocab,
        require_explicit=True,
    )
    validate_checkpoint_versions(checkpoint_config, config.summary_mode)
    validate_checkpoint_sequence_length(checkpoint_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "serializer_contract_version": SERIALIZER_CONTRACT_VERSION,
            "model_state": model.state_dict(),
            "vocab": tokenizer.vocab,
            "merges": tokenizer.merges,
            "config": checkpoint_config,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    *,
    legacy_summary_mode: str | None = None,
) -> tuple[EconomyEncoder, "BPETokenizer", dict[str, object]]:
    """Load a checkpoint and return (model, tokenizer, config).

    Legacy artifacts fail closed unless a verified full-summary contract is
    declared explicitly.  Old no-summary artifacts require retraining because
    they used a different serialized marker.
    """
    from src.models.tokenizer import BPETokenizer

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


def create_causal_lr_schedule(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup followed by cosine decay."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def pretrain(
    model: EconomyEncoder,
    texts: Sequence[str],
    tokenizer: "BPETokenizer",
    config: TrainingConfig,
    device: torch.device | None = None,
) -> dict[str, list[float]]:
    """Phase 1: Pretrain EconomyEncoder with MLM.

    Returns loss history for logging.
    """
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

    cls_idx = model.cls_idx
    sep_idx = model.sep_idx
    mask_idx = tokenizer.vocab.get("[mask]", 4)

    model.train()
    mlm_head.train()
    losses: list[float] = []

    step = 0
    for epoch in range(config.max_epochs):
        for batch in loader:
            token_ids = batch["token_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            masked_ids, labels, _ = apply_mlm(
                token_ids,
                attention_mask,
                cls_idx=cls_idx,
                sep_idx=sep_idx,
                mask_idx=mask_idx,
            )

            optimizer.zero_grad()

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

    return {"losses": losses}


def train_scores(
    model: EconomyEncoder,
    examples: Sequence,
    tokenizer: "BPETokenizer",
    config: TrainingConfig,
    device: torch.device | None = None,
) -> dict[str, list[float]]:
    """Phase 2: Train EconomyEncoder on score labels with SmoothL1Loss.

    Returns loss history for logging.
    """
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

    loss_fn = nn.SmoothL1Loss()
    model.train()
    losses: list[float] = []

    step = 0
    for epoch in range(config.max_epochs):
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

    return {"losses": losses}


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

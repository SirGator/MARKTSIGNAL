"""Command-line entry point for training: python -m training train"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .modeling.context_serializer import (
    SUMMARY_MODE_FULL,
    SUMMARY_MODE_NONE,
    normalize_summary_mode,
)

from .bridge import serialize_scenario
from .scenarios import (
    EconomicScenario,
    generate_counterexample_groups,
    generate_parametric,
)

if TYPE_CHECKING:
    from .pipeline import TrainingConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="training",
        description="Train EconomyEncoder V1 (offline, separate from production).",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    train_cmd = subcommands.add_parser("train", help="run pretraining + score training")
    train_cmd.add_argument("--epochs", type=int, default=10, help="epochs per phase")
    train_cmd.add_argument("--batch-size", type=int, default=32)
    train_cmd.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    train_cmd.add_argument("--max-seq-len", type=int, default=512)
    train_cmd.add_argument(
        "--vocab-size",
        type=int,
        default=4000,
        help="BPE vocabulary size; note the MODEL_SPEC reference architecture "
        "uses 24.000 tokens (~23.4M parameters) while the CLI default of "
        "4.000 yields the compact ~15.7M variant",
    )
    train_cmd.add_argument("--num-scenarios", type=int, default=5000, help="parametric scenarios (legacy mode)")
    train_cmd.add_argument("--num-counter-groups", type=int, default=200)
    train_cmd.add_argument("--seed", type=int, default=42)
    train_cmd.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="path to a frozen dataset directory (e.g. data/dataset_v1). "
        "When set, train/validation/OOD splits are loaded from there instead "
        "of generating parametric scenarios on the fly.",
    )
    train_cmd.add_argument("--output", type=Path, default=Path(".model_checkpoints/economy_encoder_v1.pt"))
    train_cmd.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        metavar="EPOCHS",
        help="save an inference-ready snapshot after every N epochs in each phase (default: 1)",
    )
    train_cmd.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="continue exactly from a resumable periodic checkpoint",
    )
    train_cmd.add_argument(
        "--keep-checkpoints",
        type=int,
        default=3,
        metavar="COUNT",
        help="keep only the newest COUNT periodic snapshots per phase (default: 3)",
    )
    train_cmd.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        metavar="EPOCHS",
        help="stop score training after this many non-improving epochs; 0 disables it",
    )
    train_cmd.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="minimum validation-MAE improvement (default: 0.0001)",
    )
    train_cmd.add_argument("--skip-pretrain", action="store_true", help="skip MLM pretraining")
    train_cmd.add_argument("--device", default="auto", help="cpu | cuda | auto")
    train_cmd.add_argument("--log-every", type=int, default=50, help="log loss every N steps")
    train_cmd.add_argument("--val-split", type=float, default=0.1, help="fraction held out for validation (legacy mode)")
    train_cmd.add_argument("--num-paraphrases", type=int, default=3, help="paraphrase variants per scenario (legacy mode)")
    train_cmd.add_argument("--neutralize-ratio", type=float, default=0.25, help="fraction with [NO_SUMMARY] (legacy mode)")
    train_cmd.add_argument(
        "--runtime-noise-ratio",
        type=float,
        default=0.2,
        help="fraction of training samples duplicated with mild article-like noise (default: 0.2)",
    )
    train_cmd.add_argument("--no-summary", action="store_true", help="ablation: replace ALL summaries with [NO_SUMMARY]")

    subcommands.add_parser("info", help="show training data statistics")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()

    args = parser.parse_args(argv)

    if args.command == "train" and args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be at least 1")
    if args.command == "train" and args.keep_checkpoints < 1:
        parser.error("--keep-checkpoints must be at least 1")
    if args.command == "train" and not 0.0 <= args.runtime_noise_ratio <= 1.0:
        parser.error("--runtime-noise-ratio must be between 0 and 1")
    if args.command == "train" and args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience must be non-negative")
    if args.command == "train" and (
        not math.isfinite(args.early_stopping_min_delta)
        or args.early_stopping_min_delta < 0.0
    ):
        parser.error("--early-stopping-min-delta must be finite and non-negative")

    if args.command == "info":
        scenarios = generate_parametric(num=100, seed=42)
        print(f"Parametrische Szenarien (sample): {len(scenarios)}")
        roles = set(s.case_role for s in scenarios)
        horizons = set(s.horizon_days for s in scenarios)
        print(f"Rollen: {sorted(roles)}")
        print(f"Horizonte: {sorted(horizons)}")
        print()
        print("Production importiert niemals aus training/.")
        print("Die einzige Verbindung ist der Checkpoint (.pt).")
        return 0

    if args.command == "train":
        return _run_train(args)

    return 0


def _summary_mode_from_args(args: argparse.Namespace) -> str:
    return SUMMARY_MODE_NONE if args.no_summary else SUMMARY_MODE_FULL


def _training_config_from_args(
    args: argparse.Namespace,
    *,
    vocab_size: int,
    frozen=None,
) -> TrainingConfig:
    """Build the persisted run config from the parsed CLI contract."""
    from .pipeline import TrainingConfig

    manifest = None if frozen is None else frozen.manifest
    if frozen is None:
        from .paraphrases import LEGACY_RUNTIME_NOISE_VERSION

        noise_profile_version = LEGACY_RUNTIME_NOISE_VERSION
    else:
        from .runtime_views import RUNTIME_VIEW_VERSION

        noise_profile_version = RUNTIME_VIEW_VERSION
    return TrainingConfig(
        model_dir=args.output.parent,
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        log_every=args.log_every,
        summary_mode=_summary_mode_from_args(args),
        seed=args.seed,
        dataset_version=(
            None if manifest is None else str(manifest["dataset_version"])
        ),
        dataset_schema_version=(
            None if manifest is None else str(manifest["schema_version"])
        ),
        dataset_generator_version=(
            None if manifest is None else str(manifest["generator_version"])
        ),
        dataset_seed=(None if manifest is None else int(manifest["seed"])),
        runtime_noise_ratio=args.runtime_noise_ratio,
        runtime_noise_profile_version=noise_profile_version,
    )


def _periodic_checkpoint_path(
    output: Path,
    *,
    phase: str,
    completed_epochs: int,
) -> Path:
    """Build a stable snapshot path without replacing the final checkpoint."""

    if phase not in {"pretrain", "score"}:
        raise ValueError(f"unknown checkpoint phase: {phase!r}")
    if isinstance(completed_epochs, bool) or not isinstance(completed_epochs, int):
        raise TypeError("completed_epochs must be an integer")
    if completed_epochs < 1:
        raise ValueError("completed_epochs must be a positive integer")

    checkpoint_dir = output.with_suffix(".checkpoints")
    return checkpoint_dir / f"{phase}_epoch_{completed_epochs:03d}.pt"


def _prune_periodic_checkpoints(
    output: Path,
    *,
    phase: str,
    keep: int,
) -> list[Path]:
    """Delete only obsolete periodic snapshots for one training phase."""

    if phase not in {"pretrain", "score"}:
        raise ValueError(f"unknown checkpoint phase: {phase!r}")
    if isinstance(keep, bool) or not isinstance(keep, int):
        raise TypeError("keep must be an integer")
    if keep < 1:
        raise ValueError("keep must be at least 1")

    checkpoint_dir = output.with_suffix(".checkpoints")
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob(f"{phase}_epoch_*.pt"):
        try:
            epoch = int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        candidates.append((epoch, path))
    candidates.sort(key=lambda item: (item[0], item[1].name))

    removed: list[Path] = []
    for _, path in candidates[:-keep]:
        path.unlink()
        removed.append(path)
    return removed


def _serialization_corpora(
    scenarios: list[object],
    *,
    summary_mode: str,
) -> tuple[list[str], list[str]]:
    """Return (shared tokenizer corpus, run-specific model corpus)."""

    mode = normalize_summary_mode(summary_mode)
    tokenizer_texts = [
        _serialize_evaluation_example(scenario, summary_mode=SUMMARY_MODE_FULL)
        for scenario in scenarios
    ]
    model_texts = [
        _serialize_evaluation_example(scenario, summary_mode=mode)
        for scenario in scenarios
    ]
    return tokenizer_texts, model_texts


def _seed_training(seed: int) -> None:
    """Seed model initialization consistently across ablation arms."""

    import torch

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _encode_evaluation_text(model, tokenizer, text: str) -> list[int]:
    """Apply the same sequence-length boundary as training and production."""

    max_seq_len = getattr(model, "max_seq_len", None)
    if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, int):
        raise TypeError("evaluation model.max_seq_len must be an integer")
    if max_seq_len < 1:
        raise ValueError("evaluation model.max_seq_len must be positive")
    token_ids = tokenizer.encode(text)[:max_seq_len]
    if not token_ids:
        raise ValueError("evaluation input produced no tokens")
    return token_ids


def _load_training_data(args):
    """Return (train_examples, val_examples, iid_test_examples, ood_splits) for one run.

    When ``--dataset`` points at a frozen dataset directory, the splits are
    loaded verbatim from there and the legacy parametric generator is
    bypassed.  Otherwise the legacy on-the-fly generator is used, preserving
    the existing behaviour for smoke tests and ablations that do not need a
    frozen artifact.
    """

    summary_mode = _summary_mode_from_args(args)

    if args.dataset is not None:
        from .data import load_frozen_dataset_for_training

        print(f"=== 1. Frozen Dataset laden: {args.dataset} ===")
        frozen = load_frozen_dataset_for_training(args.dataset)
        manifest = frozen.manifest
        print(
            f"Dataset: {manifest.get('dataset_version')} "
            f"(schema {manifest.get('schema_version')}, "
            f"generator {manifest.get('generator_version')}, "
            f"seed {manifest.get('seed')})"
        )
        train_examples = list(frozen.train)
        val_examples = list(frozen.validation)
        iid_test_examples = list(frozen.iid_test)
        ood_splits = dict(frozen.ood_splits)
        ood_total = sum(len(records) for records in ood_splits.values())
        print(f"Train: {len(train_examples)}, Validation: {len(val_examples)}")
        if iid_test_examples:
            print(f"Test (IID): {len(iid_test_examples)}")
        if ood_total:
            print(f"Test (OOD): {ood_total}")
        from .runtime_views import add_frozen_runtime_views

        clean_count = len(train_examples)
        train_examples = add_frozen_runtime_views(
            train_examples,
            ratio=args.runtime_noise_ratio,
            seed=args.seed + 3,
        )
        noisy_count = len(train_examples) - clean_count
        if noisy_count:
            print(f"Runtime-nahe Frozen-Views: +{noisy_count}")
        # The frozen artifact stays immutable; views exist only in this train
        # split and never enter validation or OOD evaluation.
        if summary_mode == SUMMARY_MODE_NONE:
            print("ABLATION: Alle Summaries durch [NO_SUMMARY] ersetzt")
        return train_examples, val_examples, iid_test_examples, ood_splits, summary_mode, frozen

    print("=== 1. Daten generieren (Legacy-Modus) ===")
    scenarios = generate_parametric(num=args.num_scenarios, seed=args.seed)
    counter = generate_counterexample_groups(num_groups=args.num_counter_groups, seed=args.seed + 1)
    all_scenarios = scenarios + counter

    val_count = int(len(all_scenarios) * args.val_split)
    val_scenarios = all_scenarios[:val_count]
    train_scenarios = all_scenarios[val_count:]

    from .paraphrases import add_runtime_noise, expand_with_paraphrases
    # Expand before applying the run-wide serializer so full and none see the
    # exact same sample order, labels and number of optimizer steps.
    train_scenarios = expand_with_paraphrases(
        train_scenarios,
        num_paraphrases=args.num_paraphrases,
        neutralize_ratio=args.neutralize_ratio,
        seed=args.seed + 2,
    )
    clean_count = len(train_scenarios)
    train_scenarios = add_runtime_noise(
        train_scenarios,
        ratio=args.runtime_noise_ratio,
        seed=args.seed + 3,
    )
    noisy_count = len(train_scenarios) - clean_count
    if noisy_count:
        print(f"Runtime-nahe Textvarianten: +{noisy_count}")
    if summary_mode == SUMMARY_MODE_NONE:
        print("ABLATION: Alle Summaries durch [NO_SUMMARY] ersetzt")

    print(f"Training (expandiert): {len(train_scenarios)}, Validation: {len(val_scenarios)}")
    # Legacy mode has no frozen splits; evaluation falls back to the synthetic
    # OOD tests in training/ood_tests.py and uses the held-out validation slice
    # as the IID test signal.
    return train_scenarios, val_scenarios, [], {}, summary_mode, None


def _run_train(args) -> int:
    import torch

    from .pipeline import (
        build_tokenizer,
        pretrain,
        save_checkpoint,
        load_training_checkpoint,
        train_scores,
    )

    # Identical CLI seeds produce identical initialization and loader seeds in
    # both ablation arms.  The serialized summary mode is the intended change.
    _seed_training(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print()

    (
        train_examples,
        val_examples,
        iid_test_examples,
        ood_splits,
        summary_mode,
        frozen,
    ) = _load_training_data(args)

    # Both ablation arms learn exactly the same BPE contract from the full
    # corpus.  Otherwise ``none`` would get a smaller vocabulary/model and the
    # experiment would change two variables at once.
    tokenizer_texts, texts = _serialization_corpora(
        train_examples,
        summary_mode=summary_mode,
    )
    print()

    print("=== 2. BPE Tokenizer ===")
    tokenizer = build_tokenizer(
        tokenizer_texts,
        vocab_size=args.vocab_size,
        verbose=True,
    )
    print(f"Vokabular: {tokenizer.vocab_size} Tokens")
    print(f"Tokenizer: {tokenizer.fingerprint}")
    print()

    print("=== 3. Modell ===")
    config = _training_config_from_args(
        args,
        vocab_size=tokenizer.vocab_size,
        frozen=frozen,
    )
    resume_state: dict[str, object] | None = None
    resume_phase: str | None = None
    if args.resume is not None:
        model, resumed_tokenizer, _, resume_state = load_training_checkpoint(
            args.resume
        )
        if resumed_tokenizer.fingerprint != tokenizer.fingerprint:
            raise ValueError(
                "resume checkpoint tokenizer differs from the current dataset"
            )
        tokenizer = resumed_tokenizer
        resume_phase = str(resume_state["phase"])
        if resume_phase == "pretrain" and args.skip_pretrain:
            raise ValueError(
                "--skip-pretrain conflicts with a pretrain resume checkpoint"
            )
        print(f"Resume: {args.resume} (Phase: {resume_phase})")
    else:
        from .modeling.model import EconomyEncoder

        vocab = tokenizer.vocab
        model = EconomyEncoder(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=config.max_seq_len,
            pad_idx=vocab.get("[pad]", 0),
            cls_idx=vocab.get("[cls]", 1),
            sep_idx=vocab.get("[sep]", 2),
        )
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameter: {params/1e6:.2f}M")
    print()

    def save_periodic_checkpoint(
        phase: str,
        completed_epochs: int,
        training_state: dict[str, object],
    ) -> None:
        path = _periodic_checkpoint_path(
            args.output,
            phase=phase,
            completed_epochs=completed_epochs,
        )
        save_checkpoint(
            model,
            tokenizer,
            config,
            path,
            training_state=training_state,
        )
        removed = _prune_periodic_checkpoints(
            args.output,
            phase=phase,
            keep=args.keep_checkpoints,
        )
        print(
            "Zwischencheckpoint: "
            f"{path} (Phase: {phase}, abgeschlossene Epochen: {completed_epochs})"
        )
        if removed:
            print(
                "Checkpoint-Aufbewahrung: entfernt "
                + ", ".join(item.name for item in removed)
            )

    def save_best_checkpoint(completed_epoch: int, validation_mae: float) -> None:
        path = args.output.with_suffix(".checkpoints") / "best.pt"
        save_checkpoint(model, tokenizer, config, path)
        print(
            "Bester Checkpoint: "
            f"{path} (Epoche: {completed_epoch}, MAE: {validation_mae:.4f})"
        )


    if not args.skip_pretrain and resume_phase != "score":
        print("=== 4. Pretraining (MLM) ===")
        t0 = time.time()
        pretrain(
            model,
            texts,
            tokenizer,
            config,
            device=device,
            checkpoint_every_epochs=args.checkpoint_every,
            resume_checkpoint_callback=save_periodic_checkpoint,
            resume_state=(
                resume_state if resume_phase == "pretrain" else None
            ),
        )
        print(f"Pretraining: {time.time()-t0:.1f}s")
        print()

    print("=== 5. Score-Training ===")
    t0 = time.time()
    training_result = train_scores(
        model,
        train_examples,
        tokenizer,
        config,
        device=device,
        checkpoint_every_epochs=args.checkpoint_every,
        resume_checkpoint_callback=save_periodic_checkpoint,
        resume_state=resume_state if resume_phase == "score" else None,
        validation_examples=val_examples,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        evaluation_batch_size=args.batch_size,
        best_checkpoint_callback=save_best_checkpoint if val_examples else None,
    )
    print(f"Score-Training: {time.time()-t0:.1f}s")
    print()
    if training_result["best_epoch"] is not None:
        print(
            "Bestes Validierungsmodell: "
            f"Epoche {training_result['best_epoch']}, "
            f"MAE {training_result['best_mae']:.4f}"
        )

    print("=== 6. Checkpoint ===")
    save_checkpoint(model, tokenizer, config, args.output)
    print(f"Gespeichert: {args.output}")
    print()

    _evaluate(
        model,
        tokenizer,
        val_examples,
        device,
        summary_mode=config.summary_mode,
        iid_test_examples=iid_test_examples,
        ood_splits=ood_splits,
        batch_size=args.batch_size,
    )
    return 0


def _serialize_evaluation_example(example, *, summary_mode: str) -> str:
    """Serialize either an EconomicScenario or a DatasetRecord for eval.

    Both types carry a ``.score`` attribute and share the production
    ContextSerializer path via training.bridge.
    """

    from .bridge import serialize_record

    try:
        from dataset.schema import DatasetRecord
    except ModuleNotFoundError:
        DatasetRecord = None  # type: ignore[assignment]

    if DatasetRecord is not None and isinstance(example, DatasetRecord):
        return serialize_record(example, summary_mode=summary_mode)
    return serialize_scenario(example, summary_mode=summary_mode)


def _example_score(example) -> float:
    return float(example.score)


def _predict_abs_error(model, tokenizer, example, device, *, summary_mode: str) -> float:
    import torch

    text = _serialize_evaluation_example(example, summary_mode=summary_mode)
    token_ids = _encode_evaluation_text(model, tokenizer, text)
    t = torch.tensor([token_ids], dtype=torch.long, device=device)
    a = torch.ones_like(t)
    with torch.inference_mode():
        out = model(t, a)
    pred = out["score"].item()
    return abs(pred - _example_score(example))


def _predict_examples(
    model,
    tokenizer,
    examples,
    device,
    *,
    summary_mode: str,
    batch_size: int = 64,
) -> tuple[list[object], list[float], list[float]]:
    """Run one complete evaluation split through the batched predictor."""

    from .pipeline import predict_scores

    items = list(examples)
    targets = [_example_score(example) for example in items]
    predictions = predict_scores(
        model,
        items,
        tokenizer,
        max_seq_len=model.max_seq_len,
        summary_mode=summary_mode,
        batch_size=batch_size,
        device=device,
    )
    if len(predictions) != len(items):
        raise RuntimeError("prediction count does not match evaluation examples")
    return items, targets, predictions


def _print_score_metrics(
    label: str,
    targets: list[float],
    predictions: list[float],
) -> None:
    """Print regression, direction and strong-impact metrics."""

    from .metrics import compute_score_metrics

    metrics = compute_score_metrics(targets, predictions)
    strong = (
        "--" if metrics.strong_mae is None else f"{metrics.strong_mae:.4f}"
    )
    print(
        f"{label} MAE: {metrics.mae:.4f} | "
        f"Richtung: {metrics.sign_accuracy:.1%} | "
        f"Stark-MAE: {strong} (strong_n={metrics.strong_count}, n={metrics.count})"
    )


def _evaluation_dimensions(example, target: float) -> dict[str, str]:
    """Return stable diagnostic slice labels for either training schema."""

    event = getattr(example, "event", None)
    mechanism = getattr(event, "mechanism", None) or getattr(
        example, "event_type", "unknown"
    )
    direction = getattr(event, "direction", None) or getattr(
        example, "direction", "unknown"
    )
    horizon_days = int(getattr(example, "horizon_days"))
    if horizon_days <= 30:
        horizon = "<=30d"
    elif horizon_days <= 90:
        horizon = "31-90d"
    else:
        horizon = ">90d"
    absolute_target = abs(target)
    if absolute_target <= 0.05:
        impact = "neutral"
    elif absolute_target < 0.5:
        impact = "moderate"
    else:
        impact = "strong"

    dimensions = {
        "mechanism": str(mechanism),
        "direction": str(direction),
        "horizon": horizon,
        "impact": impact,
    }
    metadata = getattr(example, "metadata", None)
    if metadata is not None:
        dimensions["source"] = str(getattr(metadata, "source", "unknown"))
        dimensions["counterfactual"] = (
            "variant"
            if getattr(metadata, "counterfactual", None) is not None
            else "base_or_standalone"
        )
    return dimensions


def _print_evaluation_slices(
    examples: list[object],
    targets: list[float],
    predictions: list[float],
) -> None:
    """Print metrics by mechanism, direction, horizon and impact band."""

    from .metrics import compute_score_metrics

    groups: dict[tuple[str, str], list[int]] = {}
    for index, (example, target) in enumerate(zip(examples, targets)):
        for dimension, value in _evaluation_dimensions(example, target).items():
            groups.setdefault((dimension, value), []).append(index)
    if not groups:
        return

    print("  Slices (* = n < 20):")
    for (dimension, value), indices in sorted(groups.items()):
        slice_metrics = compute_score_metrics(
            (targets[index] for index in indices),
            (predictions[index] for index in indices),
        )
        small = "*" if slice_metrics.count < 20 else " "
        print(
            f"  {small} {dimension}={value}: MAE={slice_metrics.mae:.4f}, "
            f"Richtung={slice_metrics.sign_accuracy:.1%}, n={slice_metrics.count}"
        )


# Mapping from a frozen OOD split name to the predicate that identifies the
# records which actually satisfy that split's OOD condition.  Records held in
# the split only because their counterfactual family was held out (but which do
# not themselves trigger the OOD rule) are excluded from the metric so each
# OOD axis measures generalization, not family dilution.
_OOD_METRIC_PREDICATES: dict[str, str] = {
    "test_parameter_ood": "parameter",
    "test_combination_ood": "combination",
    "test_hard_ood": "hard",
    "test_concept_ood": "concept",
    "test_entity_ood": "entity",
}


def _ood_predicate_for_split(
    split_name: str,
    records: tuple,
):
    """Return a callable ``(record) -> bool`` selecting the metric-relevant
    records of one OOD split, or ``None`` to include all records."""

    from dataset.splits import (
        is_combination_ood,
        is_concept_ood,
        is_hard_ood,
        is_parameter_ood,
    )

    kind = _OOD_METRIC_PREDICATES.get(split_name)
    if kind == "parameter":
        return is_parameter_ood
    if kind == "combination":
        return is_combination_ood
    if kind == "hard":
        return is_hard_ood
    if kind == "concept":
        return is_concept_ood
    if kind == "entity":
        # Entity-OOD: every record in the split is out-of-distribution by
        # construction (the whole entity is held out), so no extra filter.
        return None
    return None


def _evaluate(
    model,
    tokenizer,
    val_scenarios,
    device,
    *,
    summary_mode: str = SUMMARY_MODE_FULL,
    iid_test_examples: list | None = None,
    ood_splits: dict[str, tuple] | None = None,
    batch_size: int = 64,
) -> None:
    import torch

    summary_mode = normalize_summary_mode(summary_mode)
    model.eval()

    print("=== 7. Validation (IID) ===")
    if val_scenarios:
        evaluated, targets, predictions = _predict_examples(
            model,
            tokenizer,
            val_scenarios,
            device,
            summary_mode=summary_mode,
            batch_size=batch_size,
        )
        _print_score_metrics("IID", targets, predictions)
        _print_evaluation_slices(evaluated, targets, predictions)
    print()

    # IID test split is reported separately and never folded into the OOD
    # aggregate.  In frozen-dataset mode it comes from ``test_iid``; in legacy
    # mode there is no held-out IID test slice and we simply skip the block.
    if iid_test_examples:
        print("=== 7b. Test (IID) ===")
        evaluated, targets, predictions = _predict_examples(
            model,
            tokenizer,
            iid_test_examples,
            device,
            summary_mode=summary_mode,
            batch_size=batch_size,
        )
        _print_score_metrics("Test IID", targets, predictions)
        _print_evaluation_slices(evaluated, targets, predictions)
        print()

    print("=== 8. OOD-Tests ===")
    by_category: dict[str, tuple[list[float], list[float]]] = {}

    if not ood_splits:
        # Legacy mode: batch the synthetic OOD suite by category.
        from .ood_tests import all_ood_tests

        scenarios_by_category: dict[str, list[object]] = {}
        for test in all_ood_tests():
            category = test.category
            if summary_mode == SUMMARY_MODE_NONE and category == "OOD_LANGUAGE":
                category = "STRUCTURED_OOD (source: OOD_LANGUAGE)"
            scenarios_by_category.setdefault(category, []).append(test.scenario)
        for category, scenarios in scenarios_by_category.items():
            _, targets, predictions = _predict_examples(
                model,
                tokenizer,
                scenarios,
                device,
                summary_mode=summary_mode,
                batch_size=batch_size,
            )
            by_category[category] = (targets, predictions)

        if summary_mode == SUMMARY_MODE_NONE:
            print(
                "Hinweis: OOD_LANGUAGE enthält ohne Summary keinen Sprach-OOD; "
                "Ausgabe daher als STRUCTURED_OOD mit Quellkategorie OOD_LANGUAGE."
            )
    else:
        # Frozen splits are filtered to records satisfying the actual OOD rule.
        for split_name in (
            "test_entity_ood",
            "test_parameter_ood",
            "test_combination_ood",
            "test_hard_ood",
            "test_concept_ood",
        ):
            examples = ood_splits.get(split_name)
            if not examples:
                continue
            predicate = _ood_predicate_for_split(split_name, examples)
            metric_examples = (
                examples
                if predicate is None
                else tuple(record for record in examples if predicate(record))
            )
            held_out_only = len(examples) - len(metric_examples)
            label = split_name.upper()
            if held_out_only:
                label = f"{label} (held-out-only: {held_out_only})"
            if metric_examples:
                _, targets, predictions = _predict_examples(
                    model,
                    tokenizer,
                    metric_examples,
                    device,
                    summary_mode=summary_mode,
                    batch_size=batch_size,
                )
                by_category[label] = (targets, predictions)

    from .metrics import compute_score_metrics

    print(
        f"{'Kategorie':45s} | {'MAE':>6s} | {'Richtung':>9s} | "
        f"{'Stark-MAE':>9s} | {'n':>4s}"
    )
    print("-" * 86)
    all_targets: list[float] = []
    all_predictions: list[float] = []
    for category, (targets, predictions) in sorted(by_category.items()):
        metrics = compute_score_metrics(targets, predictions)
        strong = (
            "--" if metrics.strong_mae is None else f"{metrics.strong_mae:.4f}"
        )
        print(
            f"{category:45s} | {metrics.mae:.4f} | "
            f"{metrics.sign_accuracy:8.1%} | {strong:>9s} | {metrics.count:4d}"
        )
        all_targets.extend(targets)
        all_predictions.extend(predictions)
    print("-" * 86)
    if all_targets:
        metrics = compute_score_metrics(all_targets, all_predictions)
        strong = (
            "--" if metrics.strong_mae is None else f"{metrics.strong_mae:.4f}"
        )
        print(
            f"{'OOD gesamt':45s} | {metrics.mae:.4f} | "
            f"{metrics.sign_accuracy:8.1%} | {strong:>9s} | {metrics.count:4d}"
        )
    print()

    if ood_splits:
        return

    print("=== 9. Score-Tests ===")
    from .scenarios import _compute_score, _build_context, EconomicScenario

    test_cases = [
        ("Oil prices surged by 30 percent after supply disruptions", "airline", "consumer", "input_price_change", "increase", 30, 35, 0.1, 0.2, 0.3, 30, "commodity:oil", "percent"),
        ("Oil prices surged by 30 percent after supply disruptions", "oil_producer", "producer", "input_price_change", "increase", 30, 80, 0.0, 0.5, 0.5, 30, "commodity:oil", "percent"),
        ("Oil prices surged by 30 percent after supply disruptions", "software", "neutral", "input_price_change", "increase", 30, 0, 0, 0, 0, 30, "commodity:oil", "percent"),
        ("Jet fuel became much more expensive", "airline", "consumer", "input_price_change", "increase", 30, 33, 0.1, 0.2, 0.3, 30, "commodity:oil", "percent"),
        ("A major customer terminated its contract immediately", "company", "supplier_high", "customer_loss", "loss", 0, 27, 0, 0, 0.2, 30, "customer:major_customer", "none"),
        ("A major customer terminated its contract immediately", "company", "supplier_low", "customer_loss", "loss", 0, 5, 0, 0, 0.8, 30, "customer:major_customer", "none"),
        ("The central bank raised interest rates by 50 basis points", "bank", "bank", "interest_rate_change", "hike", 50, 50, 0, 0.3, 0, 90, "macro:interest_rate", "basis_points"),
        ("The central bank raised interest rates by 50 basis points", "real_estate", "real_estate", "interest_rate_change", "hike", 50, 60, 0, 0.3, 0, 90, "macro:interest_rate", "basis_points"),
    ]

    print(f"{'Event':50s} | {'Role':14s} | {'Ziel':>6s} | {'Modellwert':>10s} | {'Δ':>7s}")
    print("-" * 100)
    for ev, case_type, role, et, d, m, e, h, pp, s, hd, subject, mag_unit in test_cases:
        target = _compute_score(et, d, role, m, e, h, pp, s, hd)
        commodity = subject.split(":", 1)[-1] if ":" in subject else ""
        context_text = _build_context(event_type=et, role=role, commodity=commodity, exposure=e, hedging=h, pricing_power=pp, substitution=s)
        scenario = EconomicScenario(
            event_type=et, direction=d, event_text=ev,
            case_role=role, case_type=case_type, context_text=context_text,
            magnitude=m, exposure=e, hedging=h, pricing_power=pp, substitution=s,
            horizon_days=hd, score=target,
            subject=subject, magnitude_unit=mag_unit,
        )
        text = serialize_scenario(scenario, summary_mode=summary_mode)
        token_ids = _encode_evaluation_text(model, tokenizer, text)
        t = torch.tensor([token_ids], dtype=torch.long, device=device)
        a = torch.ones_like(t)
        with torch.inference_mode():
            out = model(t, a)
        pred = out["score"].item()
        delta = pred - target
        print(f"{ev[:50]:50s} | {role:14s} | {target:+.2f} | {pred:+.4f}   | {delta:+.4f}")
    print("-" * 100)


if __name__ == "__main__":
    raise SystemExit(main())

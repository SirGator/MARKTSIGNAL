"""Command-line entry point for training: python -m training train"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING

from src.models.context_serializer import (
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
    train_cmd.add_argument("--vocab-size", type=int, default=4000)
    train_cmd.add_argument("--num-scenarios", type=int, default=5000, help="parametric scenarios")
    train_cmd.add_argument("--num-counter-groups", type=int, default=200)
    train_cmd.add_argument("--seed", type=int, default=42)
    train_cmd.add_argument("--output", type=Path, default=Path(".model_checkpoints/economy_encoder_v1.pt"))
    train_cmd.add_argument("--skip-pretrain", action="store_true", help="skip MLM pretraining")
    train_cmd.add_argument("--device", default="auto", help="cpu | cuda | auto")
    train_cmd.add_argument("--log-every", type=int, default=50, help="log loss every N steps")
    train_cmd.add_argument("--val-split", type=float, default=0.1, help="fraction held out for validation")
    train_cmd.add_argument("--num-paraphrases", type=int, default=3, help="paraphrase variants per scenario")
    train_cmd.add_argument("--neutralize-ratio", type=float, default=0.25, help="fraction with [NO_SUMMARY]")
    train_cmd.add_argument("--no-summary", action="store_true", help="ablation: replace ALL summaries with [NO_SUMMARY]")

    subcommands.add_parser("info", help="show training data statistics")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()

    args = parser.parse_args(argv)

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
) -> TrainingConfig:
    """Build the persisted run config from the parsed CLI contract."""
    from .pipeline import TrainingConfig

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
    )


def _serialization_corpora(
    scenarios: list[EconomicScenario],
    *,
    summary_mode: str,
) -> tuple[list[str], list[str]]:
    """Return (shared tokenizer corpus, run-specific model corpus)."""

    mode = normalize_summary_mode(summary_mode)
    tokenizer_texts = [
        serialize_scenario(scenario, summary_mode=SUMMARY_MODE_FULL)
        for scenario in scenarios
    ]
    model_texts = [
        serialize_scenario(scenario, summary_mode=mode)
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


def _run_train(args) -> int:
    import torch

    from .pipeline import (
        build_tokenizer,
        pretrain,
        save_checkpoint,
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

    print("=== 1. Daten generieren ===")
    scenarios = generate_parametric(num=args.num_scenarios, seed=args.seed)
    counter = generate_counterexample_groups(num_groups=args.num_counter_groups, seed=args.seed + 1)
    all_scenarios = scenarios + counter

    val_count = int(len(all_scenarios) * args.val_split)
    val_scenarios = all_scenarios[:val_count]
    train_scenarios = all_scenarios[val_count:]
    summary_mode = _summary_mode_from_args(args)

    from .paraphrases import expand_with_paraphrases
    # Expand before applying the run-wide serializer so full and none see the
    # exact same sample order, labels and number of optimizer steps.
    train_scenarios = expand_with_paraphrases(
        train_scenarios,
        num_paraphrases=args.num_paraphrases,
        neutralize_ratio=args.neutralize_ratio,
        seed=args.seed + 2,
    )
    if summary_mode == SUMMARY_MODE_NONE:
        print("ABLATION: Alle Summaries durch [NO_SUMMARY] ersetzt")

    print(f"Training (expandiert): {len(train_scenarios)}, Validation: {len(val_scenarios)}")

    # Both ablation arms learn exactly the same BPE contract from the full
    # corpus.  Otherwise ``none`` would get a smaller vocabulary/model and the
    # experiment would change two variables at once.
    tokenizer_texts, texts = _serialization_corpora(
        train_scenarios,
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
    )
    from src.models.model import EconomyEncoder

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

    if not args.skip_pretrain:
        print("=== 4. Pretraining (MLM) ===")
        t0 = time.time()
        pretrain(model, texts, tokenizer, config, device=device)
        print(f"Pretraining: {time.time()-t0:.1f}s")
        print()

    print("=== 5. Score-Training ===")
    t0 = time.time()
    train_scores(model, train_scenarios, tokenizer, config, device=device)
    print(f"Score-Training: {time.time()-t0:.1f}s")
    print()

    print("=== 6. Checkpoint ===")
    save_checkpoint(model, tokenizer, config, args.output)
    print(f"Gespeichert: {args.output}")
    print()

    _evaluate(
        model,
        tokenizer,
        val_scenarios,
        device,
        summary_mode=config.summary_mode,
    )
    return 0


def _evaluate(
    model,
    tokenizer,
    val_scenarios,
    device,
    *,
    summary_mode: str = SUMMARY_MODE_FULL,
) -> None:
    import torch

    summary_mode = normalize_summary_mode(summary_mode)
    model.eval()

    print("=== 7. Validation (IID) ===")
    if val_scenarios:
        total_abs_err = 0.0
        total_count = 0
        for scenario in val_scenarios[:200]:
            text = serialize_scenario(scenario, summary_mode=summary_mode)
            token_ids = _encode_evaluation_text(model, tokenizer, text)
            t = torch.tensor([token_ids], dtype=torch.long, device=device)
            a = torch.ones_like(t)
            with torch.inference_mode():
                out = model(t, a)
            pred = out["score"].item()
            total_abs_err += abs(pred - scenario.score)
            total_count += 1
        if total_count > 0:
            print(f"IID MAE: {total_abs_err/total_count:.4f} (n={total_count})")
    print()

    print("=== 8. OOD-Tests ===")
    from .ood_tests import all_ood_tests

    ood_tests = all_ood_tests()
    by_category: dict[str, list[float]] = {}
    for test in ood_tests:
        text = serialize_scenario(test.scenario, summary_mode=summary_mode)
        token_ids = _encode_evaluation_text(model, tokenizer, text)
        t = torch.tensor([token_ids], dtype=torch.long, device=device)
        a = torch.ones_like(t)
        with torch.inference_mode():
            out = model(t, a)
        pred = out["score"].item()
        err = abs(pred - test.scenario.score)
        category = test.category
        if summary_mode == SUMMARY_MODE_NONE and category == "OOD_LANGUAGE":
            category = "STRUCTURED_OOD (source: OOD_LANGUAGE)"
        by_category.setdefault(category, []).append(err)

    if summary_mode == SUMMARY_MODE_NONE:
        print(
            "Hinweis: OOD_LANGUAGE enthält ohne Summary keinen Sprach-OOD; "
            "Ausgabe daher als STRUCTURED_OOD mit Quellkategorie OOD_LANGUAGE."
        )

    print(f"{'Kategorie':20s} | {'MAE':>6s} | {'n':>3s}")
    print("-" * 40)
    total_err = 0.0
    total_n = 0
    for cat, errs in sorted(by_category.items()):
        mae = sum(errs) / len(errs)
        print(f"{cat:20s} | {mae:.4f} | {len(errs)}")
        total_err += sum(errs)
        total_n += len(errs)
    print("-" * 40)
    print(f"{'OOD MAE':20s} | {total_err/total_n:.4f} | {total_n}")
    print()

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

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError as exc:  # Optional ``ml`` dependency.
    raise unittest.SkipTest("training tests require PyTorch") from exc

from training.modeling.model import EconomyEncoder
from training.modeling.tokenizer import BPETokenizer
from training.scenarios import generate_parametric, EconomicScenario
from training.pipeline import (
    TrainingConfig,
    build_tokenizer,
    pretrain,
    train_scores,
    save_checkpoint,
    load_checkpoint,
)
from training.cli import _build_parser, _periodic_checkpoint_path


class TrainingPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.tmpdir = tempfile.TemporaryDirectory()
        self.model_dir = Path(self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_end_to_end_train_predict_checkpoint(self) -> None:
        scenarios = generate_parametric(num=200, seed=1)
        texts = [s.to_serialized() for s in scenarios]
        tokenizer = build_tokenizer(texts, vocab_size=1000)
        self.assertIn("[cls]", tokenizer.vocab)
        self.assertEqual(tokenizer.vocab["[pad]"], 0)

        config = TrainingConfig(
            model_dir=self.model_dir,
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            num_heads=4,
            num_layers=2,
            ff_dim=64,
            max_seq_len=256,
            batch_size=8,
            learning_rate=1e-3,
            max_epochs=2,
            log_every=999,
        )
        model = EconomyEncoder(
            vocab_size=tokenizer.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            ff_dim=config.ff_dim,
            max_seq_len=config.max_seq_len,
            dropout=0.0,
            pad_idx=tokenizer.vocab.get("[pad]", 0),
            cls_idx=tokenizer.vocab.get("[cls]", 1),
            sep_idx=tokenizer.vocab.get("[sep]", 2),
        )

        pretrain(model, texts, tokenizer, config)
        train_scores(model, scenarios, tokenizer, config)

        ckpt = self.model_dir / "test.pt"
        save_checkpoint(model, tokenizer, config, ckpt)
        self.assertTrue(ckpt.exists())

        loaded_model, loaded_tok, loaded_cfg = load_checkpoint(ckpt)
        self.assertEqual(loaded_tok.vocab_size, tokenizer.vocab_size)
        self.assertEqual(loaded_cfg["d_model"], config.d_model)

        scenario = scenarios[0]
        text = scenario.to_serialized()
        token_ids = loaded_tok.encode(text)
        t = torch.tensor([token_ids], dtype=torch.long)
        a = torch.ones_like(t)
        loaded_model.eval()
        with torch.inference_mode():
            out = loaded_model(t, a)
        score = out["score"].item()
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)

    def test_training_emits_checkpoints_at_epoch_intervals(self) -> None:
        scenarios = generate_parametric(num=4, seed=11)
        texts = [scenario.to_serialized() for scenario in scenarios]
        tokenizer = build_tokenizer(texts, vocab_size=80)
        config = TrainingConfig(
            model_dir=self.model_dir,
            vocab_size=tokenizer.vocab_size,
            d_model=16,
            num_heads=4,
            num_layers=1,
            ff_dim=32,
            max_seq_len=64,
            dropout=0.0,
            batch_size=4,
            max_epochs=3,
            log_every=999,
        )
        model = EconomyEncoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            ff_dim=config.ff_dim,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            pad_idx=tokenizer.vocab.get("[pad]", 0),
            cls_idx=tokenizer.vocab.get("[cls]", 1),
            sep_idx=tokenizer.vocab.get("[sep]", 2),
        )
        checkpoints: list[tuple[str, int]] = []

        pretrain(
            model,
            texts,
            tokenizer,
            config,
            checkpoint_every_epochs=2,
            checkpoint_callback=lambda phase, epoch: checkpoints.append((phase, epoch)),
        )
        train_scores(
            model,
            scenarios,
            tokenizer,
            config,
            checkpoint_every_epochs=2,
            checkpoint_callback=lambda phase, epoch: checkpoints.append((phase, epoch)),
        )

        self.assertEqual(checkpoints, [("pretrain", 2), ("score", 2)])

    def test_checkpoint_cli_defaults_and_snapshot_paths(self) -> None:
        args = _build_parser().parse_args(["train"])
        self.assertEqual(args.checkpoint_every, 1)
        self.assertEqual(
            _periodic_checkpoint_path(
                Path(".model_checkpoints/model.pt"),
                phase="score",
                completed_epochs=12,
            ),
            Path(".model_checkpoints/model.checkpoints/score_epoch_012.pt"),
        )

    def test_counterexamples_force_case_dependency(self) -> None:
        scenarios = generate_parametric(num=2000, seed=2)
        texts = [s.to_serialized() for s in scenarios]
        tokenizer = build_tokenizer(texts, vocab_size=3000)

        config = TrainingConfig(
            model_dir=self.model_dir,
            vocab_size=tokenizer.vocab_size,
            d_model=64,
            num_heads=4,
            num_layers=2,
            ff_dim=128,
            max_seq_len=256,
            batch_size=16,
            learning_rate=3e-4,
            max_epochs=10,
            log_every=999,
        )
        # Deterministic model initialization for this empirical quality gate.
        # The canonical serializer ordering (context-serializer-v3) changes
        # token sequences, so the seed pins a reproducible training outcome.
        torch.manual_seed(123)
        model = EconomyEncoder(
            vocab_size=tokenizer.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            ff_dim=config.ff_dim,
            max_seq_len=config.max_seq_len,
            dropout=0.0,
            pad_idx=tokenizer.vocab.get("[pad]", 0),
            cls_idx=tokenizer.vocab.get("[cls]", 1),
            sep_idx=tokenizer.vocab.get("[sep]", 2),
        )

        train_scores(model, scenarios, tokenizer, config)

        from training.scenarios import EconomicScenario, _compute_score, _build_context
        from training.bridge import serialize_scenario

        consumer = EconomicScenario(
            event_type="input_price_change",
            direction="increase",
            event_text="Oil prices surged by 30 percent after supply disruptions",
            case_role="consumer",
            case_type="airline",
            context_text=_build_context(
                "input_price_change", "consumer", "oil",
                35, 0.1, 0.2, 0.3,
            ),
            magnitude=30, exposure=35, hedging=0.1, pricing_power=0.2, substitution=0.3,
            horizon_days=30,
            score=_compute_score("input_price_change", "increase", "consumer", 30, 35, 0.1, 0.2, 0.3, 30),
        )
        producer = EconomicScenario(
            event_type="input_price_change",
            direction="increase",
            event_text="Oil prices surged by 30 percent after supply disruptions",
            case_role="producer",
            case_type="oil_producer",
            context_text=_build_context(
                "input_price_change", "producer", "oil",
                80, 0.0, 0.5, 0.5,
            ),
            magnitude=30, exposure=80, hedging=0.0, pricing_power=0.5, substitution=0.5,
            horizon_days=30,
            score=_compute_score("input_price_change", "increase", "producer", 30, 80, 0.0, 0.5, 0.5, 30),
        )

        model.eval()
        scores = []
        for scenario in (consumer, producer):
            text = serialize_scenario(scenario)
            token_ids = tokenizer.encode(text)
            t = torch.tensor([token_ids], dtype=torch.long)
            a = torch.ones_like(t)
            with torch.inference_mode():
                out = model(t, a)
            scores.append(out["score"].item())

        self.assertLess(
            scores[0],
            scores[1],
            f"consumer score {scores[0]:.4f} should be less than producer {scores[1]:.4f}",
        )


class RoPEPositionTest(unittest.TestCase):
    """Verify that RoPE makes token order matter."""

    def test_reversed_sequence_produces_different_score(self) -> None:
        torch.manual_seed(7)
        model = EconomyEncoder(
            vocab_size=32,
            d_model=16,
            num_heads=4,
            num_layers=1,
            ff_dim=32,
            max_seq_len=8,
            dropout=0.0,
        ).eval()

        original = torch.tensor([[1, 3, 4, 5, 6, 2]])
        reversed_ = torch.tensor([[1, 6, 5, 4, 3, 2]])

        with torch.inference_mode():
            s1 = model(original, torch.ones_like(original))["score"].item()
            s2 = model(reversed_, torch.ones_like(reversed_))["score"].item()

        self.assertNotAlmostEqual(
            s1, s2, places=4,
            msg="RoPE must make token order matter — reversed sequence should differ",
        )


class BPETokenizerTest(unittest.TestCase):
    def test_bpe_handles_unknown_words_via_subwords(self) -> None:
        texts = ["oil prices surged by 30 percent", "revenue is tied to oil"]
        tokenizer = BPETokenizer.train(texts, vocab_size=200)

        ids_known = tokenizer.encode("oil prices surged")
        ids_unknown = tokenizer.encode("petroleum surged")

        unk_id = tokenizer.vocab["[unk]"]
        unk_in_known = unk_id in ids_known
        unk_in_unknown = unk_id in ids_unknown

        self.assertFalse(unk_in_known, "known words should not produce [unk]")
        # Unknown word 'petroleum' should be split into subwords, not just [unk]
        # With a tiny vocab some chars may still be [unk], but not the whole word
        self.assertGreater(len(ids_unknown), 1, "unknown word should be split into subwords")

    def test_special_tokens_are_preserved(self) -> None:
        texts = ["some text here"]
        tokenizer = BPETokenizer.train(texts, vocab_size=100)

        ids = tokenizer.encode("[CLS] text [SEP]")
        self.assertEqual(ids[0], tokenizer.vocab["[cls]"])
        self.assertEqual(ids[-1], tokenizer.vocab["[sep]"])

    def test_save_and_load_roundtrip(self) -> None:
        texts = ["oil prices surged by 30 percent"]
        tokenizer = BPETokenizer.train(texts, vocab_size=200)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            tokenizer.save(path)
            loaded = BPETokenizer.load(path)
            self.assertEqual(loaded.vocab_size, tokenizer.vocab_size)
            original_ids = tokenizer.encode("oil prices")
            loaded_ids = loaded.encode("oil prices")
            self.assertEqual(original_ids, loaded_ids)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

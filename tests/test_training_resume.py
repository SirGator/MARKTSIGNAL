from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("training resume tests require PyTorch") from exc

from training.bridge import serialize_scenario
from training.modeling.model import EconomyEncoder
from training.pipeline import (
    TrainingConfig,
    build_tokenizer,
    load_checkpoint,
    load_training_checkpoint,
    save_checkpoint,
    train_scores,
)
from training.scenarios import generate_parametric


class _StopAfterCheckpoint(RuntimeError):
    pass


class TrainingResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.model_dir = Path(self.temporary_directory.name)
        self.examples = generate_parametric(num=8, seed=71)
        texts = [serialize_scenario(example) for example in self.examples]
        self.tokenizer = build_tokenizer(texts, vocab_size=100)
        self.config = TrainingConfig(
            model_dir=self.model_dir,
            vocab_size=self.tokenizer.vocab_size,
            d_model=16,
            num_heads=4,
            num_layers=1,
            ff_dim=32,
            max_seq_len=64,
            dropout=0.0,
            batch_size=4,
            learning_rate=1e-3,
            warmup_steps=0,
            max_epochs=4,
            log_every=999,
            seed=19,
        )
        torch.manual_seed(123)
        template = self._new_model()
        self.initial_state = {
            name: value.detach().clone()
            for name, value in template.state_dict().items()
        }

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _new_model(self) -> EconomyEncoder:
        vocab = self.tokenizer.vocab
        return EconomyEncoder(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            ff_dim=self.config.ff_dim,
            max_seq_len=self.config.max_seq_len,
            dropout=self.config.dropout,
            pad_idx=vocab["[pad]"],
            cls_idx=vocab["[cls]"],
            sep_idx=vocab["[sep]"],
        )

    def _model_from_initial_state(self) -> EconomyEncoder:
        model = self._new_model()
        model.load_state_dict(self.initial_state)
        return model

    def test_score_training_resume_matches_uninterrupted_run(self) -> None:
        uninterrupted = self._model_from_initial_state()
        train_scores(
            uninterrupted,
            self.examples,
            self.tokenizer,
            self.config,
        )

        interrupted = self._model_from_initial_state()
        checkpoint_path = self.model_dir / "score_epoch_002.pt"

        def stop_after_second_epoch(
            phase: str,
            completed_epochs: int,
            state: dict[str, object],
        ) -> None:
            self.assertEqual((phase, completed_epochs), ("score", 2))
            save_checkpoint(
                interrupted,
                self.tokenizer,
                self.config,
                checkpoint_path,
                training_state=state,
            )
            raise _StopAfterCheckpoint

        with self.assertRaises(_StopAfterCheckpoint):
            train_scores(
                interrupted,
                self.examples,
                self.tokenizer,
                self.config,
                checkpoint_every_epochs=2,
                resume_checkpoint_callback=stop_after_second_epoch,
            )

        portable_model, portable_tokenizer, portable_config = load_checkpoint(
            checkpoint_path
        )
        self.assertEqual(portable_tokenizer.fingerprint, self.tokenizer.fingerprint)
        self.assertEqual(portable_config["vocab_size"], self.config.vocab_size)
        self.assertIsInstance(portable_model, EconomyEncoder)

        resumed, resumed_tokenizer, _, resume_state = load_training_checkpoint(
            checkpoint_path,
            expected_phase="score",
        )
        train_scores(
            resumed,
            self.examples,
            resumed_tokenizer,
            self.config,
            resume_state=resume_state,
        )

        for name, expected in uninterrupted.state_dict().items():
            self.assertTrue(
                torch.equal(expected, resumed.state_dict()[name]),
                f"resumed tensor differs: {name}",
            )


    def test_early_stopping_restores_best_epoch(self) -> None:
        from unittest.mock import patch

        model = self._model_from_initial_state()
        target = float(self.examples[0].score)
        errors = iter((0.4, 0.3, 0.31, 0.32))
        best_snapshots: list[tuple[int, dict[str, torch.Tensor]]] = []

        def fake_predictions(*_args, **_kwargs) -> list[float]:
            return [target + next(errors)]

        def capture_best(epoch: int, _mae: float) -> None:
            best_snapshots.append(
                (
                    epoch,
                    {
                        name: value.detach().clone()
                        for name, value in model.state_dict().items()
                    },
                )
            )

        with patch("training.pipeline.predict_scores", side_effect=fake_predictions):
            result = train_scores(
                model,
                self.examples,
                self.tokenizer,
                self.config,
                validation_examples=[self.examples[0]],
                early_stopping_patience=2,
                best_checkpoint_callback=capture_best,
            )

        self.assertTrue(result["stopped_early"])
        self.assertEqual(result["epochs_completed"], 4)
        self.assertEqual(result["best_epoch"], 2)
        self.assertEqual([epoch for epoch, _ in best_snapshots], [1, 2])
        best_state = best_snapshots[-1][1]
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, best_state[name]))
    def test_inference_only_checkpoint_cannot_resume(self) -> None:
        path = self.model_dir / "inference.pt"
        save_checkpoint(
            self._model_from_initial_state(),
            self.tokenizer,
            self.config,
            path,
        )

        load_checkpoint(path)
        with self.assertRaisesRegex(ValueError, "inference-only"):
            load_training_checkpoint(path)


if __name__ == "__main__":
    unittest.main()

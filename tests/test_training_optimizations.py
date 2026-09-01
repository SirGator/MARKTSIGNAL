from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from training.cli import _prune_periodic_checkpoints
from training.metrics import compute_score_metrics, score_direction
from training.paraphrases import add_runtime_noise
from training.modeling.checkpoint import (
    TRAINING_STATE_VERSION,
    training_state_from_checkpoint,
)
from training.scenarios import generate_parametric


class ScoreMetricsTests(unittest.TestCase):
    def test_metrics_include_direction_and_strong_slice(self) -> None:
        metrics = compute_score_metrics(
            [-0.8, -0.02, 0.6, 0.2],
            [-0.5, 0.01, -0.1, 0.1],
        )

        self.assertEqual(metrics.count, 4)
        self.assertAlmostEqual(metrics.mae, 0.2825)
        self.assertAlmostEqual(metrics.sign_accuracy, 0.75)
        self.assertEqual(metrics.strong_count, 2)
        self.assertAlmostEqual(metrics.strong_mae or 0.0, 0.5)

    def test_direction_uses_neutral_band(self) -> None:
        self.assertEqual(score_direction(-0.06), -1)
        self.assertEqual(score_direction(0.04), 0)
        self.assertEqual(score_direction(0.06), 1)


class RuntimeNoiseTests(unittest.TestCase):
    def test_noise_is_deterministic_and_preserves_labels(self) -> None:
        scenarios = generate_parametric(num=5, seed=11)

        first = add_runtime_noise(scenarios, ratio=1.0, seed=17)
        second = add_runtime_noise(scenarios, ratio=1.0, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(len(first), len(scenarios) * 2)
        for original, noisy in zip(scenarios, first[len(scenarios):]):
            self.assertEqual(noisy.score, original.score)
            self.assertEqual(noisy.event_type, original.event_type)
            self.assertEqual(noisy.direction, original.direction)
            self.assertNotEqual(noisy.event_text, original.event_text)


class CheckpointRetentionTests(unittest.TestCase):
    def test_pruning_keeps_newest_checkpoints_per_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "model.pt"
            checkpoint_dir = output.with_suffix(".checkpoints")
            checkpoint_dir.mkdir()
            for epoch in range(1, 6):
                (checkpoint_dir / f"score_epoch_{epoch:03d}.pt").touch()
            (checkpoint_dir / "pretrain_epoch_001.pt").touch()
            (checkpoint_dir / "best.pt").touch()

            removed = _prune_periodic_checkpoints(output, phase="score", keep=2)

            self.assertEqual(
                [path.name for path in removed],
                ["score_epoch_001.pt", "score_epoch_002.pt", "score_epoch_003.pt"],
            )
            self.assertTrue((checkpoint_dir / "score_epoch_004.pt").exists())
            self.assertTrue((checkpoint_dir / "score_epoch_005.pt").exists())
            self.assertTrue((checkpoint_dir / "pretrain_epoch_001.pt").exists())
            self.assertTrue((checkpoint_dir / "best.pt").exists())


class ResumeStateContractTests(unittest.TestCase):
    @staticmethod
    def _checkpoint(*, phase: str = "score") -> dict[str, object]:
        return {
            "training_state_version": TRAINING_STATE_VERSION,
            "training_state": {
                "phase": phase,
                "completed_epochs": 2,
                "global_step": 8,
                "steps_per_epoch": 4,
                "target_epochs": 3,
                "optimizer_state": {},
                "scheduler_state": {},
                "loader_generator_state": "opaque",
                "rng_state": {},
                "run_contract": {},
                "phase_state": {},
                "losses": [],
            },
        }

    def test_resume_state_is_versioned_and_phase_checked(self) -> None:
        checkpoint = self._checkpoint()

        state = training_state_from_checkpoint(
            checkpoint,
            expected_phase="score",
        )

        self.assertEqual(state["completed_epochs"], 2)
        with self.assertRaisesRegex(ValueError, "does not match"):
            training_state_from_checkpoint(
                checkpoint,
                expected_phase="pretrain",
            )

    def test_inference_only_and_incomplete_pretrain_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "inference-only"):
            training_state_from_checkpoint({})

        checkpoint = self._checkpoint(phase="pretrain")
        with self.assertRaisesRegex(ValueError, "mlm_head_state"):
            training_state_from_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()

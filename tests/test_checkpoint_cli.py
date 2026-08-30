"""CLI contract for periodic training snapshots (does not require PyTorch)."""

from pathlib import Path
import unittest

from training.cli import _build_parser, _periodic_checkpoint_path, main


class CheckpointCliTests(unittest.TestCase):
    def test_defaults_to_checkpoint_after_each_epoch(self) -> None:
        args = _build_parser().parse_args(["train"])

        self.assertEqual(args.checkpoint_every, 1)

    def test_snapshot_path_keeps_final_output_unchanged(self) -> None:
        self.assertEqual(
            _periodic_checkpoint_path(
                Path(".model_checkpoints/model.pt"),
                phase="score",
                completed_epochs=12,
            ),
            Path(".model_checkpoints/model.checkpoints/score_epoch_012.pt"),
        )

    def test_non_positive_checkpoint_interval_is_rejected(self) -> None:
        with self.assertRaises(SystemExit) as result:
            main(["train", "--checkpoint-every", "0"])

        self.assertEqual(result.exception.code, 2)


if __name__ == "__main__":
    unittest.main()

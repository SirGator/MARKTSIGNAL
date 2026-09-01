import unittest

try:
    import torch
except ModuleNotFoundError as exc:  # Optional ``ml`` dependency.
    raise unittest.SkipTest("model tests require PyTorch") from exc

from training.modeling.model import EconomyEncoder


class EconomyEncoderTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = EconomyEncoder(
            vocab_size=32,
            d_model=16,
            num_heads=4,
            num_layers=1,
            ff_dim=32,
            max_seq_len=8,
            dropout=0.0,
        ).eval()

    def test_forward_returns_finite_bounded_values(self) -> None:
        result = self.model(
            token_ids=torch.tensor([[1, 3, 4, 5, 2, 0], [1, 6, 7, 8, 9, 2]]),
            attention_mask=torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]]),
        )

        self.assertEqual(result["score"].shape, (2, 1))
        self.assertTrue(torch.isfinite(result["score"]).all())
        self.assertTrue((result["score"].abs() <= 1.0).all())

    def test_empty_text_is_rejected_instead_of_producing_nan(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one non-padding token"):
            self.model(
                token_ids=torch.zeros((1, 2), dtype=torch.long),
                attention_mask=torch.zeros((1, 2), dtype=torch.long),
            )

    def test_missing_cls_token_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "CLS"):
            self.model(
                token_ids=torch.tensor([[3, 4, 5]]),
                attention_mask=torch.tensor([[1, 1, 1]]),
            )

    def test_sequence_too_long_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_seq_len"):
            self.model(
                token_ids=torch.ones((1, 100), dtype=torch.long),
                attention_mask=torch.ones((1, 100), dtype=torch.long),
            )

    def test_rope_capacity_matches_configured_max_sequence_length(self) -> None:
        model = EconomyEncoder(
            vocab_size=32,
            d_model=16,
            num_heads=4,
            num_layers=1,
            ff_dim=32,
            max_seq_len=520,
            dropout=0.0,
        ).eval()
        token_ids = torch.full((1, 520), 3, dtype=torch.long)
        token_ids[0, 0] = 1

        result = model(token_ids, torch.ones_like(token_ids))

        self.assertEqual(result["score"].shape, (1, 1))
        with self.assertRaisesRegex(ValueError, "max_seq_len"):
            model(
                torch.cat((token_ids, torch.tensor([[3]])), dim=1),
                torch.ones((1, 521), dtype=torch.long),
            )


if __name__ == "__main__":
    unittest.main()

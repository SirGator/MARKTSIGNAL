"""Contract tests for MLM masking, config validation, caching and the CLS contract."""

from __future__ import annotations

from pathlib import Path
import unittest

try:
    import torch
except ModuleNotFoundError as exc:  # Optional ``ml`` dependency.
    raise unittest.SkipTest("training hardening tests require PyTorch") from exc

from training.modeling.model import EconomyEncoder
from training.pipeline import (
    ScoreDataset,
    TextDataset,
    TrainingConfig,
    apply_mlm,
    build_tokenizer,
)
from training.scenarios import generate_parametric


class ApplyMLMTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        self.token_ids = torch.randint(5, 100, (16, 64))
        self.token_ids[:, 0] = 1  # [CLS]
        self.token_ids[:, -1] = 2  # [SEP]
        self.attention_mask = torch.ones_like(self.token_ids)

    def test_selected_positions_follow_80_10_10_distribution(self) -> None:
        masked, labels, selected = apply_mlm(
            self.token_ids,
            self.attention_mask,
            mask_prob=0.5,
            cls_idx=1,
            sep_idx=2,
            mask_idx=4,
        )
        self.assertTrue(selected.any())
        self.assertEqual((labels != -100).sum().item(), selected.sum().item())

        replaced_mask = (masked == 4) & selected
        kept_original = (masked == self.token_ids) & selected
        replaced_random = selected & ~replaced_mask & ~kept_original

        n = selected.sum().item()
        self.assertGreater(n, 100)
        ratio_mask = replaced_mask.sum().item() / n
        ratio_keep = kept_original.sum().item() / n
        ratio_random = replaced_random.sum().item() / n
        self.assertAlmostEqual(ratio_mask, 0.8, delta=0.1)
        self.assertAlmostEqual(ratio_keep, 0.1, delta=0.06)
        self.assertAlmostEqual(ratio_random, 0.1, delta=0.06)

    def test_structural_tokens_are_never_masked(self) -> None:
        protected = (5, 6, 7, 8)
        only_structure = torch.tensor([[1, 2, 5, 6, 7, 8, 9, 9, 2]])
        masked, labels, selected = apply_mlm(
            only_structure,
            torch.ones_like(only_structure),
            mask_prob=1.0,
            cls_idx=1,
            sep_idx=2,
            mask_idx=4,
            protected_idx=protected,
        )
        structural_positions = torch.tensor(
            [[True, True, True, True, True, True, False, False, True]],
        )
        self.assertFalse((selected & structural_positions).any())
        # the two non-structural tokens must be selected at mask_prob=1.0
        self.assertEqual(selected.sum().item(), 2)

    def test_empty_selection_falls_back_to_one_mask_per_sample(self) -> None:
        tiny = torch.tensor([[1, 2, 3, 3, 2], [1, 2, 3, 3, 2]])
        masked, labels, selected = apply_mlm(
            tiny,
            torch.ones_like(tiny),
            mask_prob=1e-9,
            cls_idx=1,
            sep_idx=2,
            mask_idx=4,
        )
        self.assertTrue(selected.any())
        self.assertEqual((labels != -100).sum().item(), selected.sum().item())

    def test_invalid_probabilities_are_rejected(self) -> None:
        for kwargs in (
            {"mask_prob": 0.0},
            {"mask_prob": 1.5},
            {"mask_prob": float("nan")},
            {"random_replace_prob": -0.1},
            {"keep_original_prob": 0.6, "random_replace_prob": 0.6},
        ):
            with self.assertRaises((TypeError, ValueError), msg=repr(kwargs)):
                apply_mlm(
                    self.token_ids[:1],
                    self.attention_mask[:1],
                    cls_idx=1,
                    sep_idx=2,
                    mask_idx=4,
                    **kwargs,
                )


class TrainingConfigValidationTests(unittest.TestCase):
    def _valid_kwargs(self) -> dict:
        return dict(
            d_model=32,
            num_heads=4,
            num_layers=2,
            ff_dim=64,
            vocab_size=100,
            batch_size=8,
            learning_rate=1e-3,
            weight_decay=0.01,
            warmup_steps=10,
            max_epochs=2,
            log_every=10,
            dropout=0.1,
        )

    def test_valid_config_is_accepted(self) -> None:
        TrainingConfig(model_dir=Path("unused"), **self._valid_kwargs())

    def test_invalid_values_fail_closed(self) -> None:
        cases = {
            "vocab_size": (11, 1_000_001),
            "d_model": (7, 16_385),
            "num_heads": (0, 257),
            "num_layers": (0, 257),
            "ff_dim": (0, 131_073),
            "batch_size": (0, -1),
            "warmup_steps": (-1,),
            "max_epochs": (0,),
            "log_every": (0, -5),
            "dropout": (-0.1, 1.0, 1.5),
            "learning_rate": (0.0, -1e-3, float("inf"), float("nan")),
            "weight_decay": (-0.01, float("nan")),
        }
        for name, values in cases.items():
            for value in values:
                kwargs = self._valid_kwargs()
                kwargs[name] = value
                with self.assertRaises(
                    (TypeError, ValueError),
                    msg=f"{name}={value!r} was accepted",
                ):
                    TrainingConfig(model_dir=Path("unused"), **kwargs)

    def test_type_confusions_are_rejected(self) -> None:
        for name in ("batch_size", "max_epochs", "log_every", "warmup_steps"):
            kwargs = self._valid_kwargs()
            kwargs[name] = 1.5
            with self.assertRaises(TypeError, msg=name):
                TrainingConfig(model_dir=Path("unused"), **kwargs)

    def test_d_model_must_be_divisible_by_heads(self) -> None:
        kwargs = self._valid_kwargs()
        kwargs["d_model"] = 15
        kwargs["num_heads"] = 4
        with self.assertRaisesRegex(ValueError, "divisible"):
            TrainingConfig(model_dir=Path("unused"), **kwargs)


class TokenizationCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scenarios = generate_parametric(num=24, seed=5)
        texts = [s.to_serialized() for s in self.scenarios]
        self.tokenizer = build_tokenizer(texts, vocab_size=200)

    def test_score_dataset_tokenizes_once_and_reuses_cache(self) -> None:
        calls: list[str] = []

        original_encode = self.tokenizer.encode

        def counting_encode(text: str) -> list[int]:
            calls.append(text)
            return original_encode(text)

        self.tokenizer.encode = counting_encode  # type: ignore[method-assign]
        try:
            dataset = ScoreDataset(
                self.scenarios,
                self.tokenizer,
                256,
                summary_mode="full",
            )
            encode_calls = len(calls)
            for _ in range(3):
                for index in range(len(dataset)):
                    dataset[index]
        finally:
            del self.tokenizer.encode  # type: ignore[misc]

        self.assertEqual(encode_calls, len(self.scenarios))

    def test_text_dataset_returns_identical_items_across_epochs(self) -> None:
        texts = [s.to_serialized() for s in self.scenarios]
        dataset = TextDataset(texts, self.tokenizer, 256)
        first_pass = [dataset[i]["token_ids"].clone() for i in range(len(dataset))]
        second_pass = [dataset[i]["token_ids"].clone() for i in range(len(dataset))]
        for expected, actual in zip(first_pass, second_pass):
            self.assertTrue(torch.equal(expected, actual))


class ExactClsContractTests(unittest.TestCase):
    def _model(self, strict: bool = True) -> EconomyEncoder:
        torch.manual_seed(11)
        return EconomyEncoder(
            vocab_size=32,
            d_model=16,
            num_heads=4,
            num_layers=1,
            ff_dim=32,
            max_seq_len=16,
            dropout=0.0,
            strict_validation=strict,
        ).eval()

    def test_two_cls_tokens_are_rejected(self) -> None:
        ids = torch.tensor([[1, 1, 3, 2]])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self._model()(ids, torch.ones_like(ids))

    def test_missing_cls_is_rejected(self) -> None:
        ids = torch.tensor([[3, 4, 5, 2]])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self._model()(ids, torch.ones_like(ids))

    def test_non_strict_mode_skips_per_batch_reductions(self) -> None:
        ids = torch.tensor([[3, 4, 5, 2]])  # no CLS at all
        model = self._model(strict=False)
        with torch.inference_mode():
            result = model(ids, torch.ones_like(ids))
        self.assertTrue(torch.isfinite(result["score"]).all())


if __name__ == "__main__":
    unittest.main()
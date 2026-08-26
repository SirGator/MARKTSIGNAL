from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError as exc:  # Optional ``ml`` dependency.
    raise unittest.SkipTest("training summary-mode tests require PyTorch") from exc

from src.models.model import EconomyEncoder
from src.models.context_serializer import SUMMARY_MODE_FULL, SUMMARY_MODE_NONE
from src.models.tokenizer import BPETokenizer
from training.bridge import serialize_scenario
from training.cli import (
    _build_parser,
    _evaluate,
    _seed_training,
    _serialization_corpora,
    _training_config_from_args,
)
from training.ood_tests import all_ood_tests
from training.paraphrases import expand_with_paraphrases
from training.data import TrainingExample
from training.pipeline import (
    ScoreDataset,
    TrainingConfig,
    load_checkpoint,
    save_checkpoint,
    serialize_training_example,
)
from training.scenarios import generate_parametric


class _RecordingTokenizer:
    vocab = {"[cls]": 1, "[unk]": 3}

    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.texts.append(text)
        return [1, 12, 13, 14, 15]


class _ZeroModel:
    max_seq_len = 3

    def eval(self) -> "_ZeroModel":
        return self

    def __call__(self, token_ids, attention_mask):
        if token_ids.shape[1] > self.max_seq_len:
            raise AssertionError("evaluation input was not truncated")
        return {"score": torch.tensor([0.0], device=token_ids.device)}


class SummaryModeContractTests(unittest.TestCase):
    def test_full_is_default_and_invalid_mode_is_rejected(self) -> None:
        config = TrainingConfig(model_dir=Path("unused"))
        self.assertEqual(config.summary_mode, SUMMARY_MODE_FULL)
        self.assertIn("summary-full", config.model_version)
        self.assertIn("summary-full", config.feature_schema_version)

        with self.assertRaises(ValueError):
            TrainingConfig(model_dir=Path("unused"), summary_mode="sometimes")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            TrainingConfig(
                model_dir=Path("unused"),
                summary_mode=SUMMARY_MODE_NONE,
                feature_schema_version="economy-context-v2+summary-full",
            )
        with self.assertRaisesRegex(ValueError, "at least 16"):
            TrainingConfig(model_dir=Path("unused"), max_seq_len=15)

    def test_no_summary_cli_flag_sets_config_mode(self) -> None:
        parser = _build_parser()

        default_args = parser.parse_args(["train"])
        default_config = _training_config_from_args(default_args, vocab_size=123)
        self.assertEqual(default_config.summary_mode, SUMMARY_MODE_FULL)
        self.assertEqual(default_config.seed, default_args.seed)

        ablation_args = parser.parse_args(["train", "--no-summary"])
        ablation_config = _training_config_from_args(ablation_args, vocab_size=123)
        self.assertEqual(ablation_config.summary_mode, SUMMARY_MODE_NONE)
        self.assertIn("summary-none", ablation_config.model_version)
        self.assertIn("summary-none", ablation_config.feature_schema_version)
        self.assertNotEqual(default_config.model_version, ablation_config.model_version)
        self.assertNotEqual(
            default_config.feature_schema_version,
            ablation_config.feature_schema_version,
        )

    def test_serialization_neutralizes_only_summary_without_mutation(self) -> None:
        scenario = generate_parametric(num=1, seed=17)[0]
        original_summary = scenario.event_text

        full_before = serialize_scenario(scenario, summary_mode=SUMMARY_MODE_FULL)
        neutralized = serialize_scenario(scenario, summary_mode=SUMMARY_MODE_NONE)
        full_after = serialize_scenario(scenario, summary_mode=SUMMARY_MODE_FULL)

        self.assertIn(original_summary, full_before)
        self.assertNotIn(original_summary, neutralized)
        self.assertIn("summary=[NO_SUMMARY]", neutralized)
        self.assertEqual(full_before, full_after)
        self.assertEqual(scenario.event_text, original_summary)

    def test_score_dataset_uses_configured_summary_mode(self) -> None:
        scenario = generate_parametric(num=1, seed=23)[0]
        tokenizer = _RecordingTokenizer()
        dataset = ScoreDataset(
            [scenario],
            tokenizer,
            max_seq_len=128,
            summary_mode=SUMMARY_MODE_NONE,
        )

        item = dataset[0]

        self.assertAlmostEqual(item["score"].item(), scenario.score, places=6)
        self.assertEqual(len(tokenizer.texts), 1)
        self.assertIn("summary=[NO_SUMMARY]", tokenizer.texts[0])
        self.assertNotIn(scenario.event_text, tokenizer.texts[0])

    def test_none_mode_rejects_unstructured_legacy_training_examples(self) -> None:
        legacy = TrainingExample(
            event_text="Oil rose 30 percent",
            case_text="type=airline",
            context_text="fuel_cost_share=31 percent",
            score=-0.7,
        )

        with self.assertRaisesRegex(ValueError, "requires EconomicScenario"):
            serialize_training_example(
                legacy,
                summary_mode=SUMMARY_MODE_NONE,
            )

    def test_validation_and_every_ood_path_use_same_none_mode(self) -> None:
        scenario = generate_parametric(num=1, seed=31)[0]
        tokenizer = _RecordingTokenizer()

        output = io.StringIO()
        with redirect_stdout(output):
            _evaluate(
                _ZeroModel(),
                tokenizer,
                [scenario],
                torch.device("cpu"),
                summary_mode=SUMMARY_MODE_NONE,
            )

        expected_count = 1 + len(all_ood_tests()) + 8
        self.assertEqual(len(tokenizer.texts), expected_count)
        for text in tokenizer.texts:
            event_line = next(
                line for line in text.splitlines() if line.startswith("[EVENT]")
            )
            self.assertIn("[NO_SUMMARY]", event_line)
        self.assertIn("STRUCTURED_OOD", output.getvalue())
        self.assertIn("Quellkategorie OOD_LANGUAGE", output.getvalue())
        self.assertEqual(scenario.event_text, generate_parametric(num=1, seed=31)[0].event_text)

    def test_full_and_none_runs_keep_identical_expanded_samples(self) -> None:
        base = generate_parametric(num=8, seed=41)
        expansion_args = {
            "num_paraphrases": 2,
            "neutralize_ratio": 0.25,
            "seed": 43,
        }

        full_examples = expand_with_paraphrases(base, **expansion_args)
        none_examples = expand_with_paraphrases(base, **expansion_args)
        full_texts = [
            serialize_scenario(item, summary_mode=SUMMARY_MODE_FULL)
            for item in full_examples
        ]
        none_texts = [
            serialize_scenario(item, summary_mode=SUMMARY_MODE_NONE)
            for item in none_examples
        ]

        self.assertEqual(full_examples, none_examples)
        self.assertEqual(len(full_texts), len(none_texts))
        self.assertTrue(
            all("summary=[NO_SUMMARY]" in text for text in none_texts)
        )

        full_tokenizer_texts, full_model_texts = _serialization_corpora(
            full_examples,
            summary_mode=SUMMARY_MODE_FULL,
        )
        none_tokenizer_texts, none_model_texts = _serialization_corpora(
            none_examples,
            summary_mode=SUMMARY_MODE_NONE,
        )
        self.assertEqual(full_tokenizer_texts, none_tokenizer_texts)
        self.assertEqual(full_model_texts, full_texts)
        self.assertEqual(none_model_texts, none_texts)

        full_tokenizer = BPETokenizer.train(full_tokenizer_texts, vocab_size=300)
        none_tokenizer = BPETokenizer.train(none_tokenizer_texts, vocab_size=300)
        self.assertEqual(full_tokenizer.vocab, none_tokenizer.vocab)
        self.assertEqual(full_tokenizer.merges, none_tokenizer.merges)
        self.assertEqual(full_tokenizer.fingerprint, none_tokenizer.fingerprint)

        model_args = {
            "vocab_size": full_tokenizer.vocab_size,
            "d_model": 16,
            "num_heads": 4,
            "num_layers": 1,
            "ff_dim": 32,
            "max_seq_len": 64,
            "dropout": 0.0,
        }
        _seed_training(47)
        full_model = EconomyEncoder(**model_args)
        _seed_training(47)
        none_model = EconomyEncoder(**model_args)
        self.assertEqual(
            sum(parameter.numel() for parameter in full_model.parameters()),
            sum(parameter.numel() for parameter in none_model.parameters()),
        )
        for name, full_value in full_model.state_dict().items():
            self.assertTrue(torch.equal(full_value, none_model.state_dict()[name]))

    def test_checkpoint_persists_none_mode(self) -> None:
        tokenizer = BPETokenizer.train(["[CLS] tiny economy text [SEP]"], vocab_size=80)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                model_dir=Path(tmpdir),
                vocab_size=tokenizer.vocab_size,
                d_model=16,
                num_heads=4,
                num_layers=1,
                ff_dim=32,
                max_seq_len=64,
                dropout=0.0,
                summary_mode=SUMMARY_MODE_NONE,
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
            path = Path(tmpdir) / "none-mode.pt"

            save_checkpoint(model, tokenizer, config, path)
            _, _, loaded_config = load_checkpoint(path)

        self.assertEqual(loaded_config["summary_mode"], SUMMARY_MODE_NONE)


if __name__ == "__main__":
    unittest.main()

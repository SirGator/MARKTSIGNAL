"""Test: training serialization == production serialization."""

from __future__ import annotations

import unittest

from src.models.context_serializer import ContextSerializer
from training.bridge import scenario_to_context_bundle, serialize_scenario
from training.cli import _encode_evaluation_text, _serialization_corpora
from training.paraphrases import expand_with_paraphrases
from training.scenarios import generate_parametric


class SerializationParityTest(unittest.TestCase):
    """Training and production must use the exact same serialization."""

    def test_training_path_equals_production_path(self) -> None:
        scenarios = generate_parametric(num=50, seed=42)
        serializer = ContextSerializer()

        for scenario in scenarios:
            training_text = scenario.to_serialized()
            bundle = scenario_to_context_bundle(scenario)
            production_text = serializer.serialize(
                bundle,
                horizon=f"{scenario.horizon_days}d",
            )
            self.assertEqual(
                training_text,
                production_text,
                f"Serialization differs for scenario: {scenario.event_text[:60]}",
            )

    def test_no_unk_tokens_in_production_format(self) -> None:
        from src.models.tokenizer import BPETokenizer

        scenarios = generate_parametric(num=200, seed=42)
        texts = [serialize_scenario(s) for s in scenarios]
        tokenizer = BPETokenizer.train(texts, vocab_size=4000)

        unk_id = tokenizer.vocab["[unk]"]
        total_unk = 0
        total_tokens = 0
        for text in texts:
            ids = tokenizer.encode(text)
            total_unk += sum(1 for i in ids if i == unk_id)
            total_tokens += len(ids)

        self.assertEqual(
            total_unk,
            0,
            f"Production format produces {total_unk} UNK tokens out of {total_tokens}",
        )

    def test_production_format_fits_in_max_seq_len(self) -> None:
        from src.models.tokenizer import BPETokenizer

        scenarios = generate_parametric(num=200, seed=42)
        texts = [serialize_scenario(s) for s in scenarios]
        tokenizer = BPETokenizer.train(texts, vocab_size=4000)

        max_tokens = 0
        for text in texts:
            ids = tokenizer.encode(text)
            max_tokens = max(max_tokens, len(ids))

        self.assertLess(
            max_tokens,
            512,
            f"Longest production serialization is {max_tokens} tokens, must fit in max_seq_len=512",
        )

    def test_full_and_none_ablation_share_one_marker_safe_tokenizer(self) -> None:
        from src.models.tokenizer import BPETokenizer

        scenarios = expand_with_paraphrases(
            generate_parametric(num=8, seed=41),
            num_paraphrases=2,
            neutralize_ratio=0.0,
            seed=43,
        )
        full_tokenizer_texts, _ = _serialization_corpora(
            scenarios,
            summary_mode="full",
        )
        none_tokenizer_texts, none_model_texts = _serialization_corpora(
            scenarios,
            summary_mode="none",
        )

        self.assertEqual(full_tokenizer_texts, none_tokenizer_texts)
        tokenizer = BPETokenizer.train(full_tokenizer_texts, vocab_size=300)
        unk_id = tokenizer.vocab["[unk]"]
        marker_id = tokenizer.vocab["[no_summary]"]
        for text in none_model_texts:
            token_ids = tokenizer.encode(text)
            self.assertNotIn(unk_id, token_ids)
            self.assertEqual(token_ids.count(marker_id), 1)

    def test_evaluation_uses_the_same_max_sequence_boundary(self) -> None:
        class Model:
            max_seq_len = 3

        class Tokenizer:
            @staticmethod
            def encode(_text: str) -> list[int]:
                return [1, 2, 3, 4, 5]

        self.assertEqual(
            _encode_evaluation_text(Model(), Tokenizer(), "ignored"),
            [1, 2, 3],
        )


if __name__ == "__main__":
    unittest.main()

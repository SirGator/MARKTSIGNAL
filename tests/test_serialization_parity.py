"""Test: training serialization == production serialization."""

from __future__ import annotations

from datetime import datetime, timezone
import unittest

from training.domain import (
    CanonicalEvent,
    CaseRef,
    ContextBundle,
    ContextFact,
    ProvenanceRecord,
)
from training.modeling.context_serializer import ContextSerializer
from training.bridge import scenario_to_context_bundle, serialize_scenario
from training.cli import _encode_evaluation_text, _serialization_corpora
from training.paraphrases import expand_with_paraphrases
from training.scenarios import generate_parametric


UTC = timezone.utc


def _fact(fact_id: str, predicate: str, value, observed_at: datetime) -> ContextFact:
    return ContextFact(
        fact_id=fact_id,
        subject_id="company:sort-test",
        predicate=predicate,
        value=value,
        unit="ratio",
        observed_at=observed_at,
        confidence=0.9,
    )


def _hist_event(event_id: str, occurred_at: datetime) -> CanonicalEvent:
    return CanonicalEvent(
        event_id=event_id,
        event_type="input_price_change",
        summary="historical",
        occurred_at=occurred_at,
        observed_at=occurred_at,
        subject_ids=("commodity:oil",),
        direction="increase",
        magnitude=0.2,
        confidence=0.9,
    )


def _bundle(facts, historical) -> ContextBundle:
    at = datetime(2026, 1, 10, 12, tzinfo=UTC)
    earlier = datetime(2026, 1, 1, 12, tzinfo=UTC)
    return ContextBundle(
        event=CanonicalEvent(
            event_id="event:sort",
            event_type="input_price_change",
            summary="primary",
            occurred_at=at,
            observed_at=at,
            subject_ids=("commodity:oil",),
            direction="increase",
            magnitude=0.3,
            confidence=0.9,
        ),
        case=CaseRef(
            case_id="case:sort",
            entity_id="company:sort-test",
            name="Sort Test",
        ),
        cutoff=at,
        retrieved_at=at,
        horizon="30d",
        facts=tuple(facts),
        historical_events=tuple(historical),
    )


class CanonicalOrderingTests(unittest.TestCase):
    """Retriever order must not influence the serialized token sequence."""

    def test_fact_order_does_not_change_serialization(self) -> None:
        at = datetime(2026, 1, 1, 12, tzinfo=UTC)
        facts_a = (
            _fact("fact:2", "hedging", 0.1, at),
            _fact("fact:1", "input_dependency", 0.4, at),
            _fact("fact:0", "role", "consumer", at),
        )
        facts_b = tuple(reversed(facts_a))
        serializer = ContextSerializer()

        text_a = serializer.serialize(_bundle(facts_a, ()), horizon="30d")
        text_b = serializer.serialize(_bundle(facts_b, ()), horizon="30d")

        self.assertEqual(text_a, text_b)
        lines = [line for line in text_a.splitlines() if line.startswith("[CONTEXT]")]
        predicates = [line.split("predicate=")[1].split(";")[0] for line in lines]
        self.assertEqual(
            [p.casefold() for p in predicates],
            sorted(p.casefold() for p in predicates),
        )

    def test_historical_event_order_does_not_change_serialization(self) -> None:
        first = datetime(2026, 1, 2, 12, tzinfo=UTC)
        second = datetime(2026, 1, 5, 12, tzinfo=UTC)
        hist_a = (
            _hist_event("event:b", second),
            _hist_event("event:a", first),
        )
        hist_b = tuple(reversed(hist_a))
        serializer = ContextSerializer()

        text_a = serializer.serialize(_bundle((), hist_a), horizon="30d")
        text_b = serializer.serialize(_bundle((), hist_b), horizon="30d")

        self.assertEqual(text_a, text_b)


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
        from training.modeling.tokenizer import BPETokenizer

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
        from training.modeling.tokenizer import BPETokenizer

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
        from training.modeling.tokenizer import BPETokenizer

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

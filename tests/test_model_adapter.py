from __future__ import annotations

from datetime import datetime, timezone
import unittest

try:
    import torch
except ModuleNotFoundError as exc:  # Optional ``ml`` dependency.
    raise unittest.SkipTest("model-adapter tests require PyTorch") from exc
from torch import nn

from src.domain import CanonicalEvent, CaseRef, ContextBundle, ContextFact
from src.models import (
    ContextTensorEncoder,
    EconomyEncoder,
    TorchEconomyModel,
)
from src.models.tokenizer import BPETokenizer


UTC = timezone.utc


def instant(day: int) -> datetime:
    return datetime(2026, 1, day, 12, tzinfo=UTC)


def context_bundle() -> ContextBundle:
    case = CaseRef(
        case_id="case:airline",
        entity_id="company:airline",
        name="Airline X",
    )
    return ContextBundle(
        event=CanonicalEvent(
            event_id="event:oil-up",
            event_type="commodity_price_change",
            summary="Oil rises 25 percent",
            occurred_at=instant(3),
            observed_at=instant(3),
            subject_ids=("commodity:oil",),
            direction="up",
            magnitude=25,
            unit="percent",
        ),
        case=case,
        cutoff=instant(4),
        retrieved_at=instant(4),
        horizon="30d",
        facts=(
            ContextFact(
                fact_id="fact:consumes-oil",
                subject_id=case.entity_id,
                predicate="CONSUMES",
                object_id="commodity:oil",
                observed_at=instant(1),
            ),
        ),
    )


def make_tokenizer() -> BPETokenizer:
    texts = [
        "[CLS] [EVENT] type=commodity_price_change summary=Oil rises 25 percent subjects=commodity:oil direction=up magnitude=25 unit=percent confidence=0.5",
        "[CASE] type=company name=[CASE_ENTITY]",
        "[HORIZON] 30d",
        "[CONTEXT] subject=[CASE_ENTITY] predicate=consumes object=commodity:oil confidence=0.5",
        "[SEP]",
        context_bundle().event.summary,
    ]
    return BPETokenizer.train(texts, vocab_size=200)


def encoder() -> ContextTensorEncoder:
    return ContextTensorEncoder(
        tokenizer=make_tokenizer(),
        feature_schema_version="features-v1",
        max_seq_len=128,
    )


class ContextTensorEncoderTests(unittest.TestCase):
    def test_structured_bundle_maps_to_token_ids_and_mask(self) -> None:
        encoded = encoder().encode(context_bundle(), horizon="30d")

        self.assertEqual(encoded.token_ids.shape[0], 1)
        self.assertEqual(encoded.attention_mask.shape, encoded.token_ids.shape)
        self.assertEqual(encoded.feature_schema_version, "features-v1")
        self.assertGreater(len(encoded.input_hash), 0)

    def test_cls_token_is_present(self) -> None:
        encoded = encoder().encode(context_bundle(), horizon="30d")
        token_ids = encoded.token_ids[0].tolist()
        tokenizer = make_tokenizer()
        cls_id = tokenizer.vocab.get("[cls]")
        self.assertIn(cls_id, token_ids)

    def test_torch_adapter_returns_one_internal_economy_score(self) -> None:
        model = EconomyEncoder(
            vocab_size=200,
            d_model=16,
            num_heads=4,
            num_layers=1,
            ff_dim=32,
            max_seq_len=128,
            dropout=0.0,
            pad_idx=0,
            cls_idx=1,
            sep_idx=2,
        )
        adapter = TorchEconomyModel(
            model,
            encoder(),
            model_version="untrained-smoke-test",
        )

        score = adapter.predict(context_bundle(), horizon="30d")

        self.assertGreaterEqual(score.score, -1.0)
        self.assertLessEqual(score.score, 1.0)
        self.assertEqual(score.model_version, "untrained-smoke-test")
        self.assertEqual(score.feature_schema_version, "features-v1")
        self.assertEqual(len(score.input_hash), 64)

    def test_torch_adapter_rejects_grossly_out_of_range_model_output(self) -> None:
        class InvalidModel(nn.Module):
            def forward(self, token_ids, attention_mask):
                return {"score": torch.tensor([[99.0]])}

        adapter = TorchEconomyModel(
            InvalidModel(),
            encoder(),
            model_version="invalid-test-model",
        )

        with self.assertRaisesRegex(ValueError, "outside"):
            adapter.predict(context_bundle(), horizon="30d")


if __name__ == "__main__":
    unittest.main()

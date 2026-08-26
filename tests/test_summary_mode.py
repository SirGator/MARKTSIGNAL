from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from src.domain import CanonicalEvent, CaseRef, ContextBundle
from src.models import (
    BPETokenizer,
    NO_SUMMARY,
    SUMMARY_MODE_FULL,
    SUMMARY_MODE_NONE,
    ContextSerializer,
)
from src.models.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    SERIALIZER_CONTRACT_VERSION,
    load_economy_model,
    summary_mode_from_checkpoint_config,
    validate_checkpoint_metadata,
    validate_checkpoint_sequence_length,
    validate_checkpoint_token_ids,
)


UTC = timezone.utc


def instant(day: int) -> datetime:
    return datetime(2026, 1, day, 12, tzinfo=UTC)


def context_bundle() -> ContextBundle:
    case = CaseRef(
        case_id="case:airline",
        entity_id="company:airline",
        name="Airline X",
    )
    historical = CanonicalEvent(
        event_id="event:historical",
        event_type="historical_test",
        summary="HISTORICAL SUMMARY MUST NOT LEAK",
        occurred_at=instant(1),
        observed_at=instant(1),
        subject_ids=("commodity:oil",),
    )
    current = CanonicalEvent(
        event_id="event:current",
        event_type="current_test",
        summary="CURRENT SUMMARY MUST NOT LEAK",
        occurred_at=instant(2),
        observed_at=instant(2),
        subject_ids=("commodity:oil",),
    )
    return ContextBundle(
        event=current,
        case=case,
        cutoff=instant(3),
        retrieved_at=instant(3),
        horizon="30d",
        historical_events=(historical,),
    )


class ContextSerializerSummaryModeTests(unittest.TestCase):
    def test_full_mode_serializes_primary_and_historical_summaries(self) -> None:
        serialized = ContextSerializer(summary_mode="full").serialize(
            context_bundle(),
            horizon="30d",
        )

        self.assertIn("summary=CURRENT SUMMARY MUST NOT LEAK", serialized)
        self.assertIn("summary=HISTORICAL SUMMARY MUST NOT LEAK", serialized)
        self.assertNotIn(NO_SUMMARY, serialized)

    def test_none_mode_replaces_every_summary_with_exact_sentinel(self) -> None:
        serialized = ContextSerializer(summary_mode="none").serialize(
            context_bundle(),
            horizon="30d",
        )

        self.assertEqual(serialized.count(f"summary={NO_SUMMARY}"), 2)
        self.assertNotIn("CURRENT SUMMARY MUST NOT LEAK", serialized)
        self.assertNotIn("HISTORICAL SUMMARY MUST NOT LEAK", serialized)
        self.assertNotIn("(NO_SUMMARY)", serialized)

    def test_none_mode_neutralizes_raw_event_text_too(self) -> None:
        serialized = ContextSerializer(summary_mode="none").serialize_text(
            event_text="SECRET RAW SUMMARY",
            case_type="company",
            horizon="30d",
            context_text="role=consumer",
        )

        self.assertIn(f"[EVENT] {NO_SUMMARY}", serialized)
        self.assertNotIn("SECRET RAW SUMMARY", serialized)

    def test_existing_no_summary_marker_is_not_escaped_in_full_mode(self) -> None:
        serialized = ContextSerializer().serialize_text(
            event_text=NO_SUMMARY,
            case_type="company",
            horizon="30d",
            context_text="",
        )

        self.assertIn(f"[EVENT] {NO_SUMMARY}", serialized)
        self.assertNotIn("(NO_SUMMARY)", serialized)

    def test_only_full_and_none_are_accepted(self) -> None:
        self.assertEqual(SUMMARY_MODE_FULL, "full")
        self.assertEqual(SUMMARY_MODE_NONE, "none")
        self.assertEqual(
            ContextSerializer(summary_mode=" NONE ").summary_mode,
            SUMMARY_MODE_NONE,
        )
        with self.assertRaisesRegex(ValueError, "summary_mode"):
            ContextSerializer(summary_mode="automatic")
        with self.assertRaisesRegex(TypeError, "summary_mode"):
            ContextSerializer(summary_mode=None)  # type: ignore[arg-type]

    def test_no_summary_is_reserved_even_when_tokenizer_corpus_has_no_marker(self) -> None:
        full_text = ContextSerializer(summary_mode="full").serialize(
            context_bundle(),
            horizon="30d",
        )
        none_text = ContextSerializer(summary_mode="none").serialize(
            context_bundle(),
            horizon="30d",
        )
        tokenizer = BPETokenizer.train([full_text], vocab_size=160)

        token_ids = tokenizer.encode(none_text)

        self.assertIn("[no_summary]", tokenizer.vocab)
        self.assertEqual(
            token_ids.count(tokenizer.vocab["[no_summary]"]),
            2,
        )
        self.assertNotIn(tokenizer.vocab["[unk]"], token_ids)

    def test_v2_tokenizer_rejects_missing_or_misnumbered_summary_marker(self) -> None:
        valid = BPETokenizer.train(["ordinary full summary"], vocab_size=80)
        broken_vocab = valid.vocab
        broken_vocab.pop("[no_summary]")
        missing = BPETokenizer(vocab=broken_vocab, merges=valid.merges)
        with self.assertRaisesRegex(ValueError, "special-token contract"):
            missing.validate_special_tokens(include_no_summary=True)

        broken_vocab = valid.vocab
        broken_vocab["[no_summary]"] = 99
        misnumbered = BPETokenizer(vocab=broken_vocab, merges=valid.merges)
        with self.assertRaisesRegex(ValueError, "special-token contract"):
            misnumbered.validate_special_tokens(include_no_summary=True)

    def test_legacy_full_serializer_rejects_explicit_missing_summary_marker(self) -> None:
        serializer = ContextSerializer(
            summary_mode="full",
            allow_no_summary_marker=False,
        )
        with self.assertRaisesRegex(ValueError, "real event summary"):
            serializer.serialize_text(
                event_text=NO_SUMMARY,
                case_type="company",
                horizon="30d",
                context_text="role=consumer",
            )


class CheckpointSummaryModeTests(unittest.TestCase):
    def test_v2_checkpoint_requires_versioned_serializer_metadata(self) -> None:
        config = {"summary_mode": "none"}
        with self.assertRaisesRegex(ValueError, "checkpoint_format_version"):
            validate_checkpoint_metadata({}, config)
        validate_checkpoint_metadata(
            {
                "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
                "serializer_contract_version": SERIALIZER_CONTRACT_VERSION,
            },
            config,
        )

    def test_checkpoint_sequence_length_matches_production_minimum(self) -> None:
        self.assertEqual(
            validate_checkpoint_sequence_length({"max_seq_len": 16}),
            16,
        )
        for invalid in (True, 15, 12.5, None):
            with self.subTest(value=invalid), self.assertRaises(
                (TypeError, ValueError)
            ):
                validate_checkpoint_sequence_length({"max_seq_len": invalid})

    def test_checkpoint_model_token_ids_must_match_tokenizer(self) -> None:
        vocab = {"[pad]": 0, "[cls]": 1, "[sep]": 2}
        validate_checkpoint_token_ids(
            {"pad_idx": 0, "cls_idx": 1, "sep_idx": 2},
            vocab,
            require_explicit=True,
        )
        with self.assertRaisesRegex(ValueError, "cls_idx"):
            validate_checkpoint_token_ids(
                {"pad_idx": 0, "cls_idx": 9, "sep_idx": 2},
                vocab,
                require_explicit=True,
            )

    def test_missing_mode_is_ambiguous_and_legacy_none_requires_retraining(self) -> None:
        with self.assertRaisesRegex(ValueError, "legacy checkpoint"):
            summary_mode_from_checkpoint_config({})
        self.assertEqual(
            summary_mode_from_checkpoint_config(
                {},
                legacy_summary_mode="full",
            ),
            "full",
        )
        with self.assertRaisesRegex(ValueError, "incompatible.*retrain"):
            summary_mode_from_checkpoint_config(
                {},
                legacy_summary_mode="none",
            )
        self.assertEqual(
            summary_mode_from_checkpoint_config(
                {"summary_mode": "full"},
            ),
            "full",
        )
        for invalid in (None, "", "summaries-if-available"):
            with self.subTest(mode=invalid), self.assertRaisesRegex(
                ValueError,
                "config.summary_mode",
            ):
                summary_mode_from_checkpoint_config({"summary_mode": invalid})

        with self.assertRaisesRegex(ValueError, "legacy_summary_mode"):
            summary_mode_from_checkpoint_config(
                {},
                legacy_summary_mode="automatic",
            )

    def test_loader_builds_encoder_with_checkpoint_summary_mode(self) -> None:
        checkpoint = {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "serializer_contract_version": SERIALIZER_CONTRACT_VERSION,
            "config": {
                "summary_mode": "none",
                "tokenizer_hash": "fake-tokenizer-hash",
                "model_version": "checkpoint-model-v7+summary-none",
                "feature_schema_version": "checkpoint-features-v8+summary-none",
                "vocab_size": 8,
                "d_model": 4,
                "num_heads": 1,
                "num_layers": 1,
                "ff_dim": 8,
                "max_seq_len": 32,
                "dropout": 0.0,
                "pad_idx": 0,
                "cls_idx": 1,
                "sep_idx": 2,
            },
            "vocab": {"[pad]": 0, "[cls]": 1, "[sep]": 2},
            "merges": [],
            "model_state": {"weight": "fake"},
        }

        fake_torch = ModuleType("torch")
        fake_torch.load = lambda *_args, **_kwargs: checkpoint  # type: ignore[attr-defined]

        class FakeTokenizer:
            def __init__(self, *, vocab, merges):
                self.vocab = vocab
                self.merges = merges
                self.fingerprint = "fake-tokenizer-hash"
                self.vocab_size = 8

            def validate_special_tokens(self, **_kwargs):
                pass

        class FakeModel:
            def __init__(self, **kwargs):
                self.config = kwargs
                self.state = None

            def load_state_dict(self, state):
                self.state = state

        class FakeEncoder:
            def __init__(self, **kwargs):
                self.tokenizer = kwargs["tokenizer"]
                self.serializer = kwargs["serializer"]
                self.summary_mode = kwargs["summary_mode"]
                self.max_seq_len = kwargs["max_seq_len"]
                self.feature_schema_version = kwargs["feature_schema_version"]

        class FakeAdapter:
            def __init__(self, model, encoder, *, model_version):
                self.model = model
                self.encoder = encoder
                self.model_version = model_version

        fake_modules = {
            "torch": fake_torch,
            "src.models.adapter": self._module(
                "src.models.adapter",
                TorchEconomyModel=FakeAdapter,
            ),
            "src.models.context_encoder": self._module(
                "src.models.context_encoder",
                ContextTensorEncoder=FakeEncoder,
            ),
            "src.models.model": self._module(
                "src.models.model",
                EconomyEncoder=FakeModel,
            ),
            "src.models.tokenizer": self._module(
                "src.models.tokenizer",
                BPETokenizer=FakeTokenizer,
            ),
        }
        with patch.dict(sys.modules, fake_modules):
            adapter, encoder, _tokenizer = load_economy_model(
                Path("unused-checkpoint.pt")
            )

        self.assertEqual(encoder.summary_mode, "none")
        self.assertEqual(encoder.serializer.summary_mode, "none")
        self.assertEqual(encoder.max_seq_len, 32)
        self.assertEqual(
            encoder.feature_schema_version,
            "checkpoint-features-v8+summary-none",
        )
        self.assertEqual(adapter.encoder, encoder)
        self.assertEqual(
            adapter.model_version,
            "checkpoint-model-v7+summary-none",
        )

    def test_loader_rejects_legacy_no_summary_checkpoint(self) -> None:
        checkpoint = {
            "config": {
                "vocab_size": 8,
                "d_model": 4,
                "num_heads": 1,
                "num_layers": 1,
                "ff_dim": 8,
                "max_seq_len": 32,
            },
            "vocab": {"[pad]": 0, "[cls]": 1, "[sep]": 2},
            "model_state": {},
        }

        fake_torch = ModuleType("torch")
        fake_torch.load = lambda *_args, **_kwargs: checkpoint  # type: ignore[attr-defined]

        class FakeTokenizer:
            def __init__(self, **_kwargs):
                pass

        class FakeModel:
            def __init__(self, **_kwargs):
                pass

            def load_state_dict(self, _state):
                pass

        class FakeEncoder:
            def __init__(self, **kwargs):
                self.serializer = kwargs["serializer"]
                self.summary_mode = kwargs["summary_mode"]

        class FakeAdapter:
            def __init__(self, _model, encoder, **_kwargs):
                self.encoder = encoder

        fake_modules = {
            "torch": fake_torch,
            "src.models.adapter": self._module(
                "src.models.adapter",
                TorchEconomyModel=FakeAdapter,
            ),
            "src.models.context_encoder": self._module(
                "src.models.context_encoder",
                ContextTensorEncoder=FakeEncoder,
            ),
            "src.models.model": self._module(
                "src.models.model",
                EconomyEncoder=FakeModel,
            ),
            "src.models.tokenizer": self._module(
                "src.models.tokenizer",
                BPETokenizer=FakeTokenizer,
            ),
        }
        with patch.dict(sys.modules, fake_modules):
            with self.assertRaisesRegex(ValueError, "incompatible.*retrain"):
                load_economy_model(
                    Path("legacy-model-no-summary.pt"),
                    legacy_summary_mode="none",
                )

    @staticmethod
    def _module(name: str, **attributes: object) -> ModuleType:
        module = ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        return module


if __name__ == "__main__":
    unittest.main()

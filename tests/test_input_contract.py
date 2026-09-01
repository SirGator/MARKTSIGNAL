from __future__ import annotations

import unittest

from dataset.schema import (
    DatasetRecord,
    EconomicContext,
    Event,
    Metadata,
    StateDelta,
    Target,
)
from training.bridge import record_to_context_bundle, serialize_record
from training.input_contract import (
    DATASET_PROJECTION_VERSION,
    dataset_projection_metadata,
    validate_dataset_projection,
    validate_dataset_projection_metadata,
)


def _record(*, unit: str | None = None) -> DatasetRecord:
    return DatasetRecord(
        id="contract-test",
        event=Event(
            mechanism="input_cost_change",
            subject_class="commodity",
            subject="steel",
            direction="increase",
            magnitude=0.125,
            unit=unit,
        ),
        target=Target(entity_id="company:test"),
        context=EconomicContext(input_dependency=0.25, hedging=0.0),
        horizon_days=30,
        state_delta=StateDelta(
            revenue_expectation=0.0,
            demand=0.0,
            operating_costs=0.1,
            margin_expectation=-0.1,
            supply_risk=0.0,
            competitive_position=0.0,
        ),
        economic_impact=-0.1,
        confidence=0.9,
        metadata=Metadata(source="test", generator_version="test-v1", seed=1),
    )


class InputContractTests(unittest.TestCase):
    def test_projection_preserves_ratio_scale_and_observed_zero(self) -> None:
        record = _record()
        bundle = record_to_context_bundle(record)
        text = serialize_record(record)

        self.assertEqual(bundle.event.magnitude, 0.125)
        self.assertEqual(bundle.event.unit, "fraction")
        self.assertIn("magnitude=0.125", text)
        self.assertNotIn("magnitude=12.5", text)
        self.assertIn("predicate=input_dependency; value=0.25; unit=ratio", text)
        self.assertIn("predicate=hedging; value=0.0; unit=ratio", text)
        self.assertNotIn("predicate=pricing_power", text)

    def test_ambiguous_percent_unit_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            validate_dataset_projection(_record(unit="percent"))

    def test_behavioral_projection_metadata_is_stable(self) -> None:
        metadata = dataset_projection_metadata()

        self.assertEqual(
            metadata["dataset_projection_version"],
            DATASET_PROJECTION_VERSION,
        )
        validate_dataset_projection_metadata(metadata)

        tampered = dict(metadata)
        tampered["event_magnitude_scale"] = "percent"
        with self.assertRaisesRegex(ValueError, "event_magnitude_scale"):
            validate_dataset_projection_metadata(tampered)

    def test_projection_probe_suite_covers_all_mechanisms_and_factors(self) -> None:
        from dataset.mechanisms import list_mechanisms
        from dataset.schema import CONTEXT_FACTOR_FIELDS
        from training.input_contract import _projection_probe_texts

        texts = "\n".join(_projection_probe_texts())

        for mechanism in list_mechanisms():
            self.assertIn(f"type={mechanism}", texts, mechanism)
        for factor in CONTEXT_FACTOR_FIELDS:
            self.assertIn(f"predicate={factor}", texts, factor)
        self.assertIn("unit=ratio", texts)
        self.assertIn("unit=fraction", texts)
        self.assertIn("summary=[NO_SUMMARY]", texts)

    def test_projection_fingerprint_detects_serializer_changes(self) -> None:
        from unittest.mock import patch

        import training.input_contract as input_contract

        original = input_contract.dataset_projection_fingerprint()
        with patch.object(
            input_contract,
            "_projection_probe_texts",
            return_value=("",),
        ):
            tampered = input_contract.dataset_projection_fingerprint()
        self.assertNotEqual(original, tampered)

    def test_legacy_projection_declaration_is_explicit(self) -> None:
        legacy_config = {
            "dataset_schema_version": "1.0",
        }

        with self.assertRaisesRegex(ValueError, "input contract"):
            validate_dataset_projection_metadata(legacy_config)
        validate_dataset_projection_metadata(
            legacy_config,
            legacy_projection=True,
        )

        partial = dict(
            legacy_config,
            dataset_projection_version="dataset-record-to-context-v2",
        )
        with self.assertRaisesRegex(ValueError, "partially versioned"):
            validate_dataset_projection_metadata(
                partial,
                legacy_projection=True,
            )

    def test_frozen_runtime_views_are_deterministic_and_label_preserving(self) -> None:
        from training.runtime_views import add_frozen_runtime_views

        record = _record()
        first = add_frozen_runtime_views([record], ratio=1.0, seed=7)
        second = add_frozen_runtime_views([record], ratio=1.0, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(first[0], record)
        self.assertEqual(len(first), 2)
        self.assertEqual(first[1].score, record.score)
        self.assertEqual(first[1].context, record.context)
        self.assertIsNone(record.event.summary)
        self.assertIsNotNone(first[1].event.summary)
        self.assertTrue(first[1].id.endswith("::runtime-view-v1"))


if __name__ == "__main__":
    unittest.main()

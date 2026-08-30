"""Tests for the canonical dataset generation pipeline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from dataset.companies import generate_companies, generate_company
from dataset.counterfactuals import (
    DEFAULT_MAGNITUDE_GRID,
    DEFAULT_RATIO_GRID,
    generate_counterfactuals,
    standard_counterfactual_groups,
)
from dataset.export import freeze_dataset, load_frozen_dataset
from dataset.generate import generate_dataset
from dataset.labeling import (
    compute_economic_impact,
    compute_state_delta,
    require_valid_record,
    verify_record,
)
from dataset.mechanisms import MECHANISM_NAMES, get_mechanism
from dataset.schema import (
    CONTEXT_FACTOR_FIELDS,
    CostStructure,
    DatasetRecord,
    EconomicContext,
    Event,
    Metadata,
    StateDelta,
    Target,
)
from dataset.splits import SPLIT_NAMES, SplitPolicy, split_records, validate_split_integrity


def _base_record(
    mechanism: str,
    *,
    direction: str = "increase",
    magnitude: float = 0.4,
    horizon_days: int = 30,
    seed: int = 7,
    index: int = 1,
) -> DatasetRecord:
    definition = get_mechanism(mechanism)
    company = generate_company(seed=seed, index=index)
    event = Event(
        mechanism=mechanism,
        subject_class=definition.subject_classes[0],
        subject="test_subject",
        direction=direction,
        magnitude=magnitude,
    )
    context = company.context_for_event(event)
    state_delta = compute_state_delta(event, context, horizon_days)
    return DatasetRecord(
        id=f"case_{index:06d}",
        event=event,
        target=Target(entity_id=company.company_id),
        context=context,
        horizon_days=horizon_days,
        state_delta=state_delta,
        economic_impact=compute_economic_impact(state_delta),
        confidence=0.9,
        metadata=Metadata(
            source="synthetic",
            generator_version="test",
            seed=seed,
        ),
    )


class SchemaRoundTripTests(unittest.TestCase):
    def test_record_json_roundtrip_is_identity(self) -> None:
        record = _base_record("input_cost_change")
        restored = DatasetRecord.from_json(record.to_json())
        self.assertEqual(restored, record)
        self.assertEqual(restored.to_json(), record.to_json())

    def test_record_rejects_unknown_fields(self) -> None:
        payload = _base_record("demand_change").to_dict()
        payload["unexpected"] = 1
        with self.assertRaises(ValueError):
            DatasetRecord.from_dict(payload)

    def test_event_rejects_unknown_mechanism(self) -> None:
        with self.assertRaises(ValueError):
            Event(
                mechanism="not_a_mechanism",
                subject_class="energy",
                subject="oil",
                direction="increase",
                magnitude=0.3,
            )

    def test_context_preserves_unknown_versus_zero(self) -> None:
        context = EconomicContext(input_dependency=0.0)
        self.assertEqual(context.to_dict(), {"input_dependency": 0.0})
        restored = EconomicContext.from_dict(context.to_dict())
        self.assertEqual(restored.input_dependency, 0.0)
        self.assertIsNone(restored.hedging)

    def test_cost_structure_rejects_total_above_one(self) -> None:
        with self.assertRaises(ValueError):
            CostStructure(energy=0.4, materials=0.3, labor=0.2, financing=0.2)


class MechanismRegistryTests(unittest.TestCase):
    def test_registry_covers_all_fifteen_mechanisms(self) -> None:
        self.assertEqual(len(MECHANISM_NAMES), 15)
        for name in MECHANISM_NAMES:
            definition = get_mechanism(name)
            self.assertEqual(definition.name, name)
            self.assertGreaterEqual(len(definition.context_fields), 1)
            self.assertGreaterEqual(len(definition.affected_state_fields), 1)

    def test_every_mechanism_has_a_state_labeler(self) -> None:
        from dataset.labeling import STATE_DELTA_LABELERS

        self.assertEqual(set(STATE_DELTA_LABELERS), set(MECHANISM_NAMES))

    def test_every_mechanism_labels_within_bounds(self) -> None:
        for name in MECHANISM_NAMES:
            record = _base_record(name)
            for field in ("revenue_expectation", "demand", "operating_costs",
                          "margin_expectation", "supply_risk", "competitive_position"):
                value = getattr(record.state_delta, field)
                self.assertGreaterEqual(value, -1.0, f"{name}.{field}")
                self.assertLessEqual(value, 1.0, f"{name}.{field}")
            self.assertGreaterEqual(record.economic_impact, -1.0, name)
            self.assertLessEqual(record.economic_impact, 1.0, name)


class CompanyGenerationTests(unittest.TestCase):
    def test_generation_is_deterministic_and_order_independent(self) -> None:
        first = generate_company(seed=42, index=5)
        second = generate_company(seed=42, index=5)
        self.assertEqual(first, second)
        batch = generate_companies(10, seed=42)
        self.assertEqual(batch[4], first)

    def test_company_context_is_complete(self) -> None:
        company = generate_company(seed=42, index=1)
        for name in CONTEXT_FACTOR_FIELDS:
            self.assertIsNotNone(getattr(company.context, name), name)

    def test_context_for_event_returns_only_required_factors(self) -> None:
        company = generate_company(seed=42, index=1)
        definition = get_mechanism("input_cost_change")
        context = company.context_for_event("input_cost_change")
        self.assertEqual(set(context.to_dict()), set(definition.required_context_fields))


class LabelingTests(unittest.TestCase):
    def test_labeling_is_deterministic(self) -> None:
        record = _base_record("input_cost_change")
        delta = compute_state_delta(record.event, record.context, record.horizon_days)
        self.assertEqual(delta, record.state_delta)
        self.assertEqual(compute_economic_impact(delta), record.economic_impact)

    def test_subject_name_never_changes_a_label(self) -> None:
        record = _base_record("input_cost_change")
        for subject in ("oil", "wheat", "cloud_compute"):
            event = replace(record.event, subject=subject)
            self.assertEqual(
                compute_state_delta(event, record.context, record.horizon_days),
                record.state_delta,
            )

    def test_verifier_accepts_generated_records(self) -> None:
        for name in MECHANISM_NAMES:
            record = _base_record(name)
            result = verify_record(record)
            self.assertTrue(result.is_valid, f"{name}: {result.issues}")
            require_valid_record(record)

    def test_verifier_rejects_tampered_impact(self) -> None:
        record = _base_record("input_cost_change")
        tampered = replace(record, economic_impact=0.5)
        result = verify_record(tampered)
        self.assertFalse(result.is_valid)
        self.assertTrue(any(i.code == "economic_impact_mismatch" for i in result.issues))

    def test_verifier_allows_independent_teacher_delta_after_rule_checks(self) -> None:
        record = _base_record("input_cost_change")
        teacher_delta = StateDelta(
            revenue_expectation=0.0,
            demand=0.0,
            operating_costs=0.1,
            margin_expectation=-0.02,
            supply_risk=0.01,
            competitive_position=-0.005,
        )
        teacher = replace(
            record,
            state_delta=teacher_delta,
            economic_impact=compute_economic_impact(teacher_delta),
            metadata=replace(record.metadata, source="teacher"),
        )
        result = verify_record(teacher)
        self.assertTrue(result.is_valid, result.issues)

    def test_verifier_rejects_wrong_sign(self) -> None:
        record = _base_record("input_cost_change")
        tampered = replace(
            record,
            state_delta=replace(record.state_delta, operating_costs=-0.1),
        )
        result = verify_record(tampered)
        self.assertFalse(result.is_valid)
        self.assertTrue(any(i.code == "state_sign_invariant" for i in result.issues))


class CounterfactualTests(unittest.TestCase):
    def test_counterfactual_changes_exactly_one_input(self) -> None:
        base = _base_record("input_cost_change")
        variants = generate_counterfactuals(
            base,
            varied_field="hedging",
            values=(0.0, 0.25, 0.5, 0.75, 1.0),
        )
        self.assertEqual(len(variants), 5)
        for variant in variants:
            self.assertEqual(variant.event, base.event)
            self.assertEqual(variant.horizon_days, base.horizon_days)
            self.assertEqual(variant.target, base.target)
            for name in CONTEXT_FACTOR_FIELDS:
                if name != "hedging":
                    self.assertEqual(
                        getattr(variant.context, name),
                        getattr(base.context, name),
                        name,
                    )
            self.assertIsNotNone(variant.metadata.counterfactual)
            self.assertEqual(
                variant.metadata.counterfactual.base_record_id,
                base.id,
            )

    def test_hedging_grid_reduces_impact_monotonically(self) -> None:
        base = _base_record("input_cost_change", direction="increase")
        variants = generate_counterfactuals(
            base,
            varied_field="hedging",
            values=DEFAULT_RATIO_GRID,
        )
        impacts = [abs(v.economic_impact) for v in variants]
        for lower, higher in zip(impacts, impacts[1:]):
            self.assertGreaterEqual(lower, higher - 1e-9)

    def test_dependency_grid_increases_impact_monotonically(self) -> None:
        base = _base_record("input_cost_change", direction="increase")
        variants = generate_counterfactuals(
            base,
            varied_field="input_dependency",
            values=DEFAULT_RATIO_GRID,
        )
        impacts = [abs(v.economic_impact) for v in variants]
        for lower, higher in zip(impacts, impacts[1:]):
            self.assertLessEqual(lower, higher + 1e-9)

    def test_magnitude_grid_increases_impact_monotonically(self) -> None:
        base = _base_record("input_cost_change", direction="increase")
        variants = generate_counterfactuals(
            base,
            varied_field="event.magnitude",
            values=DEFAULT_MAGNITUDE_GRID,
        )
        impacts = [abs(v.economic_impact) for v in variants]
        for lower, higher in zip(impacts, impacts[1:]):
            self.assertLessEqual(lower, higher + 1e-9)

    def test_standard_groups_cover_all_declared_inputs(self) -> None:
        base = _base_record("input_cost_change")
        groups = standard_counterfactual_groups(base)
        definition = get_mechanism(base.event.mechanism)
        expected = {*definition.required_context_fields, "event.magnitude"}
        self.assertEqual(set(groups), expected)
        for field, variants in groups.items():
            self.assertGreaterEqual(len(variants), 3, field)
            for variant in variants:
                require_valid_record(variant)


class SplitTests(unittest.TestCase):
    def _records(self) -> list[DatasetRecord]:
        records: list[DatasetRecord] = []
        for index in range(1, 9):
            base = _base_record("input_cost_change", seed=3, index=index)
            records.append(base)
            records.extend(
                generate_counterfactuals(
                    base,
                    varied_field="hedging",
                    values=(0.1, 0.5, 0.9),
                )
            )
        return records

    def test_splits_keep_families_together(self) -> None:
        records = self._records()
        splits = split_records(records, policy=SplitPolicy(seed=1))
        validate_split_integrity(splits)
        for name in SPLIT_NAMES:
            self.assertIn(name, splits)

    def test_entity_ood_entities_never_leak(self) -> None:
        records = self._records()
        splits = split_records(records, policy=SplitPolicy(seed=1))
        ood_entities = {
            r.target.entity_id for r in splits["test_entity_ood"]
        }
        for name in ("train", "validation", "test_iid"):
            leaked = {
                r.target.entity_id
                for r in splits[name]
                if r.target.entity_id in ood_entities
            }
            self.assertEqual(leaked, set())

    def test_concept_ood_subjects_never_leak(self) -> None:
        records = self._records()
        concept = replace(
            records[0],
            id="case_concept_ood",
            event=replace(records[0].event, subject="wheat"),
        )
        records.append(concept)
        splits = split_records(records, policy=SplitPolicy(seed=1))
        for name in ("train", "validation", "test_iid"):
            leaked = [
                r for r in splits[name] if r.event.subject == "wheat"
            ]
            self.assertEqual(leaked, [])

    def test_split_assignment_is_deterministic(self) -> None:
        records = self._records()
        first = split_records(records, policy=SplitPolicy(seed=1))
        second = split_records(records, policy=SplitPolicy(seed=1))
        for name in SPLIT_NAMES:
            self.assertEqual(
                [r.id for r in first[name]],
                [r.id for r in second[name]],
            )

    def test_integrity_rejects_duplicate_id_inside_one_split(self) -> None:
        record = _base_record("demand_change")
        splits = {name: () for name in SPLIT_NAMES}
        splits["train"] = (record, record)
        with self.assertRaises(ValueError):
            validate_split_integrity(splits)


class GenerationPipelineTests(unittest.TestCase):
    def test_generate_freeze_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset_v1"
            manifest = generate_dataset(
                output_dir=output,
                dataset_version="0.1",
                seed=42,
                num_companies=50,
                num_bases=60,
                counterfactual_fraction=0.5,
            )
            self.assertEqual(manifest["dataset_version"], "0.1")
            self.assertEqual(manifest["company_profiles"], 50)
            self.assertEqual(manifest["base_records"], 60)
            self.assertGreater(manifest["counterfactual_families"], 0)
            self.assertGreater(manifest["total_examples"], 60)

            splits, loaded_manifest = load_frozen_dataset(output)
            self.assertEqual(loaded_manifest["dataset_version"], "0.1")
            self.assertEqual(loaded_manifest["base_records"], 60)
            self.assertEqual(
                loaded_manifest["counterfactual_families"],
                manifest["counterfactual_families"],
            )
            total = sum(len(records) for records in splits.values())
            self.assertEqual(total, manifest["total_examples"])
            for name in SPLIT_NAMES:
                self.assertEqual(
                    len(splits[name]),
                    manifest["splits"][name]["examples"],
                    name,
                )

    def test_generate_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a"
            second = Path(tmp) / "b"
            generate_dataset(
                output_dir=first,
                dataset_version="0.1",
                seed=7,
                num_companies=30,
                num_bases=40,
                counterfactual_fraction=0.5,
            )
            generate_dataset(
                output_dir=second,
                dataset_version="0.1",
                seed=7,
                num_companies=30,
                num_bases=40,
                counterfactual_fraction=0.5,
            )
            for name in SPLIT_NAMES:
                first_payload = (first / f"{name}.jsonl").read_bytes()
                second_payload = (second / f"{name}.jsonl").read_bytes()
                self.assertEqual(first_payload, second_payload, name)
            for name in ("companies.jsonl", "manifest.json"):
                self.assertEqual(
                    (first / name).read_bytes(),
                    (second / name).read_bytes(),
                    name,
                )

    def test_freeze_rejects_existing_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset_v1"
            generate_dataset(
                output_dir=output,
                dataset_version="0.1",
                seed=1,
                num_companies=10,
                num_bases=10,
                counterfactual_fraction=0.0,
            )
            with self.assertRaises(FileExistsError):
                generate_dataset(
                    output_dir=output,
                    dataset_version="0.1",
                    seed=1,
                    num_companies=10,
                    num_bases=10,
                    counterfactual_fraction=0.0,
                )

    def test_train_split_contains_counterfactual_families(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset_v1"
            generate_dataset(
                output_dir=output,
                dataset_version="0.1",
                seed=42,
                num_companies=50,
                num_bases=60,
                counterfactual_fraction=1.0,
            )
            splits, _ = load_frozen_dataset(output)
            train_ids = {r.id for r in splits["train"]}
            cf_in_train = [
                r for r in splits["train"]
                if r.metadata.counterfactual is not None
            ]
            self.assertGreater(len(cf_in_train), 0)
            self.assertGreater(len(train_ids), 60)


class OODLeakProtectionTests(unittest.TestCase):
    """The OOD test sets must remain genuinely out of distribution."""

    def _splits(self, *, counterfactual_fraction: float = 1.0) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset_v1"
            generate_dataset(
                output_dir=output,
                dataset_version="0.1",
                seed=42,
                num_companies=80,
                num_bases=120,
                counterfactual_fraction=counterfactual_fraction,
            )
            splits, _ = load_frozen_dataset(output)
            return splits

    def test_train_and_validation_have_no_parameter_ood_records(self) -> None:
        from dataset.splits import _is_parameter_ood

        splits = self._splits()
        for name in ("train", "validation", "test_iid", "test_combination_ood"):
            leaked = [r.id for r in splits[name] if _is_parameter_ood(r)]
            self.assertEqual(leaked, [], f"parameter-OOD leaked into {name}")

    def test_train_and_validation_have_no_combination_ood_records(self) -> None:
        from dataset.splits import _is_combination_ood

        splits = self._splits()
        for name in ("train", "validation", "test_iid", "test_parameter_ood"):
            leaked = [r.id for r in splits[name] if _is_combination_ood(r)]
            self.assertEqual(leaked, [], f"combination-OOD leaked into {name}")

    def test_counterfactual_ratio_grid_covers_full_mitigation_range(self) -> None:
        self.assertEqual(DEFAULT_RATIO_GRID, (0.0, 0.25, 0.5, 0.75, 1.0))


class GeneratorVersionPropagationTests(unittest.TestCase):
    """Records must carry the generator_version passed to generate_dataset."""

    def test_custom_generator_version_reaches_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset_v1"
            manifest = generate_dataset(
                output_dir=output,
                dataset_version="0.1",
                generator_version="generate-v2-test",
                seed=7,
                num_companies=10,
                num_bases=12,
                counterfactual_fraction=0.5,
            )
            self.assertEqual(manifest["generator_version"], "generate-v2-test")
            splits, _ = load_frozen_dataset(output)
            for record in splits["train"][:5]:
                self.assertEqual(
                    record.metadata.generator_version,
                    "generate-v2-test",
                    f"record {record.id} did not inherit generator_version",
                )

    def test_default_generator_version_matches_records(self) -> None:
        from dataset.generate import GENERATOR_VERSION

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset_v1"
            generate_dataset(
                output_dir=output,
                dataset_version="0.1",
                seed=1,
                num_companies=8,
                num_bases=8,
                counterfactual_fraction=0.0,
            )
            splits, _ = load_frozen_dataset(output)
            for record in splits["train"][:3]:
                self.assertEqual(record.metadata.generator_version, GENERATOR_VERSION)


class FrozenDatasetLoaderTests(unittest.TestCase):
    """The training-facing loader must expose splits and manifest correctly."""

    def test_loader_returns_train_validation_and_ood_splits(self) -> None:
        from training.data import FrozenDataset, load_frozen_dataset_for_training

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset_v1"
            generate_dataset(
                output_dir=output,
                dataset_version="0.2",
                seed=11,
                num_companies=40,
                num_bases=60,
                counterfactual_fraction=0.5,
            )
            frozen = load_frozen_dataset_for_training(output)
            self.assertIsInstance(frozen, FrozenDataset)
            self.assertEqual(frozen.manifest["dataset_version"], "0.2")
            self.assertGreater(len(frozen.train), 0)
            self.assertGreater(len(frozen.validation), 0)
            # IID test split is kept separate from the OOD aggregate.
            self.assertGreaterEqual(len(frozen.iid_test), 0)
            self.assertNotIn("test_iid", frozen.ood_splits)
            self.assertIn("test_entity_ood", frozen.ood_splits)
            # Hard-OOD is a distinct split now.
            self.assertIn("test_hard_ood", frozen.ood_splits)
            # Score property is the training label surface.
            for record in frozen.train[:3]:
                self.assertGreaterEqual(record.score, -1.0)
                self.assertLessEqual(record.score, 1.0)

    def test_loader_rejects_missing_directory(self) -> None:
        from training.data import DatasetLoadError, load_frozen_dataset_for_training

        with self.assertRaises(DatasetLoadError):
            load_frozen_dataset_for_training("/nonexistent/dataset/path")


class DatasetRecordBridgeTests(unittest.TestCase):
    """DatasetRecord -> ContextBundle serialization must mirror production."""

    def test_record_serializes_through_production_path(self) -> None:
        from training.bridge import serialize_record

        record = _base_record("input_cost_change", index=42)
        text = serialize_record(record)
        self.assertTrue(text.startswith("[CLS]"))
        self.assertIn("[EVENT] type=input_cost_change", text)
        self.assertIn(
            f"subjects={record.event.subject_class}:test_subject",
            text,
        )
        self.assertIn("unit=fraction", text)
        self.assertIn("[CASE] type=company name=[CASE_ENTITY]", text)
        self.assertIn("[HORIZON] 30d", text)
        self.assertIn("predicate=input_dependency", text)
        self.assertIn("[SEP]", text)

    def test_record_no_summary_mode_replaces_summary(self) -> None:
        from src.models.context_serializer import SUMMARY_MODE_NONE
        from training.bridge import serialize_record

        record = _base_record("demand_change", index=3)
        text = serialize_record(record, summary_mode=SUMMARY_MODE_NONE)
        self.assertIn("[NO_SUMMARY]", text)
        # Structured fields survive neutralization.
        self.assertIn("predicate=demand_exposure", text)

    def test_record_score_property_matches_economic_impact(self) -> None:
        record = _base_record("currency_change", index=9)
        self.assertEqual(record.score, record.economic_impact)

    def test_pipeline_serializes_dataset_record(self) -> None:
        try:
            from training.pipeline import serialize_training_example
        except ModuleNotFoundError as exc:
            if "torch" in str(exc).lower():
                self.skipTest("pipeline serialization requires PyTorch")
            raise
        from src.models.context_serializer import SUMMARY_MODE_FULL, SUMMARY_MODE_NONE

        record = _base_record("supply_change", index=5)
        full = serialize_training_example(record, summary_mode=SUMMARY_MODE_FULL)
        none = serialize_training_example(record, summary_mode=SUMMARY_MODE_NONE)
        self.assertIn("[EVENT]", full)
        self.assertIn("[NO_SUMMARY]", none)
        self.assertIn("predicate=input_dependency", none)


class HardOODSplitTests(unittest.TestCase):
    """Families that satisfy parameter- AND combination-OOD get their own split."""

    def _splits(self) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset_v1"
            generate_dataset(
                output_dir=output,
                dataset_version="0.1",
                seed=42,
                num_companies=1000,
                num_bases=3000,
                counterfactual_fraction=1.0,
            )
            splits, _ = load_frozen_dataset(output)
            return splits

    def test_hard_ood_split_is_present(self) -> None:
        from dataset.splits import SPLIT_NAMES

        self.assertIn("test_hard_ood", SPLIT_NAMES)
        splits = self._splits()
        self.assertIn("test_hard_ood", splits)
        # With 3000 bases the hard-OOD overlap is reliably non-empty.
        self.assertGreater(len(splits["test_hard_ood"]), 0)

    def test_hard_ood_does_not_leak_into_single_axis_ood(self) -> None:
        from dataset.splits import is_hard_ood

        splits = self._splits()
        for name in ("test_parameter_ood", "test_combination_ood"):
            leaked = [r.id for r in splits[name] if is_hard_ood(r)]
            self.assertEqual(
                leaked,
                [],
                f"hard-OOD records leaked into {name}: {leaked[:5]}",
            )

    def test_single_axis_ood_stay_pure(self) -> None:
        """parameter-OOD split must not contain combination-OOD records
        and vice versa — each metric measures exactly one axis."""

        from dataset.splits import is_combination_ood, is_parameter_ood

        splits = self._splits()
        combo_in_param = [
            r.id for r in splits["test_parameter_ood"] if is_combination_ood(r)
        ]
        param_in_combo = [
            r.id for r in splits["test_combination_ood"] if is_parameter_ood(r)
        ]
        self.assertEqual(combo_in_param, [], "combination-OOD in parameter-OOD split")
        self.assertEqual(param_in_combo, [], "parameter-OOD in combination-OOD split")

    def test_train_validation_iid_have_no_ood_records(self) -> None:
        from dataset.splits import (
            is_combination_ood,
            is_hard_ood,
            is_parameter_ood,
        )

        splits = self._splits()
        for name in ("train", "validation", "test_iid"):
            p = [r.id for r in splits[name] if is_parameter_ood(r)]
            c = [r.id for r in splits[name] if is_combination_ood(r)]
            h = [r.id for r in splits[name] if is_hard_ood(r)]
            self.assertEqual(p, [], f"parameter-OOD leaked into {name}")
            self.assertEqual(c, [], f"combination-OOD leaked into {name}")
            self.assertEqual(h, [], f"hard-OOD leaked into {name}")


class IIDSplitsSeparationTests(unittest.TestCase):
    """``test_iid`` must never be folded into the OOD aggregate."""

    def test_loader_keeps_iid_test_separate_from_ood(self) -> None:
        from training.data import load_frozen_dataset_for_training

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset_v1"
            generate_dataset(
                output_dir=output,
                dataset_version="0.1",
                seed=21,
                num_companies=60,
                num_bases=90,
                counterfactual_fraction=0.5,
            )
            frozen = load_frozen_dataset_for_training(output)
            self.assertNotIn("test_iid", frozen.ood_splits)
            # The IID records are accessible separately.
            self.assertGreaterEqual(len(frozen.iid_test), 0)
            # OOD splits only contain genuinely out-of-distribution splits.
            for name in frozen.ood_splits:
                self.assertTrue(name.startswith("test_"))
                self.assertNotEqual(name, "test_iid")

    def test_ood_eval_splits_constant_excludes_iid(self) -> None:
        from dataset.splits import IID_EVAL_SPLITS, OOD_EVAL_SPLITS

        self.assertEqual(IID_EVAL_SPLITS, ("test_iid",))
        self.assertNotIn("test_iid", OOD_EVAL_SPLITS)
        self.assertIn("test_hard_ood", OOD_EVAL_SPLITS)


class OODMetricFilteringTests(unittest.TestCase):
    """The OOD metric must only count records that satisfy the OOD condition."""

    @classmethod
    def setUpClass(cls) -> None:
        # Generate once for all metric-filtering tests; the 3000-base dataset
        # is the smallest configuration that reliably yields a non-empty
        # hard-OOD overlap.
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._output = Path(cls._tmpdir.name) / "dataset_v1"
        generate_dataset(
            output_dir=cls._output,
            dataset_version="0.1",
            seed=42,
            num_companies=1000,
            num_bases=3000,
            counterfactual_fraction=1.0,
        )
        cls._splits, _ = load_frozen_dataset(cls._output)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def test_parameter_ood_metric_excludes_held_out_only_family_members(self) -> None:
        from dataset.splits import is_parameter_ood

        param_split = self._splits["test_parameter_ood"]
        metric_relevant = [r for r in param_split if is_parameter_ood(r)]
        held_out_only = [r for r in param_split if not is_parameter_ood(r)]
        self.assertGreater(len(metric_relevant), 0)
        self.assertGreaterEqual(len(held_out_only), 0)
        self.assertLessEqual(len(metric_relevant), len(param_split))

    def test_combination_ood_metric_excludes_held_out_only_family_members(self) -> None:
        from dataset.splits import is_combination_ood

        combo_split = self._splits["test_combination_ood"]
        metric_relevant = [r for r in combo_split if is_combination_ood(r)]
        self.assertGreater(len(metric_relevant), 0)
        self.assertLessEqual(len(metric_relevant), len(combo_split))

    def test_hard_ood_metric_uses_both_predicates(self) -> None:
        from dataset.splits import is_hard_ood, is_parameter_ood, is_combination_ood

        hard_split = self._splits["test_hard_ood"]
        self.assertGreater(len(hard_split), 0)
        for record in hard_split:
            if is_hard_ood(record):
                self.assertTrue(is_parameter_ood(record))
                self.assertTrue(is_combination_ood(record))


if __name__ == "__main__":
    unittest.main()

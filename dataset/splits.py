"""Leakage-resistant, deterministic dataset split assignment."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from typing import Iterable, Mapping

from .schema import DatasetRecord


SPLIT_NAMES = (
    "train",
    "validation",
    "test_iid",
    "test_entity_ood",
    "test_parameter_ood",
    "test_combination_ood",
    "test_hard_ood",
    "test_concept_ood",
)

# Splits used by the training loop as labeled training data (everything else
# is held out for evaluation).  Kept here so the loader and the validator can
# agree on the contract without re-hard-coding the tuple.
TRAIN_SPLIT_NAMES = ("train", "validation")

# Evaluation-only splits that are genuinely in-distribution.  These never
# belong to the OOD aggregate metric.
IID_EVAL_SPLITS = ("test_iid",)

# Out-of-distribution evaluation splits.  ``test_hard_ood`` collects families
# that simultaneously satisfy parameter- and combination-OOD, so the remaining
# OOD splits stay pure and each metric measures exactly one axis.
OOD_EVAL_SPLITS = (
    "test_entity_ood",
    "test_parameter_ood",
    "test_combination_ood",
    "test_hard_ood",
    "test_concept_ood",
)

DEFAULT_CONCEPT_OOD_SUBJECTS = frozenset(
    ("wheat", "electricity", "freight", "cloud_compute", "semiconductor_wafers")
)


@dataclass(frozen=True, slots=True)
class SplitPolicy:
    """Versionable rules applied before any random IID allocation."""

    seed: int = 42
    validation_fraction: float = 0.1
    iid_test_fraction: float = 0.1
    entity_ood_fraction: float = 0.1
    concept_ood_subjects: frozenset[str] = DEFAULT_CONCEPT_OOD_SUBJECTS

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        for name in (
            "validation_fraction",
            "iid_test_fraction",
            "entity_ood_fraction",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not 0.0 <= float(value) < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
            object.__setattr__(self, name, float(value))
        if self.validation_fraction + self.iid_test_fraction >= 1.0:
            raise ValueError("validation and IID test fractions must sum to less than 1")
        subjects = frozenset(_subject_token(value) for value in self.concept_ood_subjects)
        object.__setattr__(self, "concept_ood_subjects", subjects)


def split_records(
    records: Iterable[DatasetRecord],
    policy: SplitPolicy | None = None,
) -> dict[str, tuple[DatasetRecord, ...]]:
    """Assign whole entities and counterfactual families without leakage."""

    policy = policy or SplitPolicy()
    materialized = tuple(records)
    _validate_unique_records(materialized)

    entities = sorted({record.target.entity_id for record in materialized})
    held_out_entities = _select_entity_ood(entities, policy)
    families: dict[str, list[DatasetRecord]] = defaultdict(list)
    for record in materialized:
        families[_family_key(record)].append(record)

    result: dict[str, list[DatasetRecord]] = {name: [] for name in SPLIT_NAMES}
    for family_key in sorted(families):
        family = families[family_key]
        target_ids = {record.target.entity_id for record in family}
        if len(target_ids) != 1:
            raise ValueError(
                f"counterfactual family {family_key!r} spans multiple target entities"
            )

        if next(iter(target_ids)) in held_out_entities:
            split = "test_entity_ood"
        elif any(
            _subject_token(record.event.subject) in policy.concept_ood_subjects
            for record in family
        ):
            split = "test_concept_ood"
        else:
            # Parameter-OOD uses the family's base record only.  Controlled
            # counterfactuals deliberately cover the full [0, 1] range, while
            # previously unseen event-shock magnitudes are reserved for OOD.
            base_record = _base_record(family)
            parameter_ood = _is_parameter_ood(base_record)
            # A counterfactual variant can deliberately set a factor to an
            # extreme value.  If any family member matches the
            # combination-OOD rule, the whole family is held out so the
            # train split cannot memorize that exact rare combination.
            combination_ood = any(
                _is_combination_ood(record) for record in family
            )
            if parameter_ood and combination_ood:
                # Both OOD axes fire at once — keep the family in a dedicated
                # hard-OOD split so the single-axis OOD metrics stay pure.
                split = "test_hard_ood"
            elif parameter_ood:
                split = "test_parameter_ood"
            elif combination_ood:
                split = "test_combination_ood"
            else:
                draw = _unit_hash(policy.seed, f"family:{family_key}")
                if draw < policy.validation_fraction:
                    split = "validation"
                elif draw < policy.validation_fraction + policy.iid_test_fraction:
                    split = "test_iid"
                else:
                    split = "train"
        result[split].extend(family)

    return {
        name: tuple(sorted(result[name], key=lambda record: record.id))
        for name in SPLIT_NAMES
    }


def validate_split_integrity(
    splits: Mapping[str, Iterable[DatasetRecord]],
    *,
    concept_ood_subjects: frozenset[str] = DEFAULT_CONCEPT_OOD_SUBJECTS,
) -> None:
    """Reject duplicate records, family leakage, and OOD contamination."""

    unknown = set(splits) - set(SPLIT_NAMES)
    missing = set(SPLIT_NAMES) - set(splits)
    if unknown or missing:
        raise ValueError(
            f"split keys mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )

    id_to_split: dict[str, str] = {}
    family_to_split: dict[str, str] = {}
    entity_to_splits: dict[str, set[str]] = defaultdict(set)
    for split_name in SPLIT_NAMES:
        for record in splits[split_name]:
            if not isinstance(record, DatasetRecord):
                raise TypeError("splits must contain DatasetRecord values")
            previous = id_to_split.get(record.id)
            if previous is not None:
                raise ValueError(f"record {record.id!r} occurs in multiple splits")
            id_to_split[record.id] = split_name
            family = _family_key(record)
            previous_family = family_to_split.setdefault(family, split_name)
            if previous_family != split_name:
                raise ValueError(
                    f"counterfactual family {family!r} leaks across splits"
                )
            entity_to_splits[record.target.entity_id].add(split_name)

    entity_ood_ids = {
        record.target.entity_id for record in splits["test_entity_ood"]
    }
    for entity_id in entity_ood_ids:
        if entity_to_splits[entity_id] != {"test_entity_ood"}:
            raise ValueError(f"entity-OOD target {entity_id!r} leaks into another split")

    ood_subjects = {_subject_token(value) for value in concept_ood_subjects}
    for split_name in (
        "train",
        "validation",
        "test_iid",
        "test_parameter_ood",
        "test_combination_ood",
        "test_hard_ood",
    ):
        leaked = {
            record.event.subject
            for record in splits[split_name]
            if _subject_token(record.event.subject) in ood_subjects
        }
        if leaked:
            raise ValueError(
                f"concept-OOD subjects leaked into {split_name}: {sorted(leaked)}"
            )

    # Combination-OOD records may only live in the combination- or hard-OOD
    # splits.  Every non-OOD split and the single-axis parameter-OOD split must
    # stay clean so each metric measures exactly one axis.
    for split_name in (
        "train",
        "validation",
        "test_iid",
        "test_parameter_ood",
    ):
        leaked = [
            record.id
            for record in splits[split_name]
            if _is_combination_ood(record)
        ]
        if leaked:
            raise ValueError(
                f"combination-OOD records leaked into {split_name}: {leaked[:5]}"
            )

    # Parameter-OOD records may only live in the parameter- or hard-OOD splits.
    for split_name in (
        "train",
        "validation",
        "test_iid",
        "test_combination_ood",
    ):
        leaked = [
            record.id
            for record in splits[split_name]
            if _is_parameter_ood(record)
        ]
        if leaked:
            raise ValueError(
                f"parameter-OOD records leaked into {split_name}: {leaked[:5]}"
            )


def _validate_unique_records(records: tuple[DatasetRecord, ...]) -> None:
    if any(not isinstance(record, DatasetRecord) for record in records):
        raise TypeError("records must contain only DatasetRecord values")
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("record IDs must be globally unique")


def _select_entity_ood(entities: list[str], policy: SplitPolicy) -> frozenset[str]:
    if not entities or policy.entity_ood_fraction == 0:
        return frozenset()
    count = int(len(entities) * policy.entity_ood_fraction)
    if len(entities) > 1:
        count = max(1, min(count, len(entities) - 1))
    ranked = sorted(
        entities,
        key=lambda entity: (_unit_hash(policy.seed, f"entity:{entity}"), entity),
    )
    return frozenset(ranked[:count])


def _family_key(record: DatasetRecord) -> str:
    provenance = record.metadata.counterfactual
    return provenance.base_record_id if provenance is not None else record.id


def _base_record(family: list[DatasetRecord]) -> DatasetRecord:
    """Return the family's base record (the one without counterfactual provenance)."""

    bases = [record for record in family if record.metadata.counterfactual is None]
    if len(bases) != 1:
        raise ValueError(
            "counterfactual family must contain exactly one base record; "
            f"found {len(bases)}"
        )
    return bases[0]


def _is_parameter_ood(record: DatasetRecord) -> bool:
    return record.event.magnitude >= 0.98


def _is_combination_ood(record: DatasetRecord) -> bool:
    context = record.context
    pairs = (
        (context.input_dependency, context.hedging),
        (context.debt_ratio, context.cash_buffer),
        (context.pricing_power, context.competitive_pressure),
        (context.currency_revenue_exposure, context.currency_cost_exposure),
        (context.customer_concentration, context.pricing_power),
    )
    return any(
        first is not None
        and second is not None
        and first >= 0.75
        and second >= 0.75
        for first, second in pairs
    )


def is_parameter_ood(record: DatasetRecord) -> bool:
    """Public alias used by the evaluation metric filter."""

    return _is_parameter_ood(record)


def is_combination_ood(record: DatasetRecord) -> bool:
    """Public alias used by the evaluation metric filter."""

    return _is_combination_ood(record)


def is_hard_ood(record: DatasetRecord) -> bool:
    """A record that satisfies both the parameter- and combination-OOD rules."""

    return _is_parameter_ood(record) and _is_combination_ood(record)


def is_concept_ood(
    record: DatasetRecord,
    *,
    concept_ood_subjects: frozenset[str] = DEFAULT_CONCEPT_OOD_SUBJECTS,
) -> bool:
    return _subject_token(record.event.subject) in concept_ood_subjects


def is_entity_ood(record: DatasetRecord, *, entity_ood_ids: frozenset[str]) -> bool:
    return record.target.entity_id in entity_ood_ids


def _unit_hash(seed: int, value: str) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _subject_token(value: str) -> str:
    return "_".join(value.strip().casefold().replace("-", " ").split())


__all__ = [
    "DEFAULT_CONCEPT_OOD_SUBJECTS",
    "IID_EVAL_SPLITS",
    "OOD_EVAL_SPLITS",
    "SPLIT_NAMES",
    "TRAIN_SPLIT_NAMES",
    "SplitPolicy",
    "is_combination_ood",
    "is_concept_ood",
    "is_entity_ood",
    "is_hard_ood",
    "is_parameter_ood",
    "split_records",
    "validate_split_integrity",
]

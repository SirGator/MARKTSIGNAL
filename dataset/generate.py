"""End-to-end deterministic dataset generation.

Pipeline (canonical contract, see ``dataset/schema.py``)::

    companies (persistent structural profiles)
        -> events (mechanism + subject_class + subject + direction + magnitude)
        -> context (company factors required by the mechanism)
        -> state_delta (deterministic labeling)
        -> economic_impact (fixed aggregation)
        -> counterfactual families (one-variable grids)
        -> leakage-free splits
        -> frozen JSONL dataset + manifest

Everything is seeded and order-independent: identical CLI arguments always
produce an identical dataset directory.  Existing output directories are
rejected; changing generation rules requires a new versioned path.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
from pathlib import Path
from typing import Mapping

from dataset.companies import (
    COMPANY_GENERATOR_VERSION,
    SyntheticCompany,
    generate_companies,
)
from dataset.counterfactuals import standard_counterfactual_groups
from dataset.export import freeze_dataset
from dataset.labeling import compute_economic_impact, compute_state_delta
from dataset.mechanisms import MECHANISM_NAMES, get_mechanism
from dataset.schema import (
    DatasetRecord,
    EconomicContext,
    Event,
    Metadata,
    StateDelta,
    Target,
)
from dataset.splits import (
    DEFAULT_CONCEPT_OOD_SUBJECTS,
    SPLIT_NAMES,
    SplitPolicy,
    split_records,
)


GENERATOR_VERSION = "generate-v1"

HORIZONS = (7, 14, 30, 60, 90, 180, 365)

# Subject pools used during training.  Subjects are concrete instances of an
# abstract subject class; the model must learn the class-level mechanism, not
# the name.  Concept-OOD subjects are deliberately excluded from these pools.
TRAIN_SUBJECTS: dict[str, tuple[str, ...]] = {
    "production_input": (
        "oil",
        "gas",
        "copper",
        "steel",
        "lithium",
        "aluminum",
        "nickel",
        "rare_earth",
        "timber",
        "chemicals",
    ),
    "energy": ("oil", "gas", "coal", "diesel", "jet_fuel", "natural_gas"),
    "agricultural_input": ("corn", "soybeans", "cotton", "sugar", "palm_oil"),
    "materials": (
        "steel",
        "aluminum",
        "copper",
        "plastics",
        "glass",
        "cement",
        "paper",
        "packaging",
    ),
    "compute": ("gpu_compute", "server_capacity", "data_center_power"),
    "logistics": (
        "shipping",
        "trucking",
        "rail",
        "air_cargo",
        "port_services",
        "warehousing",
    ),
    "financing": (
        "interest_rate",
        "credit_spreads",
        "corporate_bonds",
        "bank_loans",
        "commercial_paper",
    ),
    "debt": ("corporate_bonds", "bank_loans", "commercial_paper"),
    "credit": ("credit_spreads", "bank_loans"),
    "interest_rate": ("interest_rate", "central_bank_rate"),
    "labor": (
        "skilled_workers",
        "engineers",
        "logistics_staff",
        "factory_workers",
        "software_developers",
    ),
    "workforce": ("skilled_workers", "factory_workers", "engineers"),
    "occupation": ("engineers", "software_developers", "healthcare_staff"),
    "compensation": ("wages", "benefits", "bonuses"),
    "output": (
        "steel",
        "aluminum",
        "copper",
        "cement",
        "chemicals",
        "paper",
        "glass",
        "plastics",
        "textiles",
        "packaging",
    ),
    "product": (
        "electric_vehicles",
        "semiconductors",
        "solar_panels",
        "consumer_electronics",
        "batteries",
        "wind_turbines",
        "smartphones",
        "machinery",
    ),
    "service": (
        "cloud_services",
        "logistics_services",
        "consulting",
        "software_subscriptions",
        "maintenance",
    ),
    "commodity": (
        "oil",
        "gas",
        "copper",
        "steel",
        "lithium",
        "aluminum",
        "nickel",
        "corn",
        "soybeans",
    ),
    "market": (
        "ev_market",
        "semiconductor_market",
        "steel_market",
        "energy_market",
        "consumer_market",
    ),
    "customer_segment": (
        "automotive_oems",
        "industrial_customers",
        "retail_consumers",
        "enterprise_clients",
        "construction_sector",
    ),
    "market_supply": (
        "oil_supply",
        "copper_supply",
        "steel_supply",
        "lithium_supply",
        "semiconductor_supply",
    ),
    "currency": ("usd", "eur", "jpy", "gbp", "chf", "aud", "cad", "brl", "inr", "cny"),
    "currency_pair": ("usd_eur", "usd_jpy", "usd_cny", "eur_gbp"),
    "foreign_exchange": ("usd", "eur", "jpy", "cny"),
    "competitor": (
        "low_cost_entrant",
        "foreign_competitor",
        "platform_competitor",
        "private_label",
        "substitute_product",
    ),
    "regulation": (
        "emissions_rule",
        "safety_standard",
        "data_privacy_law",
        "import_quota",
        "packaging_rule",
    ),
    "jurisdiction": ("eu", "us", "china", "germany", "california"),
    "compliance_requirement": (
        "emissions_reporting",
        "product_certification",
        "data_residency",
        "import_licensing",
    ),
    "tax": ("corporate_tax", "vat", "carbon_tax", "import_tariff", "windfall_tax"),
    "tax_regime": ("corporate_tax", "carbon_tax", "import_tariff"),
    "subsidy": (
        "ev_subsidy",
        "solar_subsidy",
        "export_credit",
        "r_and_d_credit",
        "energy_subsidy",
    ),
    "government_program": ("ev_subsidy", "solar_subsidy", "export_credit", "r_and_d_credit"),
    "company": (
        "smelter",
        "refinery",
        "fab",
        "assembly_plant",
        "warehouse",
        "data_center",
        "mine",
        "port",
        "fleet",
        "grid",
    ),
    "facility": (
        "smelter",
        "refinery",
        "fab",
        "assembly_plant",
        "warehouse",
        "data_center",
        "mine",
        "port",
    ),
    "production_line": ("assembly_line", "packaging_line", "chemical_line", "steel_line"),
    "network": ("logistics_network", "retail_network", "power_grid", "telecom_network"),
    "supplier": (
        "chip_supplier",
        "battery_supplier",
        "steel_supplier",
        "chemical_supplier",
        "logistics_provider",
        "component_supplier",
        "rare_earth_supplier",
        "packaging_supplier",
    ),
    "logistics_route": (
        "suez_route",
        "panama_route",
        "transpacific_route",
        "europe_rail_route",
    ),
    "customer": (
        "major_retailer",
        "oem_customer",
        "government_customer",
        "telecom_operator",
        "auto_manufacturer",
        "distributor",
        "ecommerce_platform",
    ),
    "distribution_channel": (
        "retail_channel",
        "ecommerce_channel",
        "wholesale_channel",
        "direct_sales",
    ),
    "contract": (
        "supply_contract",
        "offtake_agreement",
        "service_contract",
        "licensing_deal",
        "procurement_contract",
    ),
    "customer_contract": ("supply_contract", "service_contract", "licensing_deal"),
    "supplier_contract": ("procurement_contract", "offtake_agreement", "supply_contract"),
}

# Concept-OOD subjects: never seen during training.  The token set must match
# the split policy exactly so the leakage checks in dataset/splits.py hold.
CONCEPT_OOD_SUBJECTS: dict[str, str] = {
    "wheat": "agricultural_input",
    "electricity": "energy",
    "freight": "logistics",
    "cloud_compute": "compute",
    "semiconductor_wafers": "materials",
}


def _validate_subject_pools() -> None:
    if set(CONCEPT_OOD_SUBJECTS) != set(DEFAULT_CONCEPT_OOD_SUBJECTS):
        raise RuntimeError(
            "concept-OOD subjects must match the split policy exactly; "
            f"expected {sorted(DEFAULT_CONCEPT_OOD_SUBJECTS)}"
        )
    training_subjects = {
        subject
        for subjects in TRAIN_SUBJECTS.values()
        for subject in subjects
    }
    overlap = training_subjects & set(CONCEPT_OOD_SUBJECTS)
    if overlap:
        raise RuntimeError(
            f"concept-OOD subjects leaked into training pools: {sorted(overlap)}"
        )
    for mechanism_name in MECHANISM_NAMES:
        definition = get_mechanism(mechanism_name)
        for subject_class in definition.subject_classes:
            if subject_class not in TRAIN_SUBJECTS:
                raise RuntimeError(
                    f"no training subject pool for subject_class {subject_class!r} "
                    f"(mechanism {mechanism_name!r})"
                )


_validate_subject_pools()


def _unit_hash(seed: int, value: str) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _fraction(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _sample_magnitude(rng: random.Random) -> float:
    """Draw a normalized event magnitude with a rare extreme tail.

    The extreme tail (>= 0.98) feeds the parameter-OOD split so the model is
    tested on shock sizes it rarely saw during training.
    """
    if rng.random() < 0.08:
        return round(rng.uniform(0.7, 1.0), 2)
    return round(rng.uniform(0.05, 0.6), 2)


def _build_base_record(
    *,
    index: int,
    company: SyntheticCompany,
    mechanism_name: str,
    direction: str,
    subject_class: str,
    subject: str,
    magnitude: float,
    horizon_days: int,
    seed: int,
    generator_version: str,
) -> DatasetRecord:
    event = Event(
        mechanism=mechanism_name,
        subject_class=subject_class,
        subject=subject,
        direction=direction,
        magnitude=magnitude,
    )
    context = company.context_for_event(event)
    state_delta = compute_state_delta(event, context, horizon_days)
    economic_impact = compute_economic_impact(state_delta)
    metadata = Metadata(
        source="synthetic",
        generator_version=generator_version,
        seed=seed,
    )
    return DatasetRecord(
        id=f"case_{index:06d}",
        event=event,
        target=Target(entity_id=company.company_id),
        context=context,
        horizon_days=horizon_days,
        state_delta=state_delta,
        economic_impact=economic_impact,
        confidence=0.9,
        metadata=metadata,
    )


def generate_dataset(
    *,
    output_dir: Path | str,
    dataset_version: str = "1.0",
    generator_version: str = GENERATOR_VERSION,
    seed: int = 42,
    num_companies: int = 10_000,
    num_bases: int = 20_000,
    counterfactual_fraction: float = 1.0,
    concept_ood_fraction: float = 0.1,
    validation_fraction: float = 0.1,
    iid_test_fraction: float = 0.1,
    entity_ood_fraction: float = 0.1,
) -> dict[str, object]:
    """Generate, split, and freeze one complete dataset version.

    Returns the manifest written by :func:`dataset.export.freeze_dataset`.
    """

    canonical_seed = _integer("seed", seed)
    company_count = _integer("num_companies", num_companies, minimum=1)
    base_count = _integer("num_bases", num_bases, minimum=1)
    cf_fraction = _fraction("counterfactual_fraction", counterfactual_fraction)
    ood_fraction = _fraction("concept_ood_fraction", concept_ood_fraction)

    rng = random.Random(canonical_seed)
    companies = generate_companies(company_count, seed=canonical_seed)
    company_by_index = {index: company for index, company in enumerate(companies, start=1)}

    bases: list[DatasetRecord] = []
    for index in range(1, base_count + 1):
        company = company_by_index[rng.randrange(1, company_count + 1)]
        mechanism_name = rng.choice(MECHANISM_NAMES)
        definition = get_mechanism(mechanism_name)
        direction = rng.choice(("increase", "decrease"))
        subject_class = rng.choice(definition.subject_classes)
        if (
            subject_class in CONCEPT_OOD_SUBJECTS.values()
            and rng.random() < ood_fraction
        ):
            subject = next(
                name
                for name, owner_class in CONCEPT_OOD_SUBJECTS.items()
                if owner_class == subject_class
            )
        else:
            subject = rng.choice(TRAIN_SUBJECTS[subject_class])
        magnitude = _sample_magnitude(rng)
        horizon_days = rng.choice(HORIZONS)
        bases.append(
            _build_base_record(
                index=index,
                company=company,
                mechanism_name=mechanism_name,
                direction=direction,
                subject_class=subject_class,
                subject=subject,
                magnitude=magnitude,
                horizon_days=horizon_days,
                seed=canonical_seed,
                generator_version=generator_version,
            )
        )

    records: list[DatasetRecord] = list(bases)
    counterfactual_families = 0
    for base in bases:
        if _unit_hash(canonical_seed, f"counterfactual:{base.id}") >= cf_fraction:
            continue
        groups = standard_counterfactual_groups(base)
        for group in groups.values():
            records.extend(group)
        counterfactual_families += len(groups)

    policy = SplitPolicy(
        seed=canonical_seed,
        validation_fraction=validation_fraction,
        iid_test_fraction=iid_test_fraction,
        entity_ood_fraction=entity_ood_fraction,
    )
    splits = split_records(records, policy=policy)

    manifest = freeze_dataset(
        output_dir,
        splits,
        dataset_version=dataset_version,
        generator_version=generator_version,
        seed=canonical_seed,
        companies=[company.profile for company in companies],
        manifest_metadata={
            "base_records": len(bases),
            "counterfactual_families": counterfactual_families,
            "company_generator_version": COMPANY_GENERATOR_VERSION,
        },
    )
    return manifest


def _print_summary(manifest: Mapping[str, object]) -> None:
    print(f"Dataset: {manifest.get('dataset_version')} "
          f"(schema {manifest.get('schema_version')}, "
          f"generator {manifest.get('generator_version')}, seed {manifest.get('seed')})")
    print(f"Companies: {manifest.get('company_profiles')}")
    print(f"Base records: {manifest.get('base_records')}")
    print(f"Counterfactual families: {manifest.get('counterfactual_families')}")
    print(f"Total examples: {manifest.get('total_examples')}")
    print()
    print(f"{'Split':24s} | {'Examples':>10s}")
    print("-" * 40)
    splits = manifest.get("splits")
    if isinstance(splits, Mapping):
        for name in SPLIT_NAMES:
            entry = splits.get(name)
            if isinstance(entry, Mapping):
                print(f"{name:24s} | {entry.get('examples', 0):>10d}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="marktsignal-dataset",
        description="Generate a frozen, versioned economic dataset.",
    )
    parser.add_argument("--output", type=Path, default=Path("data/dataset_v1"))
    parser.add_argument("--dataset-version", default="1.0")
    parser.add_argument("--generator-version", default=GENERATOR_VERSION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--companies", type=int, default=10_000)
    parser.add_argument("--bases", type=int, default=20_000)
    parser.add_argument("--counterfactual-fraction", type=float, default=1.0)
    parser.add_argument("--concept-ood-fraction", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--iid-test-fraction", type=float, default=0.1)
    parser.add_argument("--entity-ood-fraction", type=float, default=0.1)
    args = parser.parse_args(argv)

    manifest = generate_dataset(
        output_dir=args.output,
        dataset_version=args.dataset_version,
        generator_version=args.generator_version,
        seed=args.seed,
        num_companies=args.companies,
        num_bases=args.bases,
        counterfactual_fraction=args.counterfactual_fraction,
        concept_ood_fraction=args.concept_ood_fraction,
        validation_fraction=args.validation_fraction,
        iid_test_fraction=args.iid_test_fraction,
        entity_ood_fraction=args.entity_ood_fraction,
    )
    _print_summary(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

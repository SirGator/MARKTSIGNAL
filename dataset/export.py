"""Deterministic JSONL export and loading for frozen dataset versions."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Mapping

from .labeling import require_valid_record
from .schema import DATASET_SCHEMA_VERSION, CompanyProfile, DatasetRecord
from .splits import SPLIT_NAMES, validate_split_integrity


SPLIT_FILES = {name: f"{name}.jsonl" for name in SPLIT_NAMES}
MANIFEST_FILE = "manifest.json"
COMPANIES_FILE = "companies.jsonl"


def freeze_dataset(
    output_dir: Path | str,
    splits: Mapping[str, Iterable[DatasetRecord]],
    *,
    dataset_version: str,
    generator_version: str,
    seed: int,
    companies: Iterable[CompanyProfile] = (),
    manifest_metadata: Mapping[str, int | float | str | bool] | None = None,
) -> dict[str, object]:
    """Write one immutable dataset directory and return its manifest.

    Existing targets are rejected.  Callers must choose a new versioned path
    after changing generation rules; an old dataset is never modified in place.
    """

    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"frozen dataset target already exists: {target}")
    if not isinstance(dataset_version, str) or not dataset_version.strip():
        raise ValueError("dataset_version must be a non-empty string")
    if not isinstance(generator_version, str) or not generator_version.strip():
        raise ValueError("generator_version must be a non-empty string")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    extra_metadata = dict(manifest_metadata or {})
    reserved_manifest_keys = {
        "dataset_version",
        "schema_version",
        "generator_version",
        "seed",
        "train_examples",
        "validation_examples",
        "test_examples",
        "total_examples",
        "company_profiles",
        "companies",
        "splits",
    }
    overlap = set(extra_metadata) & reserved_manifest_keys
    if overlap:
        raise ValueError(
            f"manifest_metadata uses reserved keys: {sorted(overlap)}"
        )
    for key, value in extra_metadata.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("manifest_metadata keys must be non-empty strings")
        if not isinstance(value, (str, int, float, bool)):
            raise TypeError("manifest_metadata values must be JSON scalars")

    materialized = {
        name: tuple(splits.get(name, ()))
        for name in SPLIT_NAMES
    }
    unknown = set(splits) - set(SPLIT_NAMES)
    if unknown:
        raise ValueError(f"unknown dataset splits: {sorted(unknown)}")
    validate_split_integrity(materialized)
    for records in materialized.values():
        for record in records:
            require_valid_record(record)

    company_values = tuple(companies)
    if any(not isinstance(company, CompanyProfile) for company in company_values):
        raise TypeError("companies must contain only CompanyProfile values")
    company_ids = [company.company_id for company in company_values]
    if len(company_ids) != len(set(company_ids)):
        raise ValueError("company IDs must be unique")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    try:
        split_entries: dict[str, dict[str, object]] = {}
        for split_name in SPLIT_NAMES:
            file_name = SPLIT_FILES[split_name]
            payload = _records_payload(materialized[split_name])
            (staging / file_name).write_bytes(payload)
            split_entries[split_name] = {
                "file": file_name,
                "examples": len(materialized[split_name]),
                "sha256": _sha256(payload),
            }

        companies_payload = _companies_payload(company_values)
        (staging / COMPANIES_FILE).write_bytes(companies_payload)
        manifest: dict[str, object] = {
            "dataset_version": dataset_version.strip(),
            "schema_version": DATASET_SCHEMA_VERSION,
            "generator_version": generator_version.strip(),
            "seed": seed,
            "train_examples": len(materialized["train"]),
            "validation_examples": len(materialized["validation"]),
            "test_examples": sum(
                len(materialized[name])
                for name in SPLIT_NAMES
                if name.startswith("test_")
            ),
            "total_examples": sum(len(records) for records in materialized.values()),
            "company_profiles": len(company_values),
            "companies": {
                "file": COMPANIES_FILE,
                "examples": len(company_values),
                "sha256": _sha256(companies_payload),
            },
            "splits": split_entries,
        }
        manifest.update(extra_metadata)
        manifest_payload = _canonical_json(manifest)
        (staging / MANIFEST_FILE).write_bytes(manifest_payload)
        os.replace(staging, target)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_jsonl(path: Path | str) -> tuple[DatasetRecord, ...]:
    """Load and strictly validate canonical records from one JSONL file."""

    source = Path(path)
    records: list[DatasetRecord] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL line at {source}:{line_number}")
            try:
                record = DatasetRecord.from_json(line)
                require_valid_record(record)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid dataset record at {source}:{line_number}: {exc}"
                ) from exc
            records.append(record)
    return tuple(records)


def load_frozen_dataset(
    dataset_dir: Path | str,
    *,
    verify_hashes: bool = True,
) -> tuple[dict[str, tuple[DatasetRecord, ...]], dict[str, object]]:
    """Load every split and verify manifest counts and optional hashes."""

    root = Path(dataset_dir)
    manifest_path = root / MANIFEST_FILE
    manifest = _load_manifest(manifest_path)
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError(
            "dataset schema version mismatch: "
            f"expected {DATASET_SCHEMA_VERSION!r}, got {manifest.get('schema_version')!r}"
        )
    raw_splits = manifest.get("splits")
    if not isinstance(raw_splits, dict):
        raise ValueError("manifest.splits must be an object")
    if set(raw_splits) != set(SPLIT_NAMES):
        raise ValueError("manifest must declare every canonical split exactly once")

    loaded: dict[str, tuple[DatasetRecord, ...]] = {}
    for split_name in SPLIT_NAMES:
        entry = raw_splits[split_name]
        if not isinstance(entry, dict):
            raise ValueError(f"manifest split {split_name!r} must be an object")
        if entry.get("file") != SPLIT_FILES[split_name]:
            raise ValueError(f"manifest file mismatch for split {split_name!r}")
        path = root / SPLIT_FILES[split_name]
        payload = path.read_bytes()
        if verify_hashes and _sha256(payload) != entry.get("sha256"):
            raise ValueError(f"checksum mismatch for {path}")
        records = load_jsonl(path)
        if entry.get("examples") != len(records):
            raise ValueError(f"record count mismatch for {path}")
        loaded[split_name] = records

    raw_companies = manifest.get("companies")
    if not isinstance(raw_companies, dict):
        raise ValueError("manifest.companies must be an object")
    if raw_companies.get("file") != COMPANIES_FILE:
        raise ValueError("manifest companies file mismatch")
    companies_path = root / COMPANIES_FILE
    companies_payload = companies_path.read_bytes()
    if verify_hashes and _sha256(companies_payload) != raw_companies.get("sha256"):
        raise ValueError(f"checksum mismatch for {companies_path}")
    companies = load_company_profiles(companies_path)
    if raw_companies.get("examples") != len(companies):
        raise ValueError(f"record count mismatch for {companies_path}")
    if manifest.get("company_profiles") != len(companies):
        raise ValueError("manifest company_profiles does not match companies file")

    validate_split_integrity(loaded)
    total = sum(len(records) for records in loaded.values())
    if manifest.get("total_examples") != total:
        raise ValueError("manifest total_examples does not match split contents")
    if manifest.get("train_examples") != len(loaded["train"]):
        raise ValueError("manifest train_examples does not match train split")
    if manifest.get("validation_examples") != len(loaded["validation"]):
        raise ValueError("manifest validation_examples does not match validation split")
    expected_test_examples = sum(
        len(loaded[name]) for name in SPLIT_NAMES if name.startswith("test_")
    )
    if manifest.get("test_examples") != expected_test_examples:
        raise ValueError("manifest test_examples does not match test splits")
    return loaded, manifest


def load_company_profiles(path: Path | str) -> tuple[CompanyProfile, ...]:
    """Load the reproducible company world saved next to a frozen dataset."""

    source = Path(path)
    companies: list[CompanyProfile] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL line at {source}:{line_number}")
            try:
                companies.append(CompanyProfile.from_json(line))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid company profile at {source}:{line_number}: {exc}"
                ) from exc
    return tuple(companies)


def _records_payload(records: Iterable[DatasetRecord]) -> bytes:
    ordered = sorted(records, key=lambda record: record.id)
    if not ordered:
        return b""
    return ("\n".join(record.to_json() for record in ordered) + "\n").encode("utf-8")


def _companies_payload(companies: Iterable[CompanyProfile]) -> bytes:
    ordered = sorted(companies, key=lambda company: company.company_id)
    if not ordered:
        return b""
    return ("\n".join(company.to_json() for company in ordered) + "\n").encode("utf-8")


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read dataset manifest {path}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValueError("dataset manifest must be a JSON object")
    return decoded


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "COMPANIES_FILE",
    "MANIFEST_FILE",
    "SPLIT_FILES",
    "freeze_dataset",
    "load_company_profiles",
    "load_frozen_dataset",
    "load_jsonl",
]

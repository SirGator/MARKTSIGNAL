"""Deterministic training-only views that resemble noisy runtime reports."""

from __future__ import annotations

from dataclasses import replace
import hashlib


RUNTIME_VIEW_VERSION = "runtime-view-v1"

_MECHANISM_DESCRIPTIONS: dict[str, str] = {
    "input_cost_change": "input costs",
    "output_price_change": "selling prices",
    "demand_change": "market demand",
    "supply_change": "available supply",
    "capacity_change": "production capacity",
    "supplier_disruption": "supplier availability",
    "customer_dependency_change": "customer dependency",
    "financing_cost_change": "financing costs",
    "currency_change": "currency exposure",
    "regulatory_cost_change": "regulatory costs",
    "tax_change": "the tax burden",
    "subsidy_change": "subsidy support",
    "competitive_pressure_change": "competitive pressure",
    "contract_change": "the contract position",
    "labor_cost_change": "labor costs",
}


def _selected(record_id: str, *, ratio: float, seed: int) -> bool:
    payload = f"{RUNTIME_VIEW_VERSION}\0{seed}\0{record_id}".encode("utf-8")
    sample = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return sample / 2**64 < ratio


def _runtime_summary(record) -> str:
    description = _MECHANISM_DESCRIPTIONS[record.event.mechanism]
    verb = "increased" if record.event.direction == "increase" else "decreased"
    return (
        f"According to the latest report, {description} linked to "
        f"{record.event.subject} {verb}; normalized shock magnitude "
        f"{record.event.magnitude:g}. The article also contained unrelated "
        "market commentary and quarterly background."
    )


def add_frozen_runtime_views(
    records: list,
    *,
    ratio: float = 0.2,
    seed: int = 42,
) -> list:
    """Append safe noisy-summary views after split assignment."""

    from dataset.schema import DatasetRecord

    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise TypeError("ratio must be a number")
    normalized_ratio = float(ratio)
    if not 0.0 <= normalized_ratio <= 1.0:
        raise ValueError("ratio must be between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if any(not isinstance(record, DatasetRecord) for record in records):
        raise TypeError("records must contain only DatasetRecord values")

    views: list[DatasetRecord] = []
    for record in records:
        if not _selected(record.id, ratio=normalized_ratio, seed=seed):
            continue
        views.append(
            replace(
                record,
                id=f"{record.id}::{RUNTIME_VIEW_VERSION}",
                event=replace(record.event, summary=_runtime_summary(record)),
            )
        )
    return [*records, *views]

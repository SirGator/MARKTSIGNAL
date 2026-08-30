"""Controlled one-variable variants for concept-focused training data."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .labeling import compute_economic_impact, compute_state_delta
from .schema import (
    CONTEXT_FACTOR_FIELDS,
    CounterfactualProvenance,
    DatasetRecord,
    Metadata,
)


DEFAULT_RATIO_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_MAGNITUDE_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)


def generate_counterfactuals(
    base: DatasetRecord,
    *,
    varied_field: str,
    values: Iterable[float],
    group_id: str | None = None,
) -> tuple[DatasetRecord, ...]:
    """Return variants that change exactly one causal input.

    ``varied_field`` may name an :class:`EconomicContext` ratio,
    ``event.magnitude``, or ``horizon_days``.  Labels are always recomputed
    through the canonical ``Event -> StateDelta -> Impact`` path.
    """

    if not isinstance(base, DatasetRecord):
        raise TypeError("base must be a DatasetRecord")
    if not isinstance(varied_field, str) or not varied_field.strip():
        raise ValueError("varied_field must be a non-empty string")
    field_name = varied_field.strip()
    allowed = {*CONTEXT_FACTOR_FIELDS, "event.magnitude", "horizon_days"}
    if field_name not in allowed:
        raise ValueError(f"unsupported counterfactual field: {varied_field!r}")

    normalized_values = tuple(values)
    if not normalized_values:
        raise ValueError("counterfactual values must not be empty")
    if len(set(normalized_values)) != len(normalized_values):
        raise ValueError("counterfactual values must be unique")

    family_id = group_id or f"cf:{base.id}:{field_name}"
    baseline = _input_value(base, field_name)
    variants: list[DatasetRecord] = []
    for index, value in enumerate(normalized_values):
        event = base.event
        context = base.context
        horizon_days = base.horizon_days
        if field_name == "event.magnitude":
            event = replace(event, magnitude=value)
        elif field_name == "horizon_days":
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("horizon_days counterfactual values must be integers")
            horizon_days = value
        else:
            context = replace(context, **{field_name: value})

        state_delta = compute_state_delta(event, context, horizon_days)
        economic_impact = compute_economic_impact(state_delta)
        counterfactual = CounterfactualProvenance(
            group_id=family_id,
            base_record_id=base.id,
            varied_field=field_name,
            variant_index=index,
            baseline_value=baseline,
            variant_value=value,
        )
        metadata = Metadata(
            source=base.metadata.source,
            generator_version=base.metadata.generator_version,
            seed=base.metadata.seed,
            counterfactual=counterfactual,
        )
        variants.append(
            DatasetRecord(
                id=f"{base.id}__cf_{_id_token(field_name)}_{index:02d}",
                event=event,
                target=base.target,
                context=context,
                horizon_days=horizon_days,
                state_delta=state_delta,
                economic_impact=economic_impact,
                confidence=base.confidence,
                metadata=metadata,
            )
        )
    return tuple(variants)


def standard_counterfactual_groups(
    base: DatasetRecord,
) -> dict[str, tuple[DatasetRecord, ...]]:
    """Build standard grids for all declared causal inputs of one record."""

    from .mechanisms import get_mechanism

    definition = get_mechanism(base.event.mechanism)
    fields = (*definition.required_context_fields, "event.magnitude")
    return {
        field: generate_counterfactuals(
            base,
            varied_field=field,
            values=(
                DEFAULT_MAGNITUDE_GRID
                if field == "event.magnitude"
                else DEFAULT_RATIO_GRID
            ),
        )
        for field in fields
    }


def _input_value(record: DatasetRecord, field_name: str) -> float | int:
    if field_name == "event.magnitude":
        return record.event.magnitude
    if field_name == "horizon_days":
        return record.horizon_days
    value = getattr(record.context, field_name)
    if value is None:
        raise ValueError(
            f"base record does not supply counterfactual field {field_name!r}"
        )
    return value


def _id_token(value: str) -> str:
    return value.replace(".", "_")


__all__ = [
    "DEFAULT_MAGNITUDE_GRID",
    "DEFAULT_RATIO_GRID",
    "generate_counterfactuals",
    "standard_counterfactual_groups",
]

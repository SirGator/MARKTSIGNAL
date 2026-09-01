"""Versioned contract for projecting frozen records into model input."""

from __future__ import annotations

import hashlib
import math
from typing import Mapping


DATASET_PROJECTION_VERSION = "dataset-record-to-context-v2"
EVENT_MAGNITUDE_SCALE = "ratio-0-1"
CONTEXT_VALUE_SCALE = "ratio-0-1"
_SUPPORTED_DATASET_SCHEMA_VERSION = "1.0"
_ALLOWED_MAGNITUDE_UNITS = frozenset((None, "fraction", "ratio"))


def validate_dataset_projection(record):
    """Fail closed before a frozen record enters the model serializer."""

    from dataset.schema import (
        DATASET_SCHEMA_VERSION,
        EVENT_DIRECTIONS,
        MECHANISM_SET,
        DatasetRecord,
    )

    if not isinstance(record, DatasetRecord):
        raise TypeError("record must be a DatasetRecord")
    if DATASET_SCHEMA_VERSION != _SUPPORTED_DATASET_SCHEMA_VERSION:
        raise ValueError(
            "unsupported dataset schema for input projection: "
            f"{DATASET_SCHEMA_VERSION!r}"
        )
    if record.event.mechanism not in MECHANISM_SET:
        raise ValueError("record event mechanism is outside the input contract")
    if record.event.direction not in EVENT_DIRECTIONS:
        raise ValueError("record event direction is outside the input contract")
    magnitude = float(record.event.magnitude)
    if not math.isfinite(magnitude) or not 0.0 <= magnitude <= 1.0:
        raise ValueError("record event magnitude must use ratio-0-1 scale")
    if record.event.unit not in _ALLOWED_MAGNITUDE_UNITS:
        raise ValueError(
            "record event unit is ambiguous for ratio-0-1 magnitude; "
            "expected None, fraction, or ratio"
        )
    return record


def _probe_record(
    *,
    mechanism: str,
    subject_class: str,
    context_fields: dict[str, float],
    unit: str | None,
    horizon_days: int = 30,
):
    """Build one minimal valid probe record for a mechanism."""

    from dataset.schema import (
        DatasetRecord,
        EconomicContext,
        Event,
        Metadata,
        StateDelta,
        Target,
    )

    return DatasetRecord(
        id=f"projection-probe:{mechanism}",
        event=Event(
            mechanism=mechanism,
            subject_class=subject_class,
            subject="probe-input",
            direction="increase",
            magnitude=0.125,
            unit=unit,
        ),
        target=Target(entity_id="company:projection-probe"),
        context=EconomicContext(**context_fields),
        horizon_days=horizon_days,
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
        metadata=Metadata(
            source="projection-contract",
            generator_version="probe-v1",
            seed=0,
        ),
    )


def _projection_probe_texts() -> tuple[str, ...]:
    """Render a stable behavioral probe suite through the real bridge.

    One probe per declared mechanism keeps any change to a single mechanism's
    serialization visible.  A dedicated full-context probe touches every
    CONTEXT_FACTOR_FIELDS value at least once, and both accepted magnitude
    units (None and ratio) are exercised, each under ``full`` and ``none``
    summary modes.
    """

    from dataset.mechanisms import get_mechanism, list_mechanisms
    from dataset.schema import CONTEXT_FACTOR_FIELDS
    from .bridge import serialize_record

    probes = []

    for mechanism in list_mechanisms():
        definition = get_mechanism(mechanism)
        context_fields = {
            name: 0.25 for name in definition.required_context_fields
        }
        probes.append(
            _probe_record(
                mechanism=mechanism,
                subject_class=definition.subject_classes[0],
                context_fields=context_fields,
                unit=None,
            )
        )

    probes.append(
        _probe_record(
            mechanism="input_cost_change",
            subject_class="commodity",
            context_fields={name: 0.5 for name in CONTEXT_FACTOR_FIELDS},
            unit="ratio",
        )
    )
    probes.append(
        _probe_record(
            mechanism="financing_cost_change",
            subject_class="financing",
            context_fields={name: 0.5 for name in CONTEXT_FACTOR_FIELDS},
            unit="fraction",
        )
    )

    texts: list[str] = []
    for probe in probes:
        texts.append(serialize_record(probe, summary_mode="full"))
        texts.append(serialize_record(probe, summary_mode="none"))
    return tuple(texts)


def dataset_projection_fingerprint() -> str:
    """Return a hash of behavior, not of duplicated source files."""

    digest = hashlib.sha256()
    digest.update(DATASET_PROJECTION_VERSION.encode("utf-8"))
    for text in _projection_probe_texts():
        payload = text.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def dataset_projection_metadata() -> dict[str, str]:
    """Return checkpoint fields identifying the frozen-record projection."""

    return {
        "dataset_projection_version": DATASET_PROJECTION_VERSION,
        "dataset_projection_fingerprint": dataset_projection_fingerprint(),
        "event_magnitude_scale": EVENT_MAGNITUDE_SCALE,
        "context_value_scale": CONTEXT_VALUE_SCALE,
    }


def validate_dataset_projection_metadata(
    config: Mapping[str, object],
    *,
    legacy_projection: bool = False,
) -> None:
    """Reject a frozen-dataset checkpoint produced by a different projection.

    Legacy artifacts predate the projection fingerprint entirely (they carry
    ``dataset_schema_version`` without ``dataset_projection_*`` fields).  They
    fail closed unless a caller explicitly declares the legacy projection
    with ``legacy_projection=True``, mirroring the ``legacy_summary_mode``
    contract for pre-V2 summaries.
    """

    if not isinstance(config, Mapping):
        raise TypeError("checkpoint config must be a mapping")
    expected = dataset_projection_metadata()
    missing = [key for key in expected if key not in config]
    if missing and legacy_projection:
        if any(key in config for key in expected):
            raise ValueError(
                "partially versioned checkpoint config cannot be declared "
                "legacy: " + ", ".join(sorted(missing))
            )
        return
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"checkpoint config.{key} does not match the training input contract"
            )

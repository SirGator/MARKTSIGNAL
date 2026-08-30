"""Canonical, dependency-free contracts for frozen economic datasets.

The values in this module are deliberately independent from the training
runtime.  Dataset generation, verification, export, and training loaders can
therefore share one immutable JSON contract without importing PyTorch.

All magnitudes and structural factors are normalized ratios.  Event magnitude
uses ``[0, 1]`` with direction carried separately; state deltas and economic
impact use ``[-1, 1]``.  ``None`` on an :class:`EconomicContext` field means
"unknown/not supplied", while ``0.0`` is an observed absence of exposure.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from typing import Self, TypeAlias


DATASET_SCHEMA_VERSION = "1.0"

MECHANISMS: tuple[str, ...] = (
    "input_cost_change",
    "output_price_change",
    "demand_change",
    "supply_change",
    "capacity_change",
    "supplier_disruption",
    "customer_dependency_change",
    "financing_cost_change",
    "currency_change",
    "regulatory_cost_change",
    "tax_change",
    "subsidy_change",
    "competitive_pressure_change",
    "contract_change",
    "labor_cost_change",
)
MECHANISM_SET = frozenset(MECHANISMS)
EVENT_DIRECTIONS = frozenset(("increase", "decrease"))

CONTEXT_FACTOR_FIELDS: tuple[str, ...] = (
    "input_dependency",
    "output_dependency",
    "demand_exposure",
    "hedging",
    "pricing_power",
    "substitutability",
    "inventory_buffer",
    "supplier_diversification",
    "customer_concentration",
    "debt_ratio",
    "cash_buffer",
    "currency_revenue_exposure",
    "currency_cost_exposure",
    "regulatory_exposure",
    "tax_exposure",
    "subsidy_exposure",
    "competitive_pressure",
    "contract_revenue_share",
    "labor_intensity",
    "capacity_utilization",
    "fixed_cost_share",
)

STATE_DELTA_FIELDS: tuple[str, ...] = (
    "revenue_expectation",
    "demand",
    "operating_costs",
    "margin_expectation",
    "supply_risk",
    "competitive_position",
)

JsonScalar: TypeAlias = str | int | float | bool


def _required_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_text(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _required_text(name, value)


def _number(
    name: str,
    value: object,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if not minimum <= normalized <= maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return normalized


def _ratio(name: str, value: object) -> float:
    return _number(name, value, minimum=0.0, maximum=1.0)


def _optional_ratio(name: str, value: object) -> float | None:
    if value is None:
        return None
    return _ratio(name, value)


def _signed_ratio(name: str, value: object) -> float:
    return _number(name, value, minimum=-1.0, maximum=1.0)


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _optional_scalar(name: str, value: object) -> JsonScalar | None:
    if value is None:
        return None
    if isinstance(value, str):
        return _required_text(name, value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value
    raise TypeError(f"{name} must be a JSON scalar or None")


def _as_mapping(name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def _check_keys(
    name: str,
    value: Mapping[str, object],
    *,
    required: tuple[str, ...],
    optional: tuple[str, ...] = (),
) -> None:
    keys = frozenset(value)
    missing = frozenset(required) - keys
    unknown = keys - frozenset((*required, *optional))
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(sorted(unknown))}")


def _canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _decoded_json(payload: str) -> Mapping[str, object]:
    if not isinstance(payload, str):
        raise TypeError("JSON payload must be a string")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number is not permitted: {value}")

    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    decoded = json.loads(
        payload,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )
    return _as_mapping("JSON document", decoded)


class _JsonMixin:
    """Provide deterministic JSON methods to classes with dict methods."""

    def to_json(self) -> str:
        """Return a compact, key-sorted JSON representation."""

        serializer = getattr(self, "to_dict")
        return _canonical_json(serializer())

    @classmethod
    def from_json(cls, payload: str) -> Self:
        """Parse strict JSON and validate it through ``from_dict``."""

        parser = getattr(cls, "from_dict")
        return parser(_decoded_json(payload))


@dataclass(frozen=True, slots=True)
class Event(_JsonMixin):
    """One normalized economic event.

    ``magnitude`` is the absolute normalized shock size.  ``direction`` carries
    its orientation, avoiding signed magnitudes with contradictory directions.
    ``summary`` is optional because structured-only datasets do not require a
    free-language channel.
    """

    mechanism: str
    subject_class: str
    subject: str
    direction: str
    magnitude: float
    summary: str | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        mechanism = _required_text("event.mechanism", self.mechanism).casefold()
        if mechanism not in MECHANISM_SET:
            raise ValueError(
                "event.mechanism must be one of " + ", ".join(MECHANISMS)
            )
        direction = _required_text("event.direction", self.direction).casefold()
        if direction not in EVENT_DIRECTIONS:
            raise ValueError("event.direction must be 'increase' or 'decrease'")
        object.__setattr__(self, "mechanism", mechanism)
        object.__setattr__(
            self,
            "subject_class",
            _required_text("event.subject_class", self.subject_class).casefold(),
        )
        object.__setattr__(self, "subject", _required_text("event.subject", self.subject))
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "magnitude", _ratio("event.magnitude", self.magnitude))
        object.__setattr__(self, "summary", _optional_text("event.summary", self.summary))
        object.__setattr__(self, "unit", _optional_text("event.unit", self.unit))

    def to_dict(self) -> dict[str, object]:
        """Return the canonical event mapping, omitting absent optionals."""

        result: dict[str, object] = {
            "mechanism": self.mechanism,
            "subject_class": self.subject_class,
            "subject": self.subject,
            "direction": self.direction,
            "magnitude": self.magnitude,
        }
        if self.summary is not None:
            result["summary"] = self.summary
        if self.unit is not None:
            result["unit"] = self.unit
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Event:
        """Build and validate an event from its canonical mapping."""

        data = _as_mapping("event", value)
        _check_keys(
            "event",
            data,
            required=("mechanism", "subject_class", "subject", "direction", "magnitude"),
            optional=("summary", "unit"),
        )
        return cls(
            mechanism=data["mechanism"],  # type: ignore[arg-type]
            subject_class=data["subject_class"],  # type: ignore[arg-type]
            subject=data["subject"],  # type: ignore[arg-type]
            direction=data["direction"],  # type: ignore[arg-type]
            magnitude=data["magnitude"],  # type: ignore[arg-type]
            summary=data.get("summary"),  # type: ignore[arg-type]
            unit=data.get("unit"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class Target(_JsonMixin):
    """Identity of the entity whose economic impact is labeled."""

    entity_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "entity_id",
            _required_text("target.entity_id", self.entity_id),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical target mapping."""

        return {"entity_id": self.entity_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Target:
        """Build and validate a target from its canonical mapping."""

        data = _as_mapping("target", value)
        _check_keys("target", data, required=("entity_id",))
        return cls(entity_id=data["entity_id"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class EconomicContext(_JsonMixin):
    """Normalized structural factors used by the mechanism labelers.

    The explicit names cover the fifteen mechanism families without deriving
    behavior from an industry label.  Currency cost and revenue exposures are
    separate so an exchange-rate move has an unambiguous transmission side.
    All fields are optional ratios in ``[0, 1]``; missing means unknown.
    """

    input_dependency: float | None = None
    output_dependency: float | None = None
    demand_exposure: float | None = None
    hedging: float | None = None
    pricing_power: float | None = None
    substitutability: float | None = None
    inventory_buffer: float | None = None
    supplier_diversification: float | None = None
    customer_concentration: float | None = None
    debt_ratio: float | None = None
    cash_buffer: float | None = None
    currency_revenue_exposure: float | None = None
    currency_cost_exposure: float | None = None
    regulatory_exposure: float | None = None
    tax_exposure: float | None = None
    subsidy_exposure: float | None = None
    competitive_pressure: float | None = None
    contract_revenue_share: float | None = None
    labor_intensity: float | None = None
    capacity_utilization: float | None = None
    fixed_cost_share: float | None = None

    def __post_init__(self) -> None:
        for name in CONTEXT_FACTOR_FIELDS:
            object.__setattr__(
                self,
                name,
                _optional_ratio(f"context.{name}", getattr(self, name)),
            )

    def to_dict(self) -> dict[str, object]:
        """Return supplied factors only, preserving unknown versus zero."""

        return {
            name: value
            for name in CONTEXT_FACTOR_FIELDS
            if (value := getattr(self, name)) is not None
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> EconomicContext:
        """Build and validate context from a sparse factor mapping."""

        data = _as_mapping("context", value)
        _check_keys("context", data, required=(), optional=CONTEXT_FACTOR_FIELDS)
        return cls(**{name: data[name] for name in CONTEXT_FACTOR_FIELDS if name in data})  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class StateDelta(_JsonMixin):
    """Normalized change in the six canonical economic state dimensions.

    Each value lies in ``[-1, 1]``.  Positive values mean an increase in the
    named dimension; therefore a positive operating-cost or supply-risk delta
    can still contribute negatively to the final economic impact.
    """

    revenue_expectation: float
    demand: float
    operating_costs: float
    margin_expectation: float
    supply_risk: float
    competitive_position: float

    def __post_init__(self) -> None:
        for name in STATE_DELTA_FIELDS:
            object.__setattr__(
                self,
                name,
                _signed_ratio(f"state_delta.{name}", getattr(self, name)),
            )

    def to_dict(self) -> dict[str, object]:
        """Return all six state dimensions in canonical order."""

        return {name: getattr(self, name) for name in STATE_DELTA_FIELDS}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> StateDelta:
        """Build and validate a complete state delta mapping."""

        data = _as_mapping("state_delta", value)
        _check_keys("state_delta", data, required=STATE_DELTA_FIELDS)
        return cls(**{name: data[name] for name in STATE_DELTA_FIELDS})  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class CounterfactualProvenance(_JsonMixin):
    """Trace one controlled variant back to its counterfactual family."""

    group_id: str
    base_record_id: str
    varied_field: str
    variant_index: int | None = None
    baseline_value: JsonScalar | None = None
    variant_value: JsonScalar | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_id", _required_text("counterfactual.group_id", self.group_id))
        object.__setattr__(
            self,
            "base_record_id",
            _required_text("counterfactual.base_record_id", self.base_record_id),
        )
        object.__setattr__(
            self,
            "varied_field",
            _required_text("counterfactual.varied_field", self.varied_field),
        )
        if self.variant_index is not None:
            object.__setattr__(
                self,
                "variant_index",
                _integer("counterfactual.variant_index", self.variant_index),
            )
        object.__setattr__(
            self,
            "baseline_value",
            _optional_scalar("counterfactual.baseline_value", self.baseline_value),
        )
        object.__setattr__(
            self,
            "variant_value",
            _optional_scalar("counterfactual.variant_value", self.variant_value),
        )

    def to_dict(self) -> dict[str, object]:
        """Return provenance, omitting unavailable optional details."""

        result: dict[str, object] = {
            "group_id": self.group_id,
            "base_record_id": self.base_record_id,
            "varied_field": self.varied_field,
        }
        if self.variant_index is not None:
            result["variant_index"] = self.variant_index
        if self.baseline_value is not None:
            result["baseline_value"] = self.baseline_value
        if self.variant_value is not None:
            result["variant_value"] = self.variant_value
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CounterfactualProvenance:
        """Build and validate counterfactual provenance."""

        data = _as_mapping("counterfactual", value)
        _check_keys(
            "counterfactual",
            data,
            required=("group_id", "base_record_id", "varied_field"),
            optional=("variant_index", "baseline_value", "variant_value"),
        )
        return cls(
            group_id=data["group_id"],  # type: ignore[arg-type]
            base_record_id=data["base_record_id"],  # type: ignore[arg-type]
            varied_field=data["varied_field"],  # type: ignore[arg-type]
            variant_index=data.get("variant_index"),  # type: ignore[arg-type]
            baseline_value=data.get("baseline_value"),  # type: ignore[arg-type]
            variant_value=data.get("variant_value"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class Metadata(_JsonMixin):
    """Reproducibility metadata retained with every dataset record."""

    source: str
    generator_version: str
    seed: int
    counterfactual: CounterfactualProvenance | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _required_text("metadata.source", self.source))
        object.__setattr__(
            self,
            "generator_version",
            _required_text("metadata.generator_version", self.generator_version),
        )
        object.__setattr__(self, "seed", _integer("metadata.seed", self.seed))
        if self.counterfactual is not None and not isinstance(
            self.counterfactual,
            CounterfactualProvenance,
        ):
            raise TypeError(
                "metadata.counterfactual must be CounterfactualProvenance or None"
            )

    @property
    def counterfactual_provenance(self) -> CounterfactualProvenance | None:
        """Descriptive alias for the optional counterfactual metadata."""

        return self.counterfactual

    def to_dict(self) -> dict[str, object]:
        """Return canonical metadata, omitting absent provenance."""

        result: dict[str, object] = {
            "source": self.source,
            "generator_version": self.generator_version,
            "seed": self.seed,
        }
        if self.counterfactual is not None:
            result["counterfactual"] = self.counterfactual.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Metadata:
        """Build and validate record metadata."""

        data = _as_mapping("metadata", value)
        _check_keys(
            "metadata",
            data,
            required=("source", "generator_version", "seed"),
            optional=("counterfactual",),
        )
        raw_counterfactual = data.get("counterfactual")
        counterfactual = (
            None
            if raw_counterfactual is None
            else CounterfactualProvenance.from_dict(
                _as_mapping("metadata.counterfactual", raw_counterfactual)
            )
        )
        return cls(
            source=data["source"],  # type: ignore[arg-type]
            generator_version=data["generator_version"],  # type: ignore[arg-type]
            seed=data["seed"],  # type: ignore[arg-type]
            counterfactual=counterfactual,
        )


@dataclass(frozen=True, slots=True)
class CostStructure(_JsonMixin):
    """Normalized shares of the principal company cost categories."""

    energy: float
    materials: float
    labor: float
    financing: float

    def __post_init__(self) -> None:
        for name in ("energy", "materials", "labor", "financing"):
            object.__setattr__(
                self,
                name,
                _ratio(f"cost_structure.{name}", getattr(self, name)),
            )
        if math.fsum((self.energy, self.materials, self.labor, self.financing)) > (
            1.0 + 1e-12
        ):
            raise ValueError("cost structure shares must sum to at most 1")

    def to_dict(self) -> dict[str, object]:
        """Return normalized cost shares."""

        return {
            "energy": self.energy,
            "materials": self.materials,
            "labor": self.labor,
            "financing": self.financing,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CostStructure:
        """Build and validate a company cost structure."""

        data = _as_mapping("cost_structure", value)
        names = ("energy", "materials", "labor", "financing")
        _check_keys("cost_structure", data, required=names)
        return cls(**{name: data[name] for name in names})  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class FinancialProfile(_JsonMixin):
    """Normalized balance-sheet factors for a synthetic company."""

    debt_ratio: float
    cash_buffer: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "debt_ratio", _ratio("financial.debt_ratio", self.debt_ratio))
        object.__setattr__(self, "cash_buffer", _ratio("financial.cash_buffer", self.cash_buffer))

    def to_dict(self) -> dict[str, object]:
        """Return normalized financial factors."""

        return {"debt_ratio": self.debt_ratio, "cash_buffer": self.cash_buffer}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FinancialProfile:
        """Build and validate financial factors."""

        data = _as_mapping("financial", value)
        _check_keys("financial", data, required=("debt_ratio", "cash_buffer"))
        return cls(
            debt_ratio=data["debt_ratio"],  # type: ignore[arg-type]
            cash_buffer=data["cash_buffer"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ResilienceProfile(_JsonMixin):
    """Normalized operational buffers for a synthetic company."""

    inventory_buffer: float
    supplier_diversification: float
    substitutability: float

    def __post_init__(self) -> None:
        for name in (
            "inventory_buffer",
            "supplier_diversification",
            "substitutability",
        ):
            object.__setattr__(
                self,
                name,
                _ratio(f"resilience.{name}", getattr(self, name)),
            )

    def to_dict(self) -> dict[str, object]:
        """Return normalized resilience factors."""

        return {
            "inventory_buffer": self.inventory_buffer,
            "supplier_diversification": self.supplier_diversification,
            "substitutability": self.substitutability,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ResilienceProfile:
        """Build and validate operational resilience factors."""

        data = _as_mapping("resilience", value)
        names = (
            "inventory_buffer",
            "supplier_diversification",
            "substitutability",
        )
        _check_keys("resilience", data, required=names)
        return cls(**{name: data[name] for name in names})  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class CompanyProfile(_JsonMixin):
    """Persistent synthetic company state from which contexts can be sampled."""

    company_id: str
    cost_structure: CostStructure
    pricing_power: float
    financial: FinancialProfile
    resilience: ResilienceProfile

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "company_id",
            _required_text("company.company_id", self.company_id),
        )
        if not isinstance(self.cost_structure, CostStructure):
            raise TypeError("company.cost_structure must be a CostStructure")
        object.__setattr__(
            self,
            "pricing_power",
            _ratio("company.pricing_power", self.pricing_power),
        )
        if not isinstance(self.financial, FinancialProfile):
            raise TypeError("company.financial must be a FinancialProfile")
        if not isinstance(self.resilience, ResilienceProfile):
            raise TypeError("company.resilience must be a ResilienceProfile")

    def to_dict(self) -> dict[str, object]:
        """Return the canonical nested company-profile mapping."""

        return {
            "company_id": self.company_id,
            "cost_structure": self.cost_structure.to_dict(),
            "pricing_power": self.pricing_power,
            "financial": self.financial.to_dict(),
            "resilience": self.resilience.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CompanyProfile:
        """Build and validate a nested company profile."""

        data = _as_mapping("company", value)
        _check_keys(
            "company",
            data,
            required=(
                "company_id",
                "cost_structure",
                "pricing_power",
                "financial",
                "resilience",
            ),
        )
        return cls(
            company_id=data["company_id"],  # type: ignore[arg-type]
            cost_structure=CostStructure.from_dict(
                _as_mapping("company.cost_structure", data["cost_structure"])
            ),
            pricing_power=data["pricing_power"],  # type: ignore[arg-type]
            financial=FinancialProfile.from_dict(
                _as_mapping("company.financial", data["financial"])
            ),
            resilience=ResilienceProfile.from_dict(
                _as_mapping("company.resilience", data["resilience"])
            ),
        )


@dataclass(frozen=True, slots=True)
class DatasetRecord(_JsonMixin):
    """Complete immutable ground-truth record for score-model training."""

    id: str
    event: Event
    target: Target
    context: EconomicContext
    horizon_days: int
    state_delta: StateDelta
    economic_impact: float
    confidence: float
    metadata: Metadata

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required_text("record.id", self.id))
        for name, expected_type in (
            ("event", Event),
            ("target", Target),
            ("context", EconomicContext),
            ("state_delta", StateDelta),
            ("metadata", Metadata),
        ):
            if not isinstance(getattr(self, name), expected_type):
                raise TypeError(f"record.{name} must be a {expected_type.__name__}")
        object.__setattr__(
            self,
            "horizon_days",
            _integer("record.horizon_days", self.horizon_days, minimum=1),
        )
        object.__setattr__(
            self,
            "economic_impact",
            _signed_ratio("record.economic_impact", self.economic_impact),
        )
        object.__setattr__(
            self,
            "confidence",
            _ratio("record.confidence", self.confidence),
        )

    @property
    def score(self) -> float:
        """Compatibility alias expected by the current score-training pipeline."""

        return self.economic_impact

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON-ready record mapping."""

        return {
            "id": self.id,
            "event": self.event.to_dict(),
            "target": self.target.to_dict(),
            "context": self.context.to_dict(),
            "horizon_days": self.horizon_days,
            "state_delta": self.state_delta.to_dict(),
            "economic_impact": self.economic_impact,
            "confidence": self.confidence,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DatasetRecord:
        """Build and validate a complete ground-truth record mapping."""

        data = _as_mapping("record", value)
        _check_keys(
            "record",
            data,
            required=(
                "id",
                "event",
                "target",
                "context",
                "horizon_days",
                "state_delta",
                "economic_impact",
                "confidence",
                "metadata",
            ),
        )
        return cls(
            id=data["id"],  # type: ignore[arg-type]
            event=Event.from_dict(_as_mapping("record.event", data["event"])),
            target=Target.from_dict(_as_mapping("record.target", data["target"])),
            context=EconomicContext.from_dict(
                _as_mapping("record.context", data["context"])
            ),
            horizon_days=data["horizon_days"],  # type: ignore[arg-type]
            state_delta=StateDelta.from_dict(
                _as_mapping("record.state_delta", data["state_delta"])
            ),
            economic_impact=data["economic_impact"],  # type: ignore[arg-type]
            confidence=data["confidence"],  # type: ignore[arg-type]
            metadata=Metadata.from_dict(
                _as_mapping("record.metadata", data["metadata"])
            ),
        )


__all__ = [
    "CONTEXT_FACTOR_FIELDS",
    "DATASET_SCHEMA_VERSION",
    "EVENT_DIRECTIONS",
    "MECHANISMS",
    "MECHANISM_SET",
    "STATE_DELTA_FIELDS",
    "CompanyProfile",
    "CostStructure",
    "CounterfactualProvenance",
    "DatasetRecord",
    "EconomicContext",
    "Event",
    "FinancialProfile",
    "Metadata",
    "ResilienceProfile",
    "StateDelta",
    "Target",
]

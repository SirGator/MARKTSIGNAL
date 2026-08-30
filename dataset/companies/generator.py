"""Persistent synthetic company states for economic dataset generation.

Companies are represented only by structural ratios.  No industry, company
name, or other semantic shortcut is generated, so downstream models must learn
economic transmission from dependencies, buffers, and exposures.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import random

from dataset.mechanisms import get_mechanism
from dataset.schema import (
    CONTEXT_FACTOR_FIELDS,
    CompanyProfile,
    CostStructure,
    EconomicContext,
    Event,
    FinancialProfile,
    ResilienceProfile,
)

from .distributions import beta_ratio, correlated_ratio, truncated_ratio


COMPANY_GENERATOR_VERSION = "companies-v1"


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _mapping(name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def _derived_seed(seed: int, index: int) -> int:
    material = f"{COMPANY_GENERATOR_VERSION}:{seed}:{index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:16], "big")


def _same_ratio(name: str, left: float | None, right: float) -> None:
    if left is None or not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"company context {name} conflicts with company profile")


@dataclass(frozen=True, slots=True)
class SyntheticCompany:
    """One reusable synthetic company profile and its complete structural state.

    The context is sampled once and retained.  Event generation can therefore
    reuse the same company across many mechanisms and counterfactuals without
    silently changing unrelated attributes.
    """

    profile: CompanyProfile
    context: EconomicContext

    def __post_init__(self) -> None:
        if not isinstance(self.profile, CompanyProfile):
            raise TypeError("profile must be a CompanyProfile")
        if not isinstance(self.context, EconomicContext):
            raise TypeError("context must be an EconomicContext")

        missing = tuple(
            name
            for name in CONTEXT_FACTOR_FIELDS
            if getattr(self.context, name) is None
        )
        if missing:
            raise ValueError(
                "synthetic company context must contain every structural factor; "
                f"missing: {', '.join(missing)}"
            )

        costs = self.profile.cost_structure
        if math.fsum((costs.energy, costs.materials, costs.labor, costs.financing)) > (
            1.0 + 1e-12
        ):
            raise ValueError("company cost shares must sum to at most 1")

        _same_ratio("pricing_power", self.context.pricing_power, self.profile.pricing_power)
        _same_ratio("debt_ratio", self.context.debt_ratio, self.profile.financial.debt_ratio)
        _same_ratio("cash_buffer", self.context.cash_buffer, self.profile.financial.cash_buffer)
        _same_ratio(
            "inventory_buffer",
            self.context.inventory_buffer,
            self.profile.resilience.inventory_buffer,
        )
        _same_ratio(
            "supplier_diversification",
            self.context.supplier_diversification,
            self.profile.resilience.supplier_diversification,
        )
        _same_ratio(
            "substitutability",
            self.context.substitutability,
            self.profile.resilience.substitutability,
        )
        _same_ratio("labor_intensity", self.context.labor_intensity, costs.labor)

    @property
    def company_id(self) -> str:
        """Return the stable synthetic entity ID."""

        return self.profile.company_id

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready persistent company state."""

        return {
            "profile": self.profile.to_dict(),
            "context": self.context.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SyntheticCompany:
        """Restore and validate a persistent company state."""

        data = _mapping("synthetic_company", value)
        missing = {"profile", "context"} - set(data)
        unknown = set(data) - {"profile", "context"}
        if missing:
            raise ValueError(
                "synthetic_company is missing fields: "
                + ", ".join(sorted(missing))
            )
        if unknown:
            raise ValueError(
                "synthetic_company has unknown fields: "
                + ", ".join(sorted(unknown))
            )
        return cls(
            profile=CompanyProfile.from_dict(
                _mapping("synthetic_company.profile", data["profile"])
            ),
            context=EconomicContext.from_dict(
                _mapping("synthetic_company.context", data["context"])
            ),
        )

    def to_json(self) -> str:
        """Return compact deterministic JSON for persistence."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, payload: str) -> SyntheticCompany:
        """Restore a company from JSON and validate all nested ratios."""

        if not isinstance(payload, str):
            raise TypeError("JSON payload must be a string")
        return cls.from_dict(_mapping("JSON document", json.loads(payload)))

    def context_for_event(self, event_or_mechanism: Event | str) -> EconomicContext:
        """Return this company's stable factors required for one mechanism."""

        return context_for_event(self, event_or_mechanism)


def _cost_structure(
    rng: random.Random,
    *,
    capital_intensity: float,
    labor_orientation: float,
) -> CostStructure:
    """Draw four correlated cost shares with an explicit unmodeled remainder."""

    modeled_total = beta_ratio(rng, 4.5, 2.2, low=0.45, high=0.93)
    raw_weights = (
        rng.gammavariate(0.8 + 2.2 * capital_intensity, 1.0),
        rng.gammavariate(1.2 + 2.6 * capital_intensity, 1.0),
        rng.gammavariate(1.0 + 3.2 * labor_orientation, 1.0),
        rng.gammavariate(0.6 + 1.6 * capital_intensity, 1.0),
    )
    denominator = math.fsum(raw_weights)
    shares = tuple(modeled_total * weight / denominator for weight in raw_weights)
    return CostStructure(
        energy=shares[0],
        materials=shares[1],
        labor=shares[2],
        financing=shares[3],
    )


def _sample_company(rng: random.Random, company_id: str) -> SyntheticCompany:
    """Sample one company from non-uniform, structurally correlated factors."""

    capital_intensity = beta_ratio(rng, 2.2, 2.8, low=0.04, high=0.94)
    labor_orientation = beta_ratio(rng, 2.0, 2.4, low=0.08, high=0.92)
    resilience_anchor = beta_ratio(rng, 2.8, 2.3, low=0.06, high=0.94)
    global_exposure = beta_ratio(rng, 1.7, 3.1, low=0.01, high=0.92)
    competitive_anchor = beta_ratio(rng, 2.7, 2.0, low=0.06, high=0.96)

    costs = _cost_structure(
        rng,
        capital_intensity=capital_intensity,
        labor_orientation=labor_orientation,
    )

    pricing_power = correlated_ratio(
        rng,
        competitive_anchor,
        -0.68,
        mean=0.36,
        stddev=0.17,
    )
    debt_ratio = correlated_ratio(
        rng,
        capital_intensity,
        0.64,
        mean=0.42,
        stddev=0.18,
    )
    cash_buffer = correlated_ratio(
        rng,
        debt_ratio,
        -0.58,
        mean=0.24,
        stddev=0.13,
    )
    inventory_buffer = correlated_ratio(
        rng,
        resilience_anchor,
        0.68,
        mean=0.27,
        stddev=0.15,
    )
    supplier_diversification = correlated_ratio(
        rng,
        resilience_anchor,
        0.73,
        mean=0.42,
        stddev=0.18,
    )
    substitutability = correlated_ratio(
        rng,
        resilience_anchor,
        0.61,
        mean=0.39,
        stddev=0.19,
    )

    financial = FinancialProfile(debt_ratio=debt_ratio, cash_buffer=cash_buffer)
    resilience = ResilienceProfile(
        inventory_buffer=inventory_buffer,
        supplier_diversification=supplier_diversification,
        substitutability=substitutability,
    )
    profile = CompanyProfile(
        company_id=company_id,
        cost_structure=costs,
        pricing_power=pricing_power,
        financial=financial,
        resilience=resilience,
    )

    modeled_cost_total = math.fsum(
        (costs.energy, costs.materials, costs.labor, costs.financing)
    )
    input_anchor = min(1.0, (costs.energy + costs.materials) / modeled_cost_total)
    input_dependency = correlated_ratio(
        rng,
        input_anchor,
        0.76,
        mean=0.38,
        stddev=0.17,
    )
    output_dependency = beta_ratio(rng, 2.6, 1.8, low=0.10, high=0.98)
    demand_exposure = correlated_ratio(
        rng,
        output_dependency,
        0.79,
        mean=0.54,
        stddev=0.17,
    )

    hedging_capacity = math.fsum((capital_intensity, cash_buffer, global_exposure)) / 3.0
    hedging = correlated_ratio(
        rng,
        hedging_capacity,
        0.54,
        mean=0.25,
        stddev=0.16,
    )
    customer_concentration = beta_ratio(rng, 1.45, 4.0, low=0.01, high=0.94)
    currency_revenue_exposure = correlated_ratio(
        rng,
        global_exposure,
        0.76,
        mean=0.20,
        stddev=0.15,
    )
    currency_cost_anchor = math.fsum((global_exposure, input_dependency)) / 2.0
    currency_cost_exposure = correlated_ratio(
        rng,
        currency_cost_anchor,
        0.69,
        mean=0.22,
        stddev=0.15,
    )
    regulatory_anchor = math.fsum((capital_intensity, modeled_cost_total)) / 2.0
    regulatory_exposure = correlated_ratio(
        rng,
        regulatory_anchor,
        0.46,
        mean=0.38,
        stddev=0.18,
    )
    tax_exposure = beta_ratio(rng, 3.0, 1.8, low=0.12, high=0.98)
    subsidy_exposure = beta_ratio(rng, 1.05, 6.0, low=0.0, high=0.72)
    competitive_pressure = correlated_ratio(
        rng,
        pricing_power,
        -0.77,
        mean=competitive_anchor,
        stddev=0.16,
    )
    contract_revenue_share = correlated_ratio(
        rng,
        customer_concentration,
        0.72,
        mean=0.27,
        stddev=0.16,
    )
    capacity_utilization = correlated_ratio(
        rng,
        demand_exposure,
        0.34,
        mean=0.72,
        stddev=0.12,
    )
    fixed_cost_anchor = min(
        1.0,
        0.68 * capital_intensity + 0.22 * costs.financing + 0.10 * costs.energy,
    )
    fixed_cost_share = correlated_ratio(
        rng,
        fixed_cost_anchor,
        0.72,
        mean=0.43,
        stddev=0.16,
    )

    context = EconomicContext(
        input_dependency=input_dependency,
        output_dependency=output_dependency,
        demand_exposure=demand_exposure,
        hedging=hedging,
        pricing_power=pricing_power,
        substitutability=substitutability,
        inventory_buffer=inventory_buffer,
        supplier_diversification=supplier_diversification,
        customer_concentration=customer_concentration,
        debt_ratio=debt_ratio,
        cash_buffer=cash_buffer,
        currency_revenue_exposure=currency_revenue_exposure,
        currency_cost_exposure=currency_cost_exposure,
        regulatory_exposure=regulatory_exposure,
        tax_exposure=tax_exposure,
        subsidy_exposure=subsidy_exposure,
        competitive_pressure=competitive_pressure,
        contract_revenue_share=contract_revenue_share,
        labor_intensity=costs.labor,
        capacity_utilization=capacity_utilization,
        fixed_cost_share=fixed_cost_share,
    )
    return SyntheticCompany(profile=profile, context=context)


def generate_company(seed: int = 42, *, index: int = 1) -> SyntheticCompany:
    """Generate one reproducible company with ID ``synthetic_NNNNN``.

    A company-specific derived seed makes the result independent of generation
    order.  Consequently ``generate_company(seed=s, index=n)`` equals element
    ``n`` of ``generate_companies(..., seed=s)``.
    """

    canonical_seed = _integer("seed", seed)
    canonical_index = _integer("index", index, minimum=1)
    rng = random.Random(_derived_seed(canonical_seed, canonical_index))
    return _sample_company(rng, f"synthetic_{canonical_index:05d}")


def generate_companies(count: int, seed: int = 42) -> tuple[SyntheticCompany, ...]:
    """Generate a deterministic tuple of unique persistent company states."""

    canonical_count = _integer("count", count)
    canonical_seed = _integer("seed", seed)
    return tuple(
        generate_company(canonical_seed, index=index)
        for index in range(1, canonical_count + 1)
    )


def company_context(company: SyntheticCompany) -> EconomicContext:
    """Return the immutable, complete structural context stored on a company."""

    if not isinstance(company, SyntheticCompany):
        raise TypeError("company must be a SyntheticCompany")
    return company.context


def context_for_event(
    company: SyntheticCompany,
    event_or_mechanism: Event | str,
) -> EconomicContext:
    """Select stable company factors required by one event mechanism.

    Values are copied from the persistent company state and never resampled.
    Passing a full :class:`Event` additionally validates its direction and
    subject class against the mechanism registry.
    """

    source = company_context(company)
    if isinstance(event_or_mechanism, Event):
        mechanism_name = event_or_mechanism.mechanism
        definition = get_mechanism(mechanism_name)
        definition.validate_event(
            direction=event_or_mechanism.direction,
            subject_class=event_or_mechanism.subject_class,
        )
    elif isinstance(event_or_mechanism, str):
        definition = get_mechanism(event_or_mechanism)
    else:
        raise TypeError("event_or_mechanism must be an Event or mechanism string")

    selected = {
        name: getattr(source, name)
        for name in definition.required_context_fields
    }
    context = EconomicContext(**selected)
    definition.validate_context(context)
    return context


__all__ = [
    "COMPANY_GENERATOR_VERSION",
    "SyntheticCompany",
    "company_context",
    "context_for_event",
    "generate_companies",
    "generate_company",
]

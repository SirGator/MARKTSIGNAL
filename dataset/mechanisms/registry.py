"""Immutable registry for the canonical mechanism taxonomy."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from dataset.schema import EconomicContext

from .base import MechanismDefinition
from .costs import COST_MECHANISMS
from .market import MARKET_MECHANISMS
from .operations import OPERATIONAL_MECHANISMS
from .policy import POLICY_MECHANISMS
from .relationships import RELATIONSHIP_MECHANISMS


MECHANISM_NAMES = (
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


def _build_registry() -> Mapping[str, MechanismDefinition]:
    definitions = (
        *COST_MECHANISMS,
        *MARKET_MECHANISMS,
        *OPERATIONAL_MECHANISMS,
        *POLICY_MECHANISMS,
        *RELATIONSHIP_MECHANISMS,
    )
    registry: dict[str, MechanismDefinition] = {}
    for definition in definitions:
        if definition.name in registry:
            raise RuntimeError(f"duplicate mechanism definition: {definition.name}")
        registry[definition.name] = definition

    expected = set(MECHANISM_NAMES)
    actual = set(registry)
    if actual != expected:
        missing = ", ".join(sorted(expected - actual)) or "none"
        extra = ", ".join(sorted(actual - expected)) or "none"
        raise RuntimeError(
            f"mechanism registry mismatch; missing: {missing}; extra: {extra}"
        )

    ordered = {name: registry[name] for name in MECHANISM_NAMES}
    return MappingProxyType(ordered)


MECHANISM_REGISTRY = _build_registry()


def list_mechanisms() -> tuple[str, ...]:
    """Return all canonical mechanism names in stable contract order."""

    return tuple(MECHANISM_REGISTRY)


def get_mechanism(name: str) -> MechanismDefinition:
    """Return one mechanism definition or raise a descriptive ``KeyError``."""

    if not isinstance(name, str):
        raise TypeError("mechanism name must be a string")
    normalized = name.strip().casefold()
    try:
        return MECHANISM_REGISTRY[normalized]
    except KeyError:
        raise KeyError(f"unknown economic mechanism: {name!r}") from None


def validate_mechanism(
    name: str,
    direction: str,
    subject_class: str,
    context: EconomicContext,
) -> MechanismDefinition:
    """Validate an event/context tuple and return its canonical definition."""

    definition = get_mechanism(name)
    definition.validate(
        direction=direction,
        subject_class=subject_class,
        context=context,
    )
    return definition


__all__ = [
    "MECHANISM_NAMES",
    "MECHANISM_REGISTRY",
    "get_mechanism",
    "list_mechanisms",
    "validate_mechanism",
]

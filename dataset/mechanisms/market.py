"""Product, demand, supply, currency, and competition mechanisms."""

from __future__ import annotations

from .base import MechanismDefinition, ratio_field


OUTPUT_PRICE_CHANGE = MechanismDefinition(
    name="output_price_change",
    directions=("increase", "decrease"),
    subject_classes=("output", "product", "service", "commodity"),
    context_fields=(
        ratio_field("output_dependency", "Revenue share tied to the affected output."),
        ratio_field("demand_exposure", "Demand exposed to the output-price change."),
        ratio_field("pricing_power", "Ability to sustain the changed output price."),
    ),
    affected_state_fields=(
        "revenue_expectation",
        "demand",
        "margin_expectation",
        "competitive_position",
    ),
    description="A change in the realized price of a product or service sold.",
)


DEMAND_CHANGE = MechanismDefinition(
    name="demand_change",
    directions=("increase", "decrease"),
    subject_classes=("product", "service", "market", "customer_segment"),
    context_fields=(
        ratio_field("demand_exposure", "Revenue exposed to the affected demand pool."),
        ratio_field("pricing_power", "Ability to preserve pricing as demand changes."),
        ratio_field("capacity_utilization", "Existing utilization of available capacity."),
    ),
    affected_state_fields=(
        "revenue_expectation",
        "demand",
        "margin_expectation",
    ),
    description="A change in demand for the target's relevant output.",
)


SUPPLY_CHANGE = MechanismDefinition(
    name="supply_change",
    directions=("increase", "decrease"),
    subject_classes=("production_input", "product", "commodity", "market_supply"),
    context_fields=(
        ratio_field("input_dependency", "Dependence on the affected supply."),
        ratio_field("inventory_buffer", "Inventory available while supply adjusts."),
        ratio_field(
            "supplier_diversification",
            "Diversification across independent sources of supply.",
        ),
        ratio_field("substitutability", "Ability to replace the affected supply."),
    ),
    affected_state_fields=(
        "revenue_expectation",
        "operating_costs",
        "margin_expectation",
        "supply_risk",
    ),
    description="A broad change in the availability of a relevant supply.",
)


CURRENCY_CHANGE = MechanismDefinition(
    name="currency_change",
    directions=("increase", "decrease"),
    subject_classes=("currency", "currency_pair", "foreign_exchange"),
    context_fields=(
        ratio_field(
            "currency_revenue_exposure",
            "Revenue share denominated in the affected currency.",
        ),
        ratio_field(
            "currency_cost_exposure",
            "Cost share denominated in the affected currency.",
        ),
        ratio_field("hedging", "Share of net currency exposure that is hedged."),
        ratio_field("pricing_power", "Ability to reprice after currency changes."),
    ),
    affected_state_fields=(
        "revenue_expectation",
        "operating_costs",
        "margin_expectation",
        "competitive_position",
    ),
    description="A change in a currency relevant to the target's revenues or costs.",
)


COMPETITIVE_PRESSURE_CHANGE = MechanismDefinition(
    name="competitive_pressure_change",
    directions=("increase", "decrease"),
    subject_classes=("competitor", "market", "product", "service"),
    context_fields=(
        ratio_field("competitive_pressure", "Baseline competitive intensity."),
        ratio_field("demand_exposure", "Demand exposed to the competitive change."),
        ratio_field("pricing_power", "Ability to defend price and market share."),
        ratio_field(
            "customer_concentration",
            "Revenue concentration among customers able to switch.",
        ),
    ),
    affected_state_fields=(
        "revenue_expectation",
        "demand",
        "margin_expectation",
        "competitive_position",
    ),
    description="A change in competitive intensity around the target.",
)


MARKET_MECHANISMS = (
    OUTPUT_PRICE_CHANGE,
    DEMAND_CHANGE,
    SUPPLY_CHANGE,
    CURRENCY_CHANGE,
    COMPETITIVE_PRESSURE_CHANGE,
)


__all__ = [
    "COMPETITIVE_PRESSURE_CHANGE",
    "CURRENCY_CHANGE",
    "DEMAND_CHANGE",
    "MARKET_MECHANISMS",
    "OUTPUT_PRICE_CHANGE",
    "SUPPLY_CHANGE",
]

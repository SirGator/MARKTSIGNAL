"""Cost-side economic mechanism definitions."""

from __future__ import annotations

from .base import MechanismDefinition, ratio_field


INPUT_COST_CHANGE = MechanismDefinition(
    name="input_cost_change",
    directions=("increase", "decrease"),
    subject_classes=(
        "production_input",
        "energy",
        "agricultural_input",
        "materials",
        "compute",
        "logistics",
    ),
    context_fields=(
        ratio_field("input_dependency", "Share of activity dependent on the input."),
        ratio_field("hedging", "Share of the input-cost exposure that is hedged."),
        ratio_field("pricing_power", "Ability to pass costs through to customers."),
        ratio_field("substitutability", "Ability to replace the affected input."),
        ratio_field("inventory_buffer", "Existing inventory that delays cost exposure."),
    ),
    affected_state_fields=(
        "operating_costs",
        "margin_expectation",
        "supply_risk",
        "competitive_position",
    ),
    description="A change in the price of an input used by the target entity.",
)


FINANCING_COST_CHANGE = MechanismDefinition(
    name="financing_cost_change",
    directions=("increase", "decrease"),
    subject_classes=("financing", "debt", "credit", "interest_rate"),
    context_fields=(
        ratio_field("debt_ratio", "Debt relative to the target's capital base."),
        ratio_field("cash_buffer", "Liquidity available to absorb financing shocks."),
        ratio_field("hedging", "Share of financing-cost exposure that is hedged."),
        ratio_field("fixed_cost_share", "Share of costs that cannot adjust quickly."),
    ),
    affected_state_fields=(
        "revenue_expectation",
        "margin_expectation",
        "competitive_position",
    ),
    description="A change in the target's cost or availability of financing.",
)


LABOR_COST_CHANGE = MechanismDefinition(
    name="labor_cost_change",
    directions=("increase", "decrease"),
    subject_classes=("labor", "workforce", "occupation", "compensation"),
    context_fields=(
        ratio_field("labor_intensity", "Share of economic activity dependent on labor."),
        ratio_field("pricing_power", "Ability to pass labor costs to customers."),
        ratio_field("fixed_cost_share", "Share of costs that cannot adjust quickly."),
    ),
    affected_state_fields=(
        "operating_costs",
        "margin_expectation",
        "supply_risk",
        "competitive_position",
    ),
    description="A change in wages, benefits, or other workforce costs.",
)


COST_MECHANISMS = (
    INPUT_COST_CHANGE,
    FINANCING_COST_CHANGE,
    LABOR_COST_CHANGE,
)


__all__ = [
    "COST_MECHANISMS",
    "FINANCING_COST_CHANGE",
    "INPUT_COST_CHANGE",
    "LABOR_COST_CHANGE",
]

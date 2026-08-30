"""Regulatory, tax, and subsidy mechanism definitions."""

from __future__ import annotations

from .base import MechanismDefinition, ratio_field


REGULATORY_COST_CHANGE = MechanismDefinition(
    name="regulatory_cost_change",
    directions=("increase", "decrease"),
    subject_classes=("regulation", "jurisdiction", "compliance_requirement"),
    context_fields=(
        ratio_field(
            "regulatory_exposure",
            "Share of activity exposed to the regulatory change.",
        ),
        ratio_field("pricing_power", "Ability to pass compliance costs through."),
        ratio_field("cash_buffer", "Liquidity available for compliance investment."),
    ),
    affected_state_fields=(
        "operating_costs",
        "margin_expectation",
        "competitive_position",
    ),
    description="A change in recurring or transitional regulatory compliance cost.",
)


TAX_CHANGE = MechanismDefinition(
    name="tax_change",
    directions=("increase", "decrease"),
    subject_classes=("tax", "jurisdiction", "tax_regime"),
    context_fields=(
        ratio_field("tax_exposure", "Share of earnings exposed to the tax change."),
        ratio_field("pricing_power", "Ability to offset tax effects through pricing."),
        ratio_field("cash_buffer", "Liquidity available to absorb the tax change."),
    ),
    affected_state_fields=("margin_expectation", "competitive_position"),
    description="A change in taxes borne by the target entity.",
)


SUBSIDY_CHANGE = MechanismDefinition(
    name="subsidy_change",
    directions=("increase", "decrease"),
    subject_classes=("subsidy", "government_program", "jurisdiction"),
    context_fields=(
        ratio_field("subsidy_exposure", "Share of activity supported by the subsidy."),
        ratio_field("cash_buffer", "Liquidity available if support changes."),
        ratio_field("pricing_power", "Ability to reprice after support changes."),
    ),
    affected_state_fields=(
        "revenue_expectation",
        "operating_costs",
        "margin_expectation",
        "competitive_position",
    ),
    description="A change in financial support available to the target or its market.",
)


POLICY_MECHANISMS = (REGULATORY_COST_CHANGE, TAX_CHANGE, SUBSIDY_CHANGE)


__all__ = [
    "POLICY_MECHANISMS",
    "REGULATORY_COST_CHANGE",
    "SUBSIDY_CHANGE",
    "TAX_CHANGE",
]

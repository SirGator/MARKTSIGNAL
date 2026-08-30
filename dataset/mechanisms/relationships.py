"""Customer- and contract-relationship mechanism definitions."""

from __future__ import annotations

from .base import MechanismDefinition, ratio_field


CUSTOMER_DEPENDENCY_CHANGE = MechanismDefinition(
    name="customer_dependency_change",
    directions=("increase", "decrease"),
    subject_classes=("customer", "customer_segment", "distribution_channel"),
    context_fields=(
        ratio_field(
            "customer_concentration",
            "Revenue concentrated in the affected customer relationship.",
        ),
        ratio_field("demand_exposure", "Demand tied to the affected customer base."),
        ratio_field("pricing_power", "Bargaining power relative to customers."),
    ),
    affected_state_fields=(
        "revenue_expectation",
        "demand",
        "margin_expectation",
        "competitive_position",
    ),
    description=(
        "A change in demand or revenue contribution from a customer on which "
        "the target depends."
    ),
)


CONTRACT_CHANGE = MechanismDefinition(
    name="contract_change",
    directions=("increase", "decrease"),
    subject_classes=("contract", "customer_contract", "supplier_contract"),
    context_fields=(
        ratio_field(
            "contract_revenue_share",
            "Revenue share governed by the affected contract.",
        ),
        ratio_field(
            "customer_concentration",
            "Concentration associated with the contract counterparty.",
        ),
        ratio_field("pricing_power", "Bargaining power in contract terms."),
    ),
    affected_state_fields=(
        "revenue_expectation",
        "demand",
        "margin_expectation",
        "competitive_position",
    ),
    description="A change in the economic scope or value of a material contract.",
)


RELATIONSHIP_MECHANISMS = (CUSTOMER_DEPENDENCY_CHANGE, CONTRACT_CHANGE)


__all__ = [
    "CONTRACT_CHANGE",
    "CUSTOMER_DEPENDENCY_CHANGE",
    "RELATIONSHIP_MECHANISMS",
]

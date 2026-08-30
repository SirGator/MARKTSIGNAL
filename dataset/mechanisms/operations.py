"""Capacity and supplier-continuity mechanism definitions."""

from __future__ import annotations

from .base import MechanismDefinition, ratio_field


CAPACITY_CHANGE = MechanismDefinition(
    name="capacity_change",
    directions=("increase", "decrease"),
    subject_classes=("company", "facility", "production_line", "network"),
    context_fields=(
        ratio_field("capacity_utilization", "Current utilization of available capacity."),
        ratio_field("demand_exposure", "Demand served by the affected capacity."),
        ratio_field("fixed_cost_share", "Costs fixed across the capacity change."),
    ),
    affected_state_fields=(
        "revenue_expectation",
        "operating_costs",
        "margin_expectation",
        "supply_risk",
    ),
    description="A change in productive or service-delivery capacity.",
)


SUPPLIER_DISRUPTION = MechanismDefinition(
    name="supplier_disruption",
    directions=("increase", "decrease"),
    subject_classes=("supplier", "facility", "logistics_route", "production_input"),
    context_fields=(
        ratio_field("input_dependency", "Dependence on the disrupted supplier or flow."),
        ratio_field("inventory_buffer", "Inventory available during disruption."),
        ratio_field(
            "supplier_diversification",
            "Diversification across independent suppliers.",
        ),
        ratio_field("substitutability", "Ability to replace the disrupted supply."),
    ),
    affected_state_fields=(
        "revenue_expectation",
        "operating_costs",
        "margin_expectation",
        "supply_risk",
    ),
    description="A change in the severity of a specific supplier disruption.",
)


OPERATIONAL_MECHANISMS = (CAPACITY_CHANGE, SUPPLIER_DISRUPTION)


__all__ = ["CAPACITY_CHANGE", "OPERATIONAL_MECHANISMS", "SUPPLIER_DISRUPTION"]

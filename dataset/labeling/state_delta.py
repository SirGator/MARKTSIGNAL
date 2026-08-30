"""Deterministic economic-mechanism labeling into canonical state deltas.

The functions in this module deliberately use only normalized event fields,
the target's structural context, and the requested horizon.  Subject names and
industry labels never participate in a formula.  Every mechanism first passes
through the declarative registry contract and then follows exactly one pure
labeling function::

    Event + EconomicContext + horizon_days -> StateDelta

All intermediate effects are dimensionless ratios and every returned state
component is bounded to ``[-1, 1]``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
from types import MappingProxyType

from dataset.mechanisms import MECHANISM_NAMES, validate_mechanism
from dataset.schema import EconomicContext, Event, STATE_DELTA_FIELDS, StateDelta


StateLabeler = Callable[[Event, EconomicContext, int], StateDelta]


def _bounded(value: float) -> float:
    """Clamp one finite state contribution to the signed unit interval."""

    if not math.isfinite(value):
        raise ValueError("state-delta formula produced a non-finite value")
    return max(-1.0, min(1.0, float(value)))


def _delta(**changes: float) -> StateDelta:
    unknown = set(changes) - set(STATE_DELTA_FIELDS)
    if unknown:
        raise RuntimeError(
            "state labeler produced unknown fields: " + ", ".join(sorted(unknown))
        )
    values = {name: 0.0 for name in STATE_DELTA_FIELDS}
    values.update({name: _bounded(value) for name, value in changes.items()})
    return StateDelta(**values)


def _context_value(context: EconomicContext, name: str) -> float:
    value = getattr(context, name)
    if value is None:
        # Registry validation normally catches this first.  Keeping the local
        # guard makes individual formula failures descriptive as well.
        raise ValueError(f"context.{name} is required by the state labeler")
    return value


def _direction(event: Event) -> float:
    if event.direction == "increase":
        return 1.0
    if event.direction == "decrease":
        return -1.0
    raise ValueError(f"unsupported event direction: {event.direction!r}")


def _inventory_exposure(inventory_buffer: float, horizon_days: int) -> float:
    """Return exposure after a short-run inventory buffer.

    Inventory is most useful immediately after a shock.  Its protection fades
    smoothly as the horizon grows and replenishment eventually occurs at the
    new economic conditions.
    """

    short_run_weight = math.exp(-horizon_days / 90.0)
    return 1.0 - inventory_buffer * short_run_weight


def _liquidity_exposure(cash_buffer: float, horizon_days: int) -> float:
    """Return adverse-shock exposure after temporary liquidity absorption."""

    short_run_weight = math.exp(-horizon_days / 180.0)
    return 1.0 - 0.65 * cash_buffer * short_run_weight


def _residual_supply_exposure(
    *,
    inventory_buffer: float,
    supplier_diversification: float,
    substitutability: float,
    horizon_days: int,
) -> float:
    return (
        _inventory_exposure(inventory_buffer, horizon_days)
        * (1.0 - supplier_diversification)
        * (1.0 - substitutability)
    )


def _input_cost_change(
    event: Event,
    context: EconomicContext,
    horizon_days: int,
) -> StateDelta:
    dependency = _context_value(context, "input_dependency")
    hedging = _context_value(context, "hedging")
    pricing_power = _context_value(context, "pricing_power")
    substitutability = _context_value(context, "substitutability")
    inventory_buffer = _context_value(context, "inventory_buffer")

    # Core transmission required by the dataset contract.  Inventory shields
    # the short horizon, while pricing power acts only after costs have moved.
    cost_shock = (
        _direction(event)
        * event.magnitude
        * dependency
        * (1.0 - hedging)
        * (1.0 - substitutability)
        * _inventory_exposure(inventory_buffer, horizon_days)
    )
    unpassed_cost = cost_shock * (1.0 - pricing_power)
    return _delta(
        operating_costs=cost_shock,
        margin_expectation=-unpassed_cost,
        supply_risk=0.10 * cost_shock,
        competitive_position=-0.25 * unpassed_cost,
    )


def _output_price_change(
    event: Event,
    context: EconomicContext,
    _horizon_days: int,
) -> StateDelta:
    output_dependency = _context_value(context, "output_dependency")
    demand_exposure = _context_value(context, "demand_exposure")
    pricing_power = _context_value(context, "pricing_power")
    direction = _direction(event)

    realized_price = direction * event.magnitude * output_dependency
    # Pricing power amplifies capture of a price increase but cushions a
    # decrease: a strong brand can hold its price when the market falls.
    capture = 0.55 + 0.45 * pricing_power * direction
    revenue = realized_price * capture
    demand = (
        -direction
        * event.magnitude
        * demand_exposure
        * output_dependency
        * (1.0 - pricing_power)
        * 0.35
    )
    margin = revenue + 0.20 * demand
    return _delta(
        revenue_expectation=revenue,
        demand=demand,
        margin_expectation=margin,
        competitive_position=0.20 * (revenue + demand),
    )


def _demand_change(
    event: Event,
    context: EconomicContext,
    _horizon_days: int,
) -> StateDelta:
    demand_exposure = _context_value(context, "demand_exposure")
    pricing_power = _context_value(context, "pricing_power")
    capacity_utilization = _context_value(context, "capacity_utilization")
    demand = _direction(event) * event.magnitude * demand_exposure

    if demand >= 0.0:
        # Spare capacity governs how much additional demand can be served.
        realization = (
            (0.55 + 0.45 * pricing_power)
            * (1.0 - 0.60 * capacity_utilization)
        )
    else:
        # Pricing power cushions a contraction but never reverses it.
        realization = 1.0 - 0.50 * pricing_power
    revenue = demand * realization
    margin = revenue * (0.55 + 0.45 * pricing_power)
    return _delta(
        revenue_expectation=revenue,
        demand=demand,
        margin_expectation=margin,
    )


def _supply_change(
    event: Event,
    context: EconomicContext,
    horizon_days: int,
) -> StateDelta:
    dependency = _context_value(context, "input_dependency")
    residual = _residual_supply_exposure(
        inventory_buffer=_context_value(context, "inventory_buffer"),
        supplier_diversification=_context_value(
            context,
            "supplier_diversification",
        ),
        substitutability=_context_value(context, "substitutability"),
        horizon_days=horizon_days,
    )
    availability = _direction(event) * event.magnitude * dependency * residual
    return _delta(
        revenue_expectation=0.35 * availability,
        operating_costs=-0.55 * availability,
        margin_expectation=0.65 * availability,
        supply_risk=-availability,
    )


def _capacity_change(
    event: Event,
    context: EconomicContext,
    _horizon_days: int,
) -> StateDelta:
    utilization = _context_value(context, "capacity_utilization")
    demand_exposure = _context_value(context, "demand_exposure")
    fixed_cost_share = _context_value(context, "fixed_cost_share")
    direction = _direction(event)

    useful_capacity = (
        direction
        * event.magnitude
        * demand_exposure
        * (0.25 + 0.75 * utilization)
    )
    fixed_cost_change = direction * event.magnitude * fixed_cost_share * 0.25
    return _delta(
        revenue_expectation=0.80 * useful_capacity,
        operating_costs=fixed_cost_change,
        margin_expectation=0.65 * useful_capacity - 0.50 * fixed_cost_change,
        supply_risk=-0.50 * useful_capacity,
    )


def _supplier_disruption(
    event: Event,
    context: EconomicContext,
    horizon_days: int,
) -> StateDelta:
    dependency = _context_value(context, "input_dependency")
    residual = _residual_supply_exposure(
        inventory_buffer=_context_value(context, "inventory_buffer"),
        supplier_diversification=_context_value(
            context,
            "supplier_diversification",
        ),
        substitutability=_context_value(context, "substitutability"),
        horizon_days=horizon_days,
    )
    disruption = _direction(event) * event.magnitude * dependency * residual
    return _delta(
        revenue_expectation=-0.55 * disruption,
        operating_costs=0.50 * disruption,
        margin_expectation=-0.70 * disruption,
        supply_risk=disruption,
    )


def _customer_dependency_change(
    event: Event,
    context: EconomicContext,
    _horizon_days: int,
) -> StateDelta:
    concentration = _context_value(context, "customer_concentration")
    demand_exposure = _context_value(context, "demand_exposure")
    pricing_power = _context_value(context, "pricing_power")
    direction = _direction(event)

    # Direction describes the affected customer's contribution/demand, while
    # concentration describes how dependent the target is on that customer.
    revenue = direction * event.magnitude * concentration
    demand = direction * event.magnitude * demand_exposure
    # Pricing power protects the margin when the relationship turns adverse
    # and improves capture when it strengthens.
    margin = revenue * (0.55 + 0.45 * pricing_power * direction)
    bargaining_risk = (
        -direction
        * event.magnitude
        * concentration
        * (1.0 - pricing_power)
        * 0.25
    )
    return _delta(
        revenue_expectation=revenue,
        demand=demand,
        margin_expectation=margin,
        competitive_position=bargaining_risk,
    )


def _financing_cost_change(
    event: Event,
    context: EconomicContext,
    horizon_days: int,
) -> StateDelta:
    debt_ratio = _context_value(context, "debt_ratio")
    cash_buffer = _context_value(context, "cash_buffer")
    hedging = _context_value(context, "hedging")
    fixed_cost_share = _context_value(context, "fixed_cost_share")
    direction = _direction(event)

    pressure = direction * event.magnitude * debt_ratio * (1.0 - hedging)
    if pressure > 0.0:
        pressure *= _liquidity_exposure(cash_buffer, horizon_days)
    return _delta(
        revenue_expectation=-0.25 * pressure * fixed_cost_share,
        margin_expectation=-pressure,
        competitive_position=-pressure * (0.25 + 0.35 * fixed_cost_share),
    )


def _currency_change(
    event: Event,
    context: EconomicContext,
    _horizon_days: int,
) -> StateDelta:
    revenue_exposure = _context_value(context, "currency_revenue_exposure")
    cost_exposure = _context_value(context, "currency_cost_exposure")
    hedging = _context_value(context, "hedging")
    pricing_power = _context_value(context, "pricing_power")
    signed_move = _direction(event) * event.magnitude * (1.0 - hedging)

    revenue = signed_move * revenue_exposure
    operating_costs = signed_move * cost_exposure
    # Pricing power cushions the adverse side of a currency move and amplifies
    # the favorable side; it never converts a loss into a gain.
    margin = revenue - operating_costs * (1.0 - 0.65 * pricing_power * _direction(event))
    return _delta(
        revenue_expectation=revenue,
        operating_costs=operating_costs,
        margin_expectation=margin,
        competitive_position=0.25 * margin,
    )


def _regulatory_cost_change(
    event: Event,
    context: EconomicContext,
    horizon_days: int,
) -> StateDelta:
    exposure = _context_value(context, "regulatory_exposure")
    pricing_power = _context_value(context, "pricing_power")
    cash_buffer = _context_value(context, "cash_buffer")
    pressure = _direction(event) * event.magnitude * exposure
    if pressure > 0.0:
        pressure *= _liquidity_exposure(cash_buffer, horizon_days)
    unpassed = pressure * (1.0 - pricing_power)
    return _delta(
        operating_costs=pressure,
        margin_expectation=-unpassed,
        competitive_position=-0.30 * unpassed,
    )


def _tax_change(
    event: Event,
    context: EconomicContext,
    horizon_days: int,
) -> StateDelta:
    exposure = _context_value(context, "tax_exposure")
    pricing_power = _context_value(context, "pricing_power")
    cash_buffer = _context_value(context, "cash_buffer")
    pressure = _direction(event) * event.magnitude * exposure
    if pressure > 0.0:
        pressure *= _liquidity_exposure(cash_buffer, horizon_days)
    return _delta(
        margin_expectation=-pressure * (1.0 - 0.35 * pricing_power),
        competitive_position=-0.25 * pressure * (1.0 - pricing_power),
    )


def _subsidy_change(
    event: Event,
    context: EconomicContext,
    horizon_days: int,
) -> StateDelta:
    exposure = _context_value(context, "subsidy_exposure")
    cash_buffer = _context_value(context, "cash_buffer")
    pricing_power = _context_value(context, "pricing_power")
    support = _direction(event) * event.magnitude * exposure
    if support < 0.0:
        support *= _liquidity_exposure(cash_buffer, horizon_days)
    return _delta(
        revenue_expectation=0.35 * support,
        operating_costs=-0.65 * support,
        margin_expectation=support,
        competitive_position=0.30 * support * (1.0 - 0.50 * pricing_power),
    )


def _competitive_pressure_change(
    event: Event,
    context: EconomicContext,
    _horizon_days: int,
) -> StateDelta:
    baseline_pressure = _context_value(context, "competitive_pressure")
    demand_exposure = _context_value(context, "demand_exposure")
    pricing_power = _context_value(context, "pricing_power")
    concentration = _context_value(context, "customer_concentration")

    pressure = _direction(event) * event.magnitude * baseline_pressure
    vulnerability = (1.0 - 0.65 * pricing_power) * (
        0.60 + 0.40 * concentration
    )
    commercial_effect = pressure * demand_exposure * vulnerability
    return _delta(
        revenue_expectation=-commercial_effect,
        demand=-pressure * demand_exposure * (0.70 + 0.30 * concentration),
        margin_expectation=-commercial_effect * (0.70 + 0.30 * (1.0 - pricing_power)),
        competitive_position=-pressure * (1.0 - 0.40 * pricing_power),
    )


def _contract_change(
    event: Event,
    context: EconomicContext,
    _horizon_days: int,
) -> StateDelta:
    revenue_share = _context_value(context, "contract_revenue_share")
    concentration = _context_value(context, "customer_concentration")
    pricing_power = _context_value(context, "pricing_power")
    contract_effect = _direction(event) * event.magnitude * revenue_share
    return _delta(
        revenue_expectation=contract_effect,
        demand=contract_effect * (0.50 + 0.50 * concentration),
        margin_expectation=contract_effect * (0.50 + 0.50 * pricing_power * _direction(event)),
        competitive_position=0.25 * contract_effect * (0.50 + 0.50 * pricing_power * _direction(event)),
    )


def _labor_cost_change(
    event: Event,
    context: EconomicContext,
    _horizon_days: int,
) -> StateDelta:
    labor_intensity = _context_value(context, "labor_intensity")
    pricing_power = _context_value(context, "pricing_power")
    fixed_cost_share = _context_value(context, "fixed_cost_share")
    pressure = _direction(event) * event.magnitude * labor_intensity
    operating_costs = pressure * (0.70 + 0.30 * fixed_cost_share)
    unpassed = pressure * (1.0 - 0.75 * pricing_power)
    return _delta(
        operating_costs=operating_costs,
        margin_expectation=-unpassed,
        supply_risk=0.20 * pressure * fixed_cost_share,
        competitive_position=-0.25 * pressure * (1.0 - pricing_power),
    )


_LABELERS: dict[str, StateLabeler] = {
    "input_cost_change": _input_cost_change,
    "output_price_change": _output_price_change,
    "demand_change": _demand_change,
    "supply_change": _supply_change,
    "capacity_change": _capacity_change,
    "supplier_disruption": _supplier_disruption,
    "customer_dependency_change": _customer_dependency_change,
    "financing_cost_change": _financing_cost_change,
    "currency_change": _currency_change,
    "regulatory_cost_change": _regulatory_cost_change,
    "tax_change": _tax_change,
    "subsidy_change": _subsidy_change,
    "competitive_pressure_change": _competitive_pressure_change,
    "contract_change": _contract_change,
    "labor_cost_change": _labor_cost_change,
}

if set(_LABELERS) != set(MECHANISM_NAMES):
    missing = sorted(set(MECHANISM_NAMES) - set(_LABELERS))
    extra = sorted(set(_LABELERS) - set(MECHANISM_NAMES))
    raise RuntimeError(
        f"state-labeler registry mismatch; missing={missing}; extra={extra}"
    )

STATE_DELTA_LABELERS: Mapping[str, StateLabeler] = MappingProxyType(
    {name: _LABELERS[name] for name in MECHANISM_NAMES}
)


def compute_state_delta(
    event: Event,
    context: EconomicContext,
    horizon_days: int,
) -> StateDelta:
    """Compute a deterministic state delta for one validated event/context."""

    if not isinstance(event, Event):
        raise TypeError("event must be an Event")
    if not isinstance(context, EconomicContext):
        raise TypeError("context must be an EconomicContext")
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, int):
        raise TypeError("horizon_days must be an integer")
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1")

    validate_mechanism(
        event.mechanism,
        event.direction,
        event.subject_class,
        context,
    )
    try:
        labeler = STATE_DELTA_LABELERS[event.mechanism]
    except KeyError:
        raise KeyError(
            f"no state-delta labeler for mechanism {event.mechanism!r}"
        ) from None
    return labeler(event, context, horizon_days)


__all__ = ["STATE_DELTA_LABELERS", "StateLabeler", "compute_state_delta"]

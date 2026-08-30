"""Structured verification for frozen economic dataset records.

Schema constructors enforce local types and bounds.  This verifier adds the
cross-field guarantees that a single dataclass cannot establish: mechanism
registry compatibility, deterministic relabeling parity, impact aggregation,
state-field ownership, and qualitative economic sign invariants.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from dataset.mechanisms import (
    MECHANISM_NAMES,
    MECHANISM_REGISTRY,
    get_mechanism,
    validate_mechanism,
)
from dataset.schema import (
    CONTEXT_FACTOR_FIELDS,
    MECHANISMS,
    STATE_DELTA_FIELDS,
    DatasetRecord,
)

from .impact import compute_economic_impact
from .state_delta import STATE_DELTA_LABELERS, compute_state_delta


_TOLERANCE = 1e-12
_INDEPENDENT_LABEL_SOURCES = frozenset(("teacher", "curated", "real"))


@dataclass(frozen=True, slots=True)
class VerificationIssue:
    """One machine-readable dataset consistency problem."""

    code: str
    path: str
    message: str
    severity: str = "error"


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Complete non-raising result of :func:`verify_record`."""

    issues: tuple[VerificationIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def valid(self) -> bool:
        """Short compatibility alias for ``is_valid``."""

        return self.is_valid

    @property
    def errors(self) -> tuple[VerificationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    def __bool__(self) -> bool:
        return self.is_valid


class RecordVerificationError(ValueError):
    """Raised when a caller requires a fully valid dataset record."""

    def __init__(self, issues: tuple[VerificationIssue, ...]) -> None:
        self.issues = issues
        details = "; ".join(
            f"{issue.path}: {issue.message}" for issue in issues
        )
        super().__init__(f"dataset record failed verification: {details}")


def _issue(
    issues: list[VerificationIssue],
    code: str,
    path: str,
    message: str,
) -> None:
    issues.append(VerificationIssue(code=code, path=path, message=message))


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _check_bound(
    issues: list[VerificationIssue],
    *,
    path: str,
    value: object,
    minimum: float,
    maximum: float,
) -> None:
    if not _is_finite_number(value):
        _issue(issues, "non_finite_number", path, "must be a finite number")
        return
    numeric = float(value)
    if not minimum <= numeric <= maximum:
        _issue(
            issues,
            "number_out_of_bounds",
            path,
            f"must be between {minimum:g} and {maximum:g}",
        )


# Multiplier relative to the event direction.  +1 means the state component
# must not move against the event direction; -1 means it must not move with it.
# Zero remains valid because a zero exposure or a complete hedge can neutralize
# a mechanism.
_SIGN_RULES: dict[str, dict[str, float]] = {
    "input_cost_change": {
        "operating_costs": 1.0,
        "margin_expectation": -1.0,
        "supply_risk": 1.0,
        "competitive_position": -1.0,
    },
    "output_price_change": {
        "revenue_expectation": 1.0,
        "demand": -1.0,
    },
    "demand_change": {
        "revenue_expectation": 1.0,
        "demand": 1.0,
        "margin_expectation": 1.0,
    },
    "supply_change": {
        "revenue_expectation": 1.0,
        "operating_costs": -1.0,
        "margin_expectation": 1.0,
        "supply_risk": -1.0,
    },
    "capacity_change": {
        "revenue_expectation": 1.0,
        "operating_costs": 1.0,
        "supply_risk": -1.0,
    },
    "supplier_disruption": {
        "revenue_expectation": -1.0,
        "operating_costs": 1.0,
        "margin_expectation": -1.0,
        "supply_risk": 1.0,
    },
    "customer_dependency_change": {
        "revenue_expectation": 1.0,
        "demand": 1.0,
        "margin_expectation": 1.0,
        "competitive_position": -1.0,
    },
    "financing_cost_change": {
        "revenue_expectation": -1.0,
        "margin_expectation": -1.0,
        "competitive_position": -1.0,
    },
    "currency_change": {},
    "regulatory_cost_change": {
        "operating_costs": 1.0,
        "margin_expectation": -1.0,
        "competitive_position": -1.0,
    },
    "tax_change": {
        "margin_expectation": -1.0,
        "competitive_position": -1.0,
    },
    "subsidy_change": {
        "revenue_expectation": 1.0,
        "operating_costs": -1.0,
        "margin_expectation": 1.0,
        "competitive_position": 1.0,
    },
    "competitive_pressure_change": {
        "revenue_expectation": -1.0,
        "demand": -1.0,
        "margin_expectation": -1.0,
        "competitive_position": -1.0,
    },
    "contract_change": {
        "revenue_expectation": 1.0,
        "demand": 1.0,
        "margin_expectation": 1.0,
        "competitive_position": 1.0,
    },
    "labor_cost_change": {
        "operating_costs": 1.0,
        "margin_expectation": -1.0,
        "supply_risk": 1.0,
        "competitive_position": -1.0,
    },
}


def _check_registry_contract(issues: list[VerificationIssue]) -> None:
    names = tuple(MECHANISM_NAMES)
    if names != tuple(MECHANISMS):
        _issue(
            issues,
            "registry_schema_mismatch",
            "event.mechanism",
            "mechanism registry order differs from the schema contract",
        )
    if tuple(MECHANISM_REGISTRY) != names:
        _issue(
            issues,
            "registry_definition_mismatch",
            "event.mechanism",
            "mechanism definitions do not exactly cover the registry contract",
        )
    if tuple(STATE_DELTA_LABELERS) != names:
        _issue(
            issues,
            "registry_labeler_mismatch",
            "event.mechanism",
            "state labelers do not exactly cover the registry contract",
        )
    if set(_SIGN_RULES) != set(names):
        _issue(
            issues,
            "registry_sign_rule_mismatch",
            "state_delta",
            "sign rules do not exactly cover the registry contract",
        )


def _check_sign_invariants(
    record: DatasetRecord,
    issues: list[VerificationIssue],
) -> None:
    direction = 1.0 if record.event.direction == "increase" else -1.0
    rules = _SIGN_RULES.get(record.event.mechanism, {})
    for field_name, relative_sign in rules.items():
        value = getattr(record.state_delta, field_name)
        if value * direction * relative_sign < -_TOLERANCE:
            _issue(
                issues,
                "state_sign_invariant",
                f"state_delta.{field_name}",
                f"has an invalid sign for {record.event.direction} "
                f"{record.event.mechanism}",
            )

    if record.event.mechanism == "input_cost_change":
        cost = abs(record.state_delta.operating_costs)
        margin = abs(record.state_delta.margin_expectation)
        supply_risk = abs(record.state_delta.supply_risk)
        competitive_position = abs(record.state_delta.competitive_position)
        if margin > cost + _TOLERANCE:
            _issue(
                issues,
                "input_cost_margin_invariant",
                "state_delta.margin_expectation",
                "pricing-power mitigation cannot make margin move more than costs",
            )
        if supply_risk > cost + _TOLERANCE:
            _issue(
                issues,
                "input_cost_supply_invariant",
                "state_delta.supply_risk",
                "supply-risk movement cannot exceed the transmitted cost shock",
            )
        if competitive_position > margin + _TOLERANCE:
            _issue(
                issues,
                "input_cost_competition_invariant",
                "state_delta.competitive_position",
                "competitive movement cannot exceed the unpassed margin shock",
            )


def verify_record(record: DatasetRecord) -> VerificationResult:
    """Return all cross-field consistency issues without raising.

    A valid result guarantees that recomputing the record through the current
    mechanism labeler and impact aggregation reproduces its stored labels.
    """

    issues: list[VerificationIssue] = []
    _check_registry_contract(issues)

    if not isinstance(record, DatasetRecord):
        _issue(
            issues,
            "record_type",
            "record",
            "must be a DatasetRecord",
        )
        return VerificationResult(tuple(issues))

    _check_bound(
        issues,
        path="event.magnitude",
        value=record.event.magnitude,
        minimum=0.0,
        maximum=1.0,
    )
    if (
        isinstance(record.horizon_days, bool)
        or not isinstance(record.horizon_days, int)
        or record.horizon_days < 1
    ):
        _issue(
            issues,
            "invalid_horizon",
            "horizon_days",
            "must be an integer of at least 1",
        )
    _check_bound(
        issues,
        path="economic_impact",
        value=record.economic_impact,
        minimum=-1.0,
        maximum=1.0,
    )
    _check_bound(
        issues,
        path="confidence",
        value=record.confidence,
        minimum=0.0,
        maximum=1.0,
    )

    for name in CONTEXT_FACTOR_FIELDS:
        value = getattr(record.context, name)
        if value is not None:
            _check_bound(
                issues,
                path=f"context.{name}",
                value=value,
                minimum=0.0,
                maximum=1.0,
            )
    for name in STATE_DELTA_FIELDS:
        _check_bound(
            issues,
            path=f"state_delta.{name}",
            value=getattr(record.state_delta, name),
            minimum=-1.0,
            maximum=1.0,
        )

    definition = None
    try:
        definition = get_mechanism(record.event.mechanism)
        validate_mechanism(
            record.event.mechanism,
            record.event.direction,
            record.event.subject_class,
            record.context,
        )
    except (KeyError, TypeError, ValueError) as exc:
        _issue(
            issues,
            "mechanism_contract",
            "event.mechanism",
            str(exc),
        )

    if definition is not None:
        affected = set(definition.affected_state_fields)
        for name in STATE_DELTA_FIELDS:
            if name not in affected and abs(getattr(record.state_delta, name)) > _TOLERANCE:
                _issue(
                    issues,
                    "undeclared_state_effect",
                    f"state_delta.{name}",
                    f"is not declared as affected by {definition.name}",
                )

    # Rule-generated records must reproduce the exact versioned formula.
    # Teacher/curated records use the same schema, affected-field contract,
    # sign checks, and impact aggregation, while retaining independent deltas.
    expected_delta = None
    if (
        definition is not None
        and record.horizon_days >= 1
        and record.metadata.source.casefold() not in _INDEPENDENT_LABEL_SOURCES
    ):
        try:
            expected_delta = compute_state_delta(
                record.event,
                record.context,
                record.horizon_days,
            )
        except (KeyError, TypeError, ValueError) as exc:
            _issue(
                issues,
                "state_labeling_failure",
                "state_delta",
                str(exc),
            )

    if expected_delta is not None:
        for name in STATE_DELTA_FIELDS:
            actual = getattr(record.state_delta, name)
            expected = getattr(expected_delta, name)
            if not math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=_TOLERANCE,
            ):
                _issue(
                    issues,
                    "state_delta_mismatch",
                    f"state_delta.{name}",
                    f"stored {actual:g} does not match deterministic label "
                    f"{expected:g}",
                )

    try:
        expected_impact = compute_economic_impact(record.state_delta)
    except (TypeError, ValueError) as exc:
        _issue(issues, "impact_failure", "economic_impact", str(exc))
    else:
        if not math.isclose(
            record.economic_impact,
            expected_impact,
            rel_tol=0.0,
            abs_tol=_TOLERANCE,
        ):
            _issue(
                issues,
                "economic_impact_mismatch",
                "economic_impact",
                f"stored {record.economic_impact:g} does not match StateDelta "
                f"aggregation {expected_impact:g}",
            )

    _check_sign_invariants(record, issues)
    return VerificationResult(tuple(issues))


def require_valid_record(record: DatasetRecord) -> DatasetRecord:
    """Return ``record`` or raise :class:`RecordVerificationError`."""

    result = verify_record(record)
    if not result.is_valid:
        raise RecordVerificationError(result.errors)
    return record


__all__ = [
    "RecordVerificationError",
    "VerificationIssue",
    "VerificationResult",
    "require_valid_record",
    "verify_record",
]

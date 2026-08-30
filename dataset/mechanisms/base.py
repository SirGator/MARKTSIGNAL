"""Declarative contracts for abstract economic event mechanisms.

Mechanism definitions intentionally describe inputs and possible state effects
only.  Economic labeling formulas belong in a later labeling layer so that
the mechanism taxonomy can remain stable while formulas are reviewed and
versioned independently.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
import re

from dataset.schema import EconomicContext, StateDelta


_TOKEN_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _token(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip().casefold()
    if not normalized or _TOKEN_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a non-empty snake_case token")
    return normalized


def _finite_number(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class ContextFieldSpec:
    """One required, bounded numeric input for a mechanism."""

    name: str
    minimum: float = 0.0
    maximum: float = 1.0
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _token("name", self.name))
        minimum = _finite_number("minimum", self.minimum)
        maximum = _finite_number("maximum", self.maximum)
        if minimum > maximum:
            raise ValueError("minimum must not exceed maximum")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        if not isinstance(self.description, str):
            raise TypeError("description must be a string")
        object.__setattr__(self, "description", self.description.strip())

        context_fields = {field.name for field in fields(EconomicContext)}
        if self.name not in context_fields:
            raise ValueError(
                f"context field {self.name!r} is not part of EconomicContext"
            )

    def validate(self, value: float) -> float:
        """Return a canonical float or reject an invalid field value."""

        result = _finite_number(self.name, value)
        if not self.minimum <= result <= self.maximum:
            raise ValueError(
                f"{self.name} must be between {self.minimum:g} and "
                f"{self.maximum:g}"
            )
        return result


def ratio_field(name: str, description: str) -> ContextFieldSpec:
    """Build a required context ratio constrained to the closed unit interval."""

    return ContextFieldSpec(
        name=name,
        minimum=0.0,
        maximum=1.0,
        description=description,
    )


@dataclass(frozen=True, slots=True)
class MechanismDefinition:
    """Immutable declaration of one reusable economic mechanism."""

    name: str
    directions: tuple[str, ...]
    subject_classes: tuple[str, ...]
    context_fields: tuple[ContextFieldSpec, ...]
    affected_state_fields: tuple[str, ...]
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _token("name", self.name))
        directions = self._normalized_tokens("directions", self.directions)
        subject_classes = self._normalized_tokens(
            "subject_classes",
            self.subject_classes,
        )
        affected_state_fields = self._normalized_tokens(
            "affected_state_fields",
            self.affected_state_fields,
        )
        context_fields = tuple(self.context_fields)

        if set(directions) != {"increase", "decrease"}:
            raise ValueError(
                "directions must declare both 'increase' and 'decrease'"
            )
        if not subject_classes:
            raise ValueError("subject_classes must not be empty")
        if not context_fields:
            raise ValueError("context_fields must not be empty")
        if not affected_state_fields:
            raise ValueError("affected_state_fields must not be empty")
        if any(not isinstance(spec, ContextFieldSpec) for spec in context_fields):
            raise TypeError("context_fields must contain ContextFieldSpec values")

        context_names = tuple(spec.name for spec in context_fields)
        if len(context_names) != len(set(context_names)):
            raise ValueError("context_fields must not contain duplicate names")

        state_fields = {field.name for field in fields(StateDelta)}
        unknown_state_fields = set(affected_state_fields) - state_fields
        if unknown_state_fields:
            unknown = ", ".join(sorted(unknown_state_fields))
            raise ValueError(f"unknown StateDelta fields: {unknown}")

        if not isinstance(self.description, str):
            raise TypeError("description must be a string")

        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "subject_classes", subject_classes)
        object.__setattr__(self, "context_fields", context_fields)
        object.__setattr__(self, "affected_state_fields", affected_state_fields)
        object.__setattr__(self, "description", self.description.strip())

    @staticmethod
    def _normalized_tokens(name: str, values: tuple[str, ...]) -> tuple[str, ...]:
        if isinstance(values, str):
            raise TypeError(f"{name} must be a tuple of strings, not a string")
        normalized = tuple(_token(name, value) for value in values)
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"{name} must not contain duplicates")
        return normalized

    @property
    def required_context_fields(self) -> tuple[str, ...]:
        """Names of all context values required by this mechanism."""

        return tuple(spec.name for spec in self.context_fields)

    def validate_event(self, *, direction: str, subject_class: str) -> None:
        """Validate event taxonomy values without inspecting a subject name."""

        normalized_direction = _token("direction", direction)
        if normalized_direction not in self.directions:
            raise ValueError(
                f"direction {direction!r} is not valid for mechanism {self.name!r}"
            )
        normalized_subject_class = _token("subject_class", subject_class)
        if normalized_subject_class not in self.subject_classes:
            raise ValueError(
                f"subject_class {subject_class!r} is not valid for mechanism "
                f"{self.name!r}"
            )

    def validate_context(self, context: EconomicContext) -> None:
        """Require this mechanism's declared values on an EconomicContext."""

        if not isinstance(context, EconomicContext):
            raise TypeError("context must be an EconomicContext")
        missing: list[str] = []
        for spec in self.context_fields:
            value = getattr(context, spec.name)
            if value is None:
                missing.append(spec.name)
            else:
                spec.validate(value)
        if missing:
            names = ", ".join(missing)
            raise ValueError(
                f"mechanism {self.name!r} requires context fields: {names}"
            )

    def validate(
        self,
        *,
        direction: str,
        subject_class: str,
        context: EconomicContext,
    ) -> None:
        """Validate an event and context against this declaration."""

        self.validate_event(direction=direction, subject_class=subject_class)
        self.validate_context(context)


__all__ = ["ContextFieldSpec", "MechanismDefinition", "ratio_field"]

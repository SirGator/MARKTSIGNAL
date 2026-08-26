"""Parametric economic scenario generator (V2 — consistent text/label).

Every variable that influences the score is explicitly part of the input text.
Every template has exactly one direction. Text and label can never contradict.

Score = f(magnitude, exposure, role, hedging, pricing_power, substitution, horizon)
ALL of these appear in the serialized context.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EconomicScenario:
    """One parametric economic scenario."""

    event_type: str
    direction: str
    event_text: str
    case_role: str
    case_type: str
    context_text: str
    magnitude: float
    exposure: float
    hedging: float
    pricing_power: float
    substitution: float
    horizon_days: int
    score: float
    subject: str = ""
    magnitude_unit: str = "percent"
    paraphrases: tuple[str, ...] = ()

    def to_serialized(self) -> str:
        """Serialize through the production ContextSerializer path.

        Uses training.bridge.serialize_scenario() which builds a real
        ContextBundle and serializes it with ContextSerializer.serialize()
        — the exact same code path as production inference.
        """
        from training.bridge import serialize_scenario

        return serialize_scenario(self)


_EVENT_FAMILIES: tuple[dict[str, object], ...] = (
    {
        "type": "input_price_change",
        "subject_type": "commodity",
        "magnitude_unit": "percent",
        "directions": {
            "increase": (
                "{commodity} prices surged by {mag}%",
                "{commodity} costs increased sharply by {mag}%",
                "{commodity} prices rose by {mag}% amid geopolitical tensions",
                "{commodity} prices jumped by {mag}% after supply disruptions",
            ),
            "decrease": (
                "{commodity} prices dropped by {mag}%",
                "{commodity} costs fell by {mag}% due to weak demand",
                "{commodity} prices declined by {mag}% following overproduction",
                "{commodity} prices collapsed by {mag}%",
            ),
        },
        "commodities": ("oil", "copper", "lithium", "steel", "natural gas", "cotton", "wheat"),
    },
    {
        "type": "demand_change",
        "subject_type": "product",
        "magnitude_unit": "percent",
        "directions": {
            "increase": (
                "Demand for {commodity} increased substantially by {mag}%",
                "Market demand for {commodity} rose sharply by {mag}%",
                "Orders for {commodity} surged by {mag}%",
            ),
            "decrease": (
                "Demand for {commodity} dropped by {mag}%",
                "Orders for {commodity} declined by {mag}%",
                "Market demand for {commodity} fell by {mag}%",
            ),
        },
        "commodities": ("electric vehicles", "semiconductors", "solar panels", "steel", "consumer electronics"),
    },
    {
        "type": "interest_rate_change",
        "subject_type": "macro",
        "magnitude_unit": "basis_points",
        "directions": {
            "hike": (
                "The central bank raised interest rates by {mag} basis points",
                "Interest rates increased by {mag} basis points",
            ),
            "cut": (
                "The central bank cut interest rates by {mag} basis points",
                "Interest rates were reduced by {mag} basis points",
            ),
        },
        "commodities": ("interest_rate",),
    },
    {
        "type": "customer_loss",
        "subject_type": "customer",
        "magnitude_unit": "none",
        "directions": {
            "loss": (
                "A major customer terminated its contract immediately",
                "The largest client cancelled all future orders",
                "A key customer announced it will stop purchasing",
            ),
        },
        "commodities": ("major_customer",),
    },
    {
        "type": "export_restriction",
        "subject_type": "product",
        "magnitude_unit": "percent",
        "directions": {
            "restriction": (
                "China imposed export restrictions on {commodity} affecting {mag}% of global supply",
                "New export controls limit sales of {commodity} covering {mag}% of the market",
                "Trade sanctions prohibit exports of {commodity} representing {mag}% of global trade",
            ),
        },
        "commodities": ("semiconductor chips", "rare earth materials", "critical components", "technology products"),
    },
)

_ROLE_BY_EVENT: dict[str, tuple[str, ...]] = {
    "input_price_change": ("consumer", "producer", "neutral"),
    "demand_change": ("producer", "neutral"),
    "interest_rate_change": ("bank", "real_estate"),
    "customer_loss": ("supplier_high", "supplier_low"),
    "export_restriction": ("consumer", "competitor", "neutral"),
}

_CASE_TYPE_BY_ROLE: dict[str, str] = {
    "consumer": "company",
    "producer": "company",
    "competitor": "company",
    "neutral": "company",
    "bank": "bank",
    "real_estate": "real_estate",
    "supplier_high": "company",
    "supplier_low": "company",
}

_ROLE_LABEL: dict[str, str] = {
    "consumer": "consumer",
    "producer": "producer",
    "competitor": "competitor",
    "neutral": "neutral",
    "bank": "bank",
    "real_estate": "real_estate",
    "supplier_high": "supplier_high",
    "supplier_low": "supplier_low",
}


def _build_context(
    event_type: str,
    role: str,
    commodity: str,
    exposure: float,
    hedging: float,
    pricing_power: float,
    substitution: float,
) -> str:
    """Build context text that contains ALL label-influencing variables.

    Context is built by (event_type, role) so that only variables that
    actually influence the score appear — no irrelevant features.
    """
    if role == "neutral":
        return (
            f"role=neutral; {commodity}_exposure=0%; "
            f"the company has no significant {commodity} exposure"
        )

    parts = [
        f"role={_ROLE_LABEL[role]}",
        f"{commodity or 'input'}_exposure={exposure:g}%",
    ]

    if event_type == "input_price_change":
        if role == "consumer":
            parts.append(f"hedging_ratio={hedging*100:.0f}%")
            parts.append(f"pricing_power={pricing_power*100:.0f}%")
            parts.append(f"substitution_ability={substitution*100:.0f}%")
        elif role == "producer":
            parts.append(f"hedging_ratio={hedging*100:.0f}%")
    elif event_type == "demand_change":
        if role == "producer":
            parts.append(f"pricing_power={pricing_power*100:.0f}%")
    elif event_type == "interest_rate_change":
        if role == "bank":
            parts.append(f"rate_sensitivity={pricing_power*100:.0f}%")
        elif role == "real_estate":
            parts.append(f"debt_ratio={exposure/100:.2f}")
            parts.append(f"rate_sensitivity={pricing_power*100:.0f}%")
    elif event_type == "customer_loss":
        if role in ("supplier_high", "supplier_low"):
            parts.append(f"replacement_difficulty={substitution*100:.0f}%")
    elif event_type == "export_restriction":
        if role == "consumer":
            parts.append(f"substitution_ability={substitution*100:.0f}%")
        elif role == "competitor":
            parts.append(f"market_share_gain_potential={pricing_power*100:.0f}%")
            parts.append(f"substitution_barrier={substitution*100:.0f}%")

    return "; ".join(parts)


def _compute_score(
    event_type: str,
    direction: str,
    role: str,
    magnitude: float,
    exposure: float,
    hedging: float,
    pricing_power: float,
    substitution: float,
    horizon_days: int,
) -> float:
    """Compute an economically grounded score in [-1, +1].

    Every variable here is explicitly in the input text.
    """
    if role == "neutral":
        return 0.0

    mag_norm = min(1.0, magnitude / 40.0)
    exp_norm = min(1.0, exposure / 40.0)
    hedge_factor = 1.0 - hedging * 0.6
    power_factor = 1.0 - pricing_power * 0.3
    sub_factor = 1.0 - substitution * 0.3

    horizon_decay = 0.4 + 0.6 * math.exp(-horizon_days / 365.0)

    if event_type == "input_price_change":
        if role == "consumer":
            if direction == "increase":
                raw = -mag_norm * exp_norm * hedge_factor * power_factor * sub_factor
            else:
                raw = mag_norm * exp_norm * hedge_factor * power_factor * sub_factor
        elif role == "producer":
            if direction == "increase":
                raw = mag_norm * exp_norm * hedge_factor
            else:
                raw = -mag_norm * exp_norm * hedge_factor
        else:
            raw = 0.0
    elif event_type == "demand_change":
        if role == "producer":
            if direction == "increase":
                raw = mag_norm * exp_norm * power_factor
            else:
                raw = -mag_norm * exp_norm * power_factor
        else:
            raw = 0.0
    elif event_type == "interest_rate_change":
        if role == "bank":
            raw = mag_norm * 0.8 if direction == "hike" else -mag_norm * 0.8
        elif role == "real_estate":
            raw = -mag_norm * exp_norm * 0.9 if direction == "hike" else mag_norm * exp_norm * 0.9
        else:
            raw = 0.0
    elif event_type == "customer_loss":
        if role == "supplier_high":
            raw = -exp_norm * 1.2 * sub_factor
        elif role == "supplier_low":
            raw = -exp_norm * 0.4 * sub_factor
        else:
            raw = 0.0
    elif event_type == "export_restriction":
        if role == "consumer":
            raw = -mag_norm * exp_norm * sub_factor
        elif role == "competitor":
            raw = mag_norm * 0.7
        else:
            raw = 0.0
    else:
        raw = 0.0

    score = raw * horizon_decay
    return max(-1.0, min(1.0, score))


def generate_scenario(rng: random.Random) -> EconomicScenario:
    """Generate one random parametric scenario.

    Text and label are guaranteed consistent: the template direction
    matches the internal direction used for scoring.
    """
    family = rng.choice(_EVENT_FAMILIES)
    event_type = family["type"]
    subject_type = family.get("subject_type", "commodity")
    magnitude_unit = family.get("magnitude_unit", "percent")
    directions = family["directions"]
    direction = rng.choice(tuple(directions.keys()))
    templates = directions[direction]
    template = rng.choice(templates)
    commodity = rng.choice(family["commodities"])

    if event_type == "customer_loss":
        magnitude = 0.0
    else:
        magnitude = round(rng.uniform(5, 80), 1)
    event_text = template.format(commodity=commodity, mag=f"{magnitude:g}")

    subject = f"{subject_type}:{commodity}" if commodity else subject_type

    roles = _ROLE_BY_EVENT.get(event_type, ("neutral",))
    role = rng.choice(roles)

    if event_type == "customer_loss":
        if role == "supplier_high":
            exposure = float(rng.choice([15, 20, 27, 35, 40, 50]))
        else:
            exposure = float(rng.choice([2, 3, 5]))
    elif role in ("bank",):
        exposure = 50.0
    elif role == "real_estate":
        exposure = float(rng.choice([30, 40, 50, 60, 70]))
    elif role == "neutral":
        exposure = 0.0
    else:
        exposure = float(rng.choice([5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80]))

    hedging = round(rng.uniform(0, 0.8), 2)
    pricing_power = round(rng.uniform(0, 1.0), 2)
    substitution = round(rng.uniform(0, 1.0), 2)
    horizon = rng.choice([7, 14, 30, 60, 90, 180, 365])

    context_text = _build_context(
        event_type=event_type,
        role=role,
        commodity=commodity,
        exposure=exposure,
        hedging=hedging,
        pricing_power=pricing_power,
        substitution=substitution,
    )

    score = _compute_score(
        event_type=event_type,
        direction=direction,
        role=role,
        magnitude=magnitude,
        exposure=exposure,
        hedging=hedging,
        pricing_power=pricing_power,
        substitution=substitution,
        horizon_days=horizon,
    )

    case_type = _CASE_TYPE_BY_ROLE.get(role, "company")

    return EconomicScenario(
        event_type=event_type,
        direction=direction,
        event_text=event_text,
        case_role=role,
        case_type=case_type,
        context_text=context_text,
        magnitude=magnitude,
        exposure=exposure,
        hedging=hedging,
        pricing_power=pricing_power,
        substitution=substitution,
        horizon_days=horizon,
        score=score,
        subject=subject,
        magnitude_unit=magnitude_unit if magnitude > 0 else "none",
    )


def generate_parametric(
    num: int = 5000,
    seed: int = 42,
) -> list[EconomicScenario]:
    """Generate num parametric scenarios."""
    rng = random.Random(seed)
    return [generate_scenario(rng) for _ in range(num)]


def generate_counterexample_groups(
    num_groups: int = 100,
    seed: int = 42,
) -> list[EconomicScenario]:
    """Generate controlled counterexamples.

    Each group changes EXACTLY ONE variable while holding everything else
    constant. This forces the model to learn the partial derivative of each
    variable independently.

    Group types:
        1. role groups: same event, same magnitude, same horizon, different role
        2. exposure groups: same everything, only exposure changes
        3. hedging groups: same everything, only hedging changes
        4. pricing_power groups: same everything, only pricing_power changes
        5. horizon groups: same everything, only horizon changes
    """
    rng = random.Random(seed)
    scenarios: list[EconomicScenario] = []

    for _ in range(num_groups):
        family = rng.choice(_EVENT_FAMILIES)
        event_type = family["type"]
        subject_type = family.get("subject_type", "commodity")
        magnitude_unit = family.get("magnitude_unit", "percent")
        directions = family["directions"]
        direction = rng.choice(tuple(directions.keys()))
        templates = directions[direction]
        template = rng.choice(templates)
        commodity = rng.choice(family["commodities"])
        magnitude = round(rng.uniform(20, 60), 1)
        event_text = template.format(commodity=commodity, mag=f"{magnitude:g}")
        subject = f"{subject_type}:{commodity}" if commodity else subject_type
        horizon = rng.choice([30, 90])

        roles = _ROLE_BY_EVENT.get(event_type, ("neutral",))

        # Group 1: role changes (same event, same magnitude, same horizon)
        base_exposure = float(rng.choice([20, 35, 50]))
        base_hedging = round(rng.uniform(0.2, 0.4), 2)
        base_pricing = round(rng.uniform(0.3, 0.6), 2)
        base_sub = round(rng.uniform(0.2, 0.5), 2)

        for role in roles:
            exposure = 0.0 if role == "neutral" else base_exposure
            context_text = _build_context(
                event_type=event_type, role=role, commodity=commodity,
                exposure=exposure, hedging=base_hedging,
                pricing_power=base_pricing, substitution=base_sub,
            )
            score = _compute_score(
                event_type=event_type, direction=direction, role=role,
                magnitude=magnitude, exposure=exposure,
                hedging=base_hedging, pricing_power=base_pricing,
                substitution=base_sub, horizon_days=horizon,
            )
            case_type = _CASE_TYPE_BY_ROLE.get(role, "company")
            scenarios.append(_make_scenario(
                event_type, direction, event_text, role, case_type,
                context_text, magnitude, exposure, base_hedging,
                base_pricing, base_sub, horizon, score,
                subject=subject, magnitude_unit=magnitude_unit,
            ))

        # Group 2-5: single-variable changes (only for non-neutral roles)
        active_roles = [r for r in roles if r != "neutral"]
        if active_roles:
            base_role = rng.choice(active_roles)
            exposure = base_exposure
            for var_name, var_values, var_key in [
                ("exposure", [10, 30, 50, 70], "exposure"),
                ("hedging", [0.0, 0.3, 0.6], "hedging"),
                ("pricing_power", [0.1, 0.5, 0.9], "pricing_power"),
                ("horizon", [7, 30, 90, 365], "horizon"),
            ]:
                for val in var_values:
                    e = val if var_key == "exposure" else exposure
                    h = val if var_key == "hedging" else base_hedging
                    p = val if var_key == "pricing_power" else base_pricing
                    s = base_sub
                    hd = int(val) if var_key == "horizon" else horizon
                    context_text = _build_context(
                        event_type=event_type, role=base_role, commodity=commodity,
                        exposure=e, hedging=h, pricing_power=p, substitution=s,
                    )
                    score = _compute_score(
                        event_type=event_type, direction=direction, role=base_role,
                        magnitude=magnitude, exposure=e, hedging=h,
                        pricing_power=p, substitution=s, horizon_days=hd,
                    )
                    case_type = _CASE_TYPE_BY_ROLE.get(base_role, "company")
                    scenarios.append(_make_scenario(
                        event_type, direction, event_text, base_role, case_type,
                        context_text, magnitude, e, h, p, s, hd, score,
                        subject=subject, magnitude_unit=magnitude_unit,
                    ))

    rng.shuffle(scenarios)
    return scenarios


def _make_scenario(
    event_type: str, direction: str, event_text: str, role: str, case_type: str,
    context_text: str, magnitude: float, exposure: float, hedging: float,
    pricing_power: float, substitution: float, horizon_days: int, score: float,
    subject: str = "", magnitude_unit: str = "percent",
) -> EconomicScenario:
    return EconomicScenario(
        event_type=event_type, direction=direction, event_text=event_text,
        case_role=role, case_type=case_type, context_text=context_text,
        magnitude=magnitude, exposure=exposure, hedging=hedging,
        pricing_power=pricing_power, substitution=substitution,
        horizon_days=horizon_days, score=score,
        subject=subject, magnitude_unit=magnitude_unit,
    )
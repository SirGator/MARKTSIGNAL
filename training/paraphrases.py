"""Paraphrase augmentation and summary neutralization.

For each scenario, generate multiple natural language variants of the
summary while keeping all structured fields identical. This teaches the
model to rely on structured fields (type, direction, magnitude) rather
than memorizing specific phrases.

Two strategies:
    1. Paraphrases: same economic concept, different wording, same label
    2. NO_SUMMARY: replace the summary with a neutral token, forcing the
       model to use structured fields only
"""

from __future__ import annotations

import random

from src.models.context_serializer import NO_SUMMARY

from .scenarios import EconomicScenario


_PARAPHRASES: dict[str, dict[str, tuple[str, ...]]] = {
    "input_price_change": {
        "increase": (
            "{commodity} prices surged by {mag}%",
            "{commodity} costs rose approximately {mag}%",
            "{commodity} prices climbed by {mag}%",
            "{commodity} became roughly {mag}% more expensive",
            "{commodity} rallied {mag}% on supply concerns",
            "{commodity} benchmarks advanced {mag}%",
            "{commodity} prices jumped {mag}% after supply disruptions",
            "{commodity} costs rose {mag}% quarter-over-quarter",
            "{commodity} prices increased by {mag}%",
            "{commodity} costs accelerated by {mag}%",
        ),
        "decrease": (
            "{commodity} prices dropped by {mag}%",
            "{commodity} costs fell approximately {mag}%",
            "{commodity} prices declined by {mag}%",
            "{commodity} retreated {mag}% on demand weakness",
            "{commodity} benchmarks fell {mag}%",
            "{commodity} prices collapsed by {mag}%",
            "{commodity} costs decreased by {mag}%",
            "{commodity} prices fell {mag}%",
        ),
    },
    "demand_change": {
        "increase": (
            "Demand for {commodity} increased by {mag}%",
            "Orders for {commodity} surged {mag}%",
            "Customer appetite for {commodity} climbed {mag}%",
            "Market demand for {commodity} rose {mag}%",
            "Order volumes for {commodity} grew {mag}% year-over-year",
        ),
        "decrease": (
            "Demand for {commodity} fell by {mag}%",
            "Orders for {commodity} declined {mag}%",
            "Customer appetite for {commodity} waned {mag}%",
            "Market demand for {commodity} dropped {mag}%",
            "Order volumes for {commodity} fell {mag}% year-over-year",
        ),
    },
    "interest_rate_change": {
        "hike": (
            "The central bank raised interest rates by {mag} basis points",
            "Interest rates increased by {mag} basis points",
            "Policymakers hiked rates by {mag} basis points",
            "The central bank tightened policy by {mag} basis points",
        ),
        "cut": (
            "The central bank cut interest rates by {mag} basis points",
            "Interest rates were reduced by {mag} basis points",
            "Policymakers eased rates by {mag} basis points",
            "The central bank delivered a {mag} basis point rate cut",
        ),
    },
    "customer_loss": {
        "loss": (
            "A major customer terminated its contract immediately",
            "The largest client cancelled all future orders",
            "A key customer announced it will stop purchasing",
            "A significant customer shifted to a competitor",
            "A material client discontinued its procurement agreement",
        ),
    },
    "export_restriction": {
        "restriction": (
            "China imposed export restrictions on {commodity} affecting {mag}% of global supply",
            "New export controls limit sales of {commodity} covering {mag}% of the market",
            "Trade sanctions prohibit exports of {commodity} representing {mag}% of global trade",
            "New trade barriers limit {commodity} exports covering {mag}% of supply",
            "Sanctions removed {mag}% of {commodity} capacity from the market",
        ),
    },
}

def generate_paraphrase(
    scenario: EconomicScenario,
    rng: random.Random,
) -> str:
    """Generate one alternative summary for a scenario."""
    event_paraphrases = _PARAPHRASES.get(scenario.event_type, {})
    direction_paraphrases = event_paraphrases.get(scenario.direction, ())
    if not direction_paraphrases:
        return scenario.event_text

    template = rng.choice(direction_paraphrases)
    commodity = scenario.subject.split(":", 1)[-1] if ":" in scenario.subject else ""
    return template.format(
        commodity=commodity,
        mag=f"{scenario.magnitude:g}",
    )


def expand_with_paraphrases(
    scenarios: list[EconomicScenario],
    num_paraphrases: int = 3,
    neutralize_ratio: float = 0.25,
    seed: int = 42,
) -> list[EconomicScenario]:
    """Expand each scenario with paraphrases and NO_SUMMARY variants.

    Args:
        scenarios: base scenarios
        num_paraphrases: how many alternative summaries per scenario
        neutralize_ratio: fraction of scenarios where summary is replaced
                         with [NO_SUMMARY]
        seed: RNG seed

    Returns:
        Expanded list including originals, paraphrases, and neutralized.
    """
    rng = random.Random(seed)
    expanded: list[EconomicScenario] = []

    for scenario in scenarios:
        expanded.append(scenario)

        for _ in range(num_paraphrases):
            alt_text = generate_paraphrase(scenario, rng)
            expanded.append(_clone_with_text(scenario, alt_text))

        if rng.random() < neutralize_ratio:
            expanded.append(_clone_with_text(scenario, NO_SUMMARY))

    rng.shuffle(expanded)
    return expanded


def _clone_with_text(scenario: EconomicScenario, new_text: str) -> EconomicScenario:
    """Clone a scenario with a different event_text but identical everything else."""
    return EconomicScenario(
        event_type=scenario.event_type,
        direction=scenario.direction,
        event_text=new_text,
        case_role=scenario.case_role,
        case_type=scenario.case_type,
        context_text=scenario.context_text,
        magnitude=scenario.magnitude,
        exposure=scenario.exposure,
        hedging=scenario.hedging,
        pricing_power=scenario.pricing_power,
        substitution=scenario.substitution,
        horizon_days=scenario.horizon_days,
        score=scenario.score,
        subject=scenario.subject,
        magnitude_unit=scenario.magnitude_unit,
        paraphrases=scenario.paraphrases,
    )

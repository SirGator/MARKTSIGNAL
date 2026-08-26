"""Out-of-distribution test data for EconomyEncoder V1.

OOD_LANGUAGE: ONLY the summary changes. All structured fields (type, subject,
direction, magnitude, role, exposure, etc.) stay identical to a training-like
scenario. This isolates language generalization from concept generalization.

The summaries use genuinely different vocabulary NOT present in training:
    Training: "Oil prices surged by 30%"
    OOD:      "Spot crude quotations appreciated by roughly 30%"

OOD_COMBINATION: unusual parameter combinations with normal language.
HARD_OOD: genuinely new event types with own score functions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .scenarios import EconomicScenario, _compute_score, _build_context


@dataclass(frozen=True, slots=True)
class OODTest:
    """One out-of-distribution test case."""
    category: str
    scenario: EconomicScenario
    description: str


def _score_regulatory_fine(magnitude, exposure, pricing_power, horizon_days):
    mag_norm = min(1.0, magnitude / 40.0)
    exp_norm = min(1.0, exposure / 40.0)
    power_factor = 1.0 - pricing_power * 0.4
    horizon_decay = 0.4 + 0.6 * math.exp(-horizon_days / 365.0)
    raw = -mag_norm * exp_norm * power_factor
    return max(-1.0, min(1.0, raw * horizon_decay))


def _score_supply_disruption(magnitude, exposure, substitution, horizon_days):
    mag_norm = min(1.0, magnitude / 100.0)
    exp_norm = min(1.0, exposure / 50.0)
    sub_factor = 1.0 - substitution * 0.5
    horizon_decay = 0.5 + 0.5 * math.exp(-horizon_days / 365.0)
    raw = -mag_norm * exp_norm * sub_factor
    return max(-1.0, min(1.0, raw * horizon_decay))


def _build_new_context(event_type, role, magnitude, exposure, pricing_power=0.0, substitution=0.0, horizon_days=30):
    parts = [f"role={role}", f"magnitude={magnitude:g}%", f"exposure={exposure:g}%"]
    if event_type == "regulatory_fine":
        parts.append(f"pricing_power={pricing_power*100:.0f}%")
    elif event_type == "supply_disruption":
        parts.append(f"substitution_ability={substitution*100:.0f}%")
    parts.append(f"horizon={horizon_days}d")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# OOD_LANGUAGE: ONLY summary changes, all structured fields stay identical.
# Summaries use vocabulary NOT present in training templates.
# ---------------------------------------------------------------------------

_OOD_LANGUAGE_TESTS: tuple[OODTest, ...] = (
    OODTest(
        category="OOD_LANGUAGE",
        description="oil consumer — financial press jargon",
        scenario=EconomicScenario(
            event_type="input_price_change", direction="increase",
            event_text="Spot crude quotations appreciated by roughly 30% on supply fears",
            case_role="consumer", case_type="airline",
            context_text=_build_context("input_price_change", "consumer", "oil", 35, 0.1, 0.2, 0.3),
            magnitude=30, exposure=35, hedging=0.1, pricing_power=0.2, substitution=0.3,
            horizon_days=30,
            score=_compute_score("input_price_change", "increase", "consumer", 30, 35, 0.1, 0.2, 0.3, 30),
            subject="commodity:oil", magnitude_unit="percent",
        ),
    ),
    OODTest(
        category="OOD_LANGUAGE",
        description="oil producer — trader slang",
        scenario=EconomicScenario(
            event_type="input_price_change", direction="increase",
            event_text="Brent benchmarks firmed notably amid OPEC output curbs",
            case_role="producer", case_type="oil_producer",
            context_text=_build_context("input_price_change", "producer", "oil", 80, 0.0, 0.5, 0.5),
            magnitude=30, exposure=80, hedging=0.0, pricing_power=0.5, substitution=0.5,
            horizon_days=30,
            score=_compute_score("input_price_change", "increase", "producer", 30, 80, 0.0, 0.5, 0.5, 30),
            subject="commodity:oil", magnitude_unit="percent",
        ),
    ),
    OODTest(
        category="OOD_LANGUAGE",
        description="copper consumer — industry report style",
        scenario=EconomicScenario(
            event_type="input_price_change", direction="increase",
            event_text="Copper spot values moved sharply higher by approximately 35%",
            case_role="consumer", case_type="company",
            context_text=_build_context("input_price_change", "consumer", "copper", 40, 0.15, 0.3, 0.2),
            magnitude=35, exposure=40, hedging=0.15, pricing_power=0.3, substitution=0.2,
            horizon_days=60,
            score=_compute_score("input_price_change", "increase", "consumer", 35, 40, 0.15, 0.3, 0.2, 60),
            subject="commodity:copper", magnitude_unit="percent",
        ),
    ),
    OODTest(
        category="OOD_LANGUAGE",
        description="oil producer — price decrease, energy market language",
        scenario=EconomicScenario(
            event_type="input_price_change", direction="decrease",
            event_text="Crude benchmarks retreated 25% on demand weakness",
            case_role="producer", case_type="oil_producer",
            context_text=_build_context("input_price_change", "producer", "oil", 70, 0.2, 0.3, 0.0),
            magnitude=25, exposure=70, hedging=0.2, pricing_power=0.3, substitution=0.0,
            horizon_days=60,
            score=_compute_score("input_price_change", "decrease", "producer", 25, 70, 0.2, 0.3, 0.0, 60),
            subject="commodity:oil", magnitude_unit="percent",
        ),
    ),
    OODTest(
        category="OOD_LANGUAGE",
        description="semiconductor producer — demand drop, analyst note style",
        scenario=EconomicScenario(
            event_type="demand_change", direction="decrease",
            event_text="End-market appetite for semiconductors waned considerably by 40%",
            case_role="producer", case_type="company",
            context_text=_build_context("demand_change", "producer", "semiconductors", 60, 0.0, 0.45, 0.0),
            magnitude=40, exposure=60, hedging=0.0, pricing_power=0.45, substitution=0.0,
            horizon_days=90,
            score=_compute_score("demand_change", "decrease", "producer", 40, 60, 0.0, 0.45, 0.0, 90),
            subject="product:semiconductors", magnitude_unit="percent",
        ),
    ),
    OODTest(
        category="OOD_LANGUAGE",
        description="electric vehicles producer — demand increase, industry language",
        scenario=EconomicScenario(
            event_type="demand_change", direction="increase",
            event_text="Order intake for electric vehicles accelerated meaningfully by 55%",
            case_role="producer", case_type="company",
            context_text=_build_context("demand_change", "producer", "electric vehicles", 70, 0.0, 0.6, 0.0),
            magnitude=55, exposure=70, hedging=0.0, pricing_power=0.6, substitution=0.0,
            horizon_days=60,
            score=_compute_score("demand_change", "increase", "producer", 55, 70, 0.0, 0.6, 0.0, 60),
            subject="product:electric vehicles", magnitude_unit="percent",
        ),
    ),
    OODTest(
        category="OOD_LANGUAGE",
        description="bank — rate hike, dovish/hawkish policy language",
        scenario=EconomicScenario(
            event_type="interest_rate_change", direction="hike",
            event_text="The monetary policy committee opted for a 50 basis point tightening",
            case_role="bank", case_type="bank",
            context_text=_build_context("interest_rate_change", "bank", "", 50, 0.0, 0.7, 0.0),
            magnitude=50, exposure=50, hedging=0.0, pricing_power=0.7, substitution=0.0,
            horizon_days=90,
            score=_compute_score("interest_rate_change", "hike", "bank", 50, 50, 0.0, 0.7, 0.0, 90),
            subject="macro:interest_rate", magnitude_unit="basis_points",
        ),
    ),
    OODTest(
        category="OOD_LANGUAGE",
        description="real estate — rate cut, dovish policy language",
        scenario=EconomicScenario(
            event_type="interest_rate_change", direction="cut",
            event_text="Policymakers opted to ease the policy rate by 75 basis points",
            case_role="real_estate", case_type="real_estate",
            context_text=_build_context("interest_rate_change", "real_estate", "", 60, 0.0, 0.4, 0.0),
            magnitude=75, exposure=60, hedging=0.0, pricing_power=0.4, substitution=0.0,
            horizon_days=90,
            score=_compute_score("interest_rate_change", "cut", "real_estate", 75, 60, 0.0, 0.4, 0.0, 90),
            subject="macro:interest_rate", magnitude_unit="basis_points",
        ),
    ),
    OODTest(
        category="OOD_LANGUAGE",
        description="customer loss — legal/formal procurement language",
        scenario=EconomicScenario(
            event_type="customer_loss", direction="loss",
            event_text="A material client elected to discontinue procurement effective immediately",
            case_role="supplier_high", case_type="company",
            context_text=_build_context("customer_loss", "supplier_high", "", 27, 0.0, 0.0, 0.8),
            magnitude=0, exposure=27, hedging=0.0, pricing_power=0.0, substitution=0.8,
            horizon_days=30,
            score=_compute_score("customer_loss", "loss", "supplier_high", 0, 27, 0.0, 0.0, 0.8, 30),
            subject="customer:major_customer", magnitude_unit="none",
        ),
    ),
    OODTest(
        category="OOD_LANGUAGE",
        description="customer loss — minimal exposure, formal",
        scenario=EconomicScenario(
            event_type="customer_loss", direction="loss",
            event_text="A small client terminated its purchasing agreement",
            case_role="supplier_low", case_type="company",
            context_text=_build_context("customer_loss", "supplier_low", "", 3, 0.0, 0.0, 0.2),
            magnitude=0, exposure=3, hedging=0.0, pricing_power=0.0, substitution=0.2,
            horizon_days=30,
            score=_compute_score("customer_loss", "loss", "supplier_low", 0, 3, 0.0, 0.0, 0.2, 30),
            subject="customer:major_customer", magnitude_unit="none",
        ),
    ),
    OODTest(
        category="OOD_LANGUAGE",
        description="export restriction — trade policy language",
        scenario=EconomicScenario(
            event_type="export_restriction", direction="restriction",
            event_text="New trade barriers curtail chip exports covering 45% of global supply",
            case_role="consumer", case_type="company",
            context_text=_build_context("export_restriction", "consumer", "semiconductor chips", 35, 0.0, 0.0, 0.3),
            magnitude=45, exposure=35, hedging=0.0, pricing_power=0.0, substitution=0.3,
            horizon_days=90,
            score=_compute_score("export_restriction", "restriction", "consumer", 45, 35, 0.0, 0.0, 0.3, 90),
            subject="product:semiconductor chips", magnitude_unit="percent",
        ),
    ),
    OODTest(
        category="OOD_LANGUAGE",
        description="export restriction — competitor benefits",
        scenario=EconomicScenario(
            event_type="export_restriction", direction="restriction",
            event_text="Sanctions eliminated 50% of rival capacity from the market",
            case_role="competitor", case_type="company",
            context_text=_build_context("export_restriction", "competitor", "critical components", 40, 0.0, 0.6, 0.3),
            magnitude=50, exposure=40, hedging=0.0, pricing_power=0.6, substitution=0.3,
            horizon_days=90,
            score=_compute_score("export_restriction", "restriction", "competitor", 50, 40, 0.0, 0.6, 0.3, 90),
            subject="product:critical components", magnitude_unit="percent",
        ),
    ),
)


_OOD_COMBINATION_TESTS: tuple[OODTest, ...] = (
    OODTest(
        category="OOD_COMBINATION",
        description="high hedging + high pricing power (rare combo for consumer)",
        scenario=EconomicScenario(
            event_type="input_price_change", direction="increase",
            event_text="Oil prices surged by 45% after supply disruptions",
            case_role="consumer", case_type="company",
            context_text=_build_context("input_price_change", "consumer", "oil", 50, 0.75, 0.8, 0.6),
            magnitude=45, exposure=50, hedging=0.75, pricing_power=0.8, substitution=0.6,
            horizon_days=180,
            score=_compute_score("input_price_change", "increase", "consumer", 45, 50, 0.75, 0.8, 0.6, 180),
            subject="commodity:oil", magnitude_unit="percent",
        ),
    ),
    OODTest(
        category="OOD_COMBINATION",
        description="neutral despite high commodity exposure (other commodity)",
        scenario=EconomicScenario(
            event_type="input_price_change", direction="increase",
            event_text="Copper prices surged by 60%",
            case_role="neutral", case_type="software",
            context_text=_build_context("input_price_change", "neutral", "copper", 0, 0.0, 0.0, 0.0),
            magnitude=60, exposure=0, hedging=0.0, pricing_power=0.0, substitution=0.0,
            horizon_days=365, score=0.0,
            subject="commodity:copper", magnitude_unit="percent",
        ),
    ),
    OODTest(
        category="OOD_COMBINATION",
        description="very short horizon + high magnitude (acute shock)",
        scenario=EconomicScenario(
            event_type="input_price_change", direction="increase",
            event_text="Oil prices jumped 70% overnight",
            case_role="consumer", case_type="airline",
            context_text=_build_context("input_price_change", "consumer", "oil", 40, 0.05, 0.1, 0.1),
            magnitude=70, exposure=40, hedging=0.05, pricing_power=0.1, substitution=0.1,
            horizon_days=7,
            score=_compute_score("input_price_change", "increase", "consumer", 70, 40, 0.05, 0.1, 0.1, 7),
            subject="commodity:oil", magnitude_unit="percent",
        ),
    ),
    OODTest(
        category="OOD_COMBINATION",
        description="very long horizon + small magnitude (fade out)",
        scenario=EconomicScenario(
            event_type="input_price_change", direction="increase",
            event_text="Steel prices rose 10%",
            case_role="consumer", case_type="company",
            context_text=_build_context("input_price_change", "consumer", "steel", 30, 0.3, 0.4, 0.5),
            magnitude=10, exposure=30, hedging=0.3, pricing_power=0.4, substitution=0.5,
            horizon_days=365,
            score=_compute_score("input_price_change", "increase", "consumer", 10, 30, 0.3, 0.4, 0.5, 365),
            subject="commodity:steel", magnitude_unit="percent",
        ),
    ),
)


_HARD_OOD_TESTS: tuple[OODTest, ...] = (
    OODTest(
        category="HARD_OOD",
        description="regulatory fine — new event type",
        scenario=EconomicScenario(
            event_type="regulatory_fine", direction="penalty",
            event_text="Regulators imposed a compliance penalty of 25% of annual revenue",
            case_role="affected", case_type="company",
            context_text=_build_new_context("regulatory_fine", "affected", 25, 20, 0.2, 0, 90),
            magnitude=25, exposure=20, hedging=0.0, pricing_power=0.2, substitution=0.0,
            horizon_days=90, score=_score_regulatory_fine(25, 20, 0.2, 90),
            subject="regulatory:compliance", magnitude_unit="percent",
        ),
    ),
    OODTest(
        category="HARD_OOD",
        description="supply chain disruption — new event type",
        scenario=EconomicScenario(
            event_type="supply_disruption", direction="disruption",
            event_text="A fire at a key supplier halted 60% of component deliveries",
            case_role="affected", case_type="company",
            context_text=_build_new_context("supply_disruption", "affected", 60, 50, 0, 0.3, 30),
            magnitude=60, exposure=50, hedging=0.0, pricing_power=0.0, substitution=0.3,
            horizon_days=30, score=_score_supply_disruption(60, 50, 0.3, 30),
            subject="supplier:key_supplier", magnitude_unit="percent",
        ),
    ),
    OODTest(
        category="HARD_OOD",
        description="regulatory fine — large fine, high pricing power",
        scenario=EconomicScenario(
            event_type="regulatory_fine", direction="penalty",
            event_text="Antitrust authorities levied a fine equivalent to 35% of revenue",
            case_role="affected", case_type="company",
            context_text=_build_new_context("regulatory_fine", "affected", 35, 35, 0.6, 0, 180),
            magnitude=35, exposure=35, hedging=0.0, pricing_power=0.6, substitution=0.0,
            horizon_days=180, score=_score_regulatory_fine(35, 35, 0.6, 180),
            subject="regulatory:antitrust", magnitude_unit="percent",
        ),
    ),
    OODTest(
        category="HARD_OOD",
        description="supply disruption — high substitution ability",
        scenario=EconomicScenario(
            event_type="supply_disruption", direction="disruption",
            event_text="A logistics strike blocked 40% of raw material imports",
            case_role="affected", case_type="company",
            context_text=_build_new_context("supply_disruption", "affected", 40, 30, 0, 0.7, 60),
            magnitude=40, exposure=30, hedging=0.0, pricing_power=0.0, substitution=0.7,
            horizon_days=60, score=_score_supply_disruption(40, 30, 0.7, 60),
            subject="supplier:logistics", magnitude_unit="percent",
        ),
    ),
)


def all_ood_tests() -> tuple[OODTest, ...]:
    return _OOD_LANGUAGE_TESTS + _OOD_COMBINATION_TESTS + _HARD_OOD_TESTS


def ood_by_category(category: str) -> tuple[OODTest, ...]:
    return tuple(t for t in all_ood_tests() if t.category == category)
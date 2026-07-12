from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if math.isnan(result) or math.isinf(result):
        return default

    return result


@dataclass
class EnsembleComponent:
    name: str
    probability: float
    base_weight: float
    reliability: float
    uncertainty: float = 0.0

    @property
    def effective_weight(self) -> float:
        reliability_factor = 0.25 + 0.75 * clamp(
            self.reliability,
            0.0,
            1.0,
        )
        uncertainty_factor = 1.0 - 0.65 * clamp(
            self.uncertainty,
            0.0,
            1.0,
        )

        return max(
            0.0,
            self.base_weight
            * reliability_factor
            * uncertainty_factor,
        )


@dataclass
class FootballEnsembleV14Result:
    probability: float
    internal_probability: float
    market_probability: float

    market_weight: float
    internal_weight: float

    dispersion: float
    combined_reliability: float
    disagreement_penalty: float

    component_weights: dict[str, float]
    reason: str


def weighted_dispersion(
    values: list[tuple[float, float]],
    average: float,
) -> float:
    total_weight = sum(weight for _, weight in values)

    if total_weight <= 0:
        return 0.0

    variance = sum(
        weight * (value - average) ** 2
        for value, weight in values
    ) / total_weight

    return math.sqrt(max(0.0, variance))


def build_football_ensemble_v14(
    *,
    market_probability: float,
    elo_probability: float,
    form_probability: float,
    dixon_probability: float,

    elo_reliability: float,
    form_reliability: float,
    xg_reliability: float,

    market_overround: float,
    bookmaker_count: int,

    league_calibration_reliability: float = 0.0,
) -> FootballEnsembleV14Result:
    market_probability = clamp(
        safe_float(market_probability, 0.50),
        0.01,
        0.99,
    )

    components = [
        EnsembleComponent(
            name="dixon_coles",
            probability=clamp(
                safe_float(dixon_probability, market_probability),
                0.01,
                0.99,
            ),
            base_weight=0.42,
            reliability=clamp(
                0.45
                + xg_reliability * 0.35
                + league_calibration_reliability * 0.20,
                0.0,
                1.0,
            ),
            uncertainty=1.0 - clamp(
                xg_reliability,
                0.0,
                1.0,
            ),
        ),
        EnsembleComponent(
            name="team_elo",
            probability=clamp(
                safe_float(elo_probability, market_probability),
                0.01,
                0.99,
            ),
            base_weight=0.33,
            reliability=elo_reliability,
            uncertainty=1.0 - elo_reliability,
        ),
        EnsembleComponent(
            name="team_form",
            probability=clamp(
                safe_float(form_probability, market_probability),
                0.01,
                0.99,
            ),
            base_weight=0.25,
            reliability=form_reliability,
            uncertainty=1.0 - form_reliability,
        ),
    ]

    effective = [
        (
            component.probability,
            component.effective_weight,
            component.name,
        )
        for component in components
    ]

    total_internal_weight = sum(
        weight
        for _, weight, _ in effective
    )

    if total_internal_weight <= 0:
        internal_probability = market_probability
        normalized_weights = {
            component.name: 0.0
            for component in components
        }
    else:
        internal_probability = sum(
            probability * weight
            for probability, weight, _ in effective
        ) / total_internal_weight

        normalized_weights = {
            name: weight / total_internal_weight
            for _, weight, name in effective
        }

    dispersion = weighted_dispersion(
        [
            (probability, weight)
            for probability, weight, _ in effective
        ],
        internal_probability,
    )

    component_reliabilities = [
        xg_reliability,
        elo_reliability,
        form_reliability,
        league_calibration_reliability,
    ]
    combined_reliability = clamp(
        sum(component_reliabilities)
        / len(component_reliabilities),
        0.0,
        1.0,
    )

    disagreement_penalty = clamp(
        dispersion * 4.0,
        0.0,
        0.60,
    )

    bookmaker_depth = clamp(
        safe_float(bookmaker_count, 0.0) / 12.0,
        0.0,
        1.0,
    )
    market_quality = clamp(
        0.55
        + bookmaker_depth * 0.25
        - clamp(market_overround, 0.0, 0.30) * 1.20,
        0.20,
        0.90,
    )

    market_weight = clamp(
        0.18
        + (1.0 - combined_reliability) * 0.42
        + disagreement_penalty * 0.35
        + market_quality * 0.08,
        0.18,
        0.72,
    )
    internal_weight = 1.0 - market_weight

    final_probability = (
        internal_probability * internal_weight
        + market_probability * market_weight
    )

    # Extra guard against unstable extreme outputs.
    maximum_market_gap = (
        0.05
        + combined_reliability * 0.12
        - disagreement_penalty * 0.04
    )
    maximum_market_gap = clamp(
        maximum_market_gap,
        0.04,
        0.17,
    )

    final_probability = clamp(
        final_probability,
        market_probability - maximum_market_gap,
        market_probability + maximum_market_gap,
    )
    final_probability = clamp(
        final_probability,
        0.01,
        0.99,
    )

    reason = (
        f"v14 ensemble: final={final_probability:.4f}; "
        f"internal={internal_probability:.4f}; "
        f"market={market_probability:.4f}; "
        f"market_weight={market_weight:.3f}; "
        f"reliability={combined_reliability:.3f}; "
        f"dispersion={dispersion:.4f}; "
        f"disagreement_penalty={disagreement_penalty:.3f}; "
        f"weights={normalized_weights}"
    )

    return FootballEnsembleV14Result(
        probability=final_probability,
        internal_probability=internal_probability,
        market_probability=market_probability,
        market_weight=market_weight,
        internal_weight=internal_weight,
        dispersion=dispersion,
        combined_reliability=combined_reliability,
        disagreement_penalty=disagreement_penalty,
        component_weights=normalized_weights,
        reason=reason,
    )

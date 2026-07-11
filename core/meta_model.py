from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetaFeatures:
    market_probability: float
    elo_adjustment: float
    form_adjustment: float
    clv_adjustment: float
    bookmaker_grade: float
    sport_weight: float
    league_weight: float
    confidence: float
    monte_carlo_probability: float


def predict_probability(features: MetaFeatures) -> float:
    p = (
        features.market_probability
        + features.elo_adjustment
        + features.form_adjustment
        + features.clv_adjustment
    )

    p *= features.bookmaker_grade
    p *= features.sport_weight
    p *= features.league_weight

    p += (features.monte_carlo_probability - p) * 0.30

    return max(0.01, min(0.99, p))

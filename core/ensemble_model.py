from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class EnsembleInput:
    market_probability: float
    elo_adjustment: float = 0.0
    form_adjustment: float = 0.0
    clv_adjustment: float = 0.0
    bookmaker_adjustment: float = 0.0
    sport_adjustment: float = 0.0


@dataclass
class EnsembleOutput:
    probability: float
    edge: float
    score: float
    reason: str


def clamp(value: float, low: float = 0.01, high: float = 0.99) -> float:
    return max(low, min(high, value))


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def build_ensemble_probability(
    data: EnsembleInput,
    odds: float,
) -> EnsembleOutput:
    market_weight = env_float("ENSEMBLE_MARKET_WEIGHT", 0.55)
    elo_weight = env_float("ENSEMBLE_ELO_WEIGHT", 0.20)
    form_weight = env_float("ENSEMBLE_FORM_WEIGHT", 0.10)
    clv_weight = env_float("ENSEMBLE_CLV_WEIGHT", 0.05)
    bookmaker_weight = env_float("ENSEMBLE_BOOKMAKER_WEIGHT", 0.05)
    sport_weight = env_float("ENSEMBLE_SPORT_WEIGHT", 0.05)

    base = data.market_probability

    probability = (
        base * market_weight
        + clamp(base + data.elo_adjustment) * elo_weight
        + clamp(base + data.form_adjustment) * form_weight
        + clamp(base + data.clv_adjustment) * clv_weight
        + clamp(base + data.bookmaker_adjustment) * bookmaker_weight
        + clamp(base + data.sport_adjustment) * sport_weight
    )

    total_weight = (
        market_weight
        + elo_weight
        + form_weight
        + clv_weight
        + bookmaker_weight
        + sport_weight
    )

    if total_weight <= 0:
        probability = base
    else:
        probability = probability / total_weight

    probability = clamp(probability)

    edge = probability * odds - 1.0
    score = edge * 100

    reason = (
        "Ensemble: "
        f"market={market_weight:.2f}, "
        f"elo={elo_weight:.2f}, "
        f"form={form_weight:.2f}, "
        f"clv={clv_weight:.2f}, "
        f"bookmaker={bookmaker_weight:.2f}, "
        f"sport={sport_weight:.2f}"
    )

    return EnsembleOutput(
        probability=probability,
        edge=edge,
        score=score,
        reason=reason,
    )

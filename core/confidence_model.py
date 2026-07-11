from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConfidenceInput:
    edge: float
    consensus: float
    bayesian: float = 0.50
    clv: float = 0.0
    bookmaker_weight: float = 1.0
    sport_weight: float = 1.0
    league_weight: float = 1.0
    samples: int = 0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def confidence_score(data: ConfidenceInput) -> int:
    score = 50.0

    score += data.edge * 260
    score += data.consensus * 120

    score += (data.bayesian - 0.50) * 80

    score += data.clv * 40

    score += (data.bookmaker_weight - 1.0) * 120
    score += (data.sport_weight - 1.0) * 100
    score += (data.league_weight - 1.0) * 100

    if data.samples >= 200:
        score += 5
    elif data.samples >= 100:
        score += 3
    elif data.samples >= 50:
        score += 1

    return int(clamp(score, 1, 100))

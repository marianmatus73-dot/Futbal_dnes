from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConsensusInput:
    sport: str
    league: str
    match: str
    pick: str
    odds: float

    elo_probability: float | None = None
    xg_probability: float | None = None
    form_probability: float | None = None
    market_probability: float | None = None
    injury_penalty: float = 0.0
    news_penalty: float = 0.0


@dataclass
class ConsensusResult:
    sport: str
    league: str
    match: str
    pick: str
    odds: float
    model_probability: float
    confidence_boost: float
    reason: str


WEIGHTS = {
    "elo": 0.25,
    "xg": 0.25,
    "form": 0.20,
    "market": 0.20,
    "context": 0.10,
}


def clamp(value: float, low: float = 0.01, high: float = 0.99) -> float:
    return max(low, min(high, value))


def build_consensus(data: ConsensusInput) -> ConsensusResult:
    signals = []

    if data.elo_probability is not None:
        signals.append(("elo", data.elo_probability, WEIGHTS["elo"]))

    if data.xg_probability is not None:
        signals.append(("xg", data.xg_probability, WEIGHTS["xg"]))

    if data.form_probability is not None:
        signals.append(("form", data.form_probability, WEIGHTS["form"]))

    if data.market_probability is not None:
        signals.append(("market", data.market_probability, WEIGHTS["market"]))

    if not signals:
        probability = 1 / data.odds
        reason = "Fallback podľa trhovej pravdepodobnosti."
    else:
        total_weight = sum(weight for _, _, weight in signals)
        probability = sum(prob * weight for _, prob, weight in signals) / total_weight
        reason = "Consensus: " + ", ".join(name for name, _, _ in signals)

    penalty = data.injury_penalty + data.news_penalty
    probability = clamp(probability - penalty)

    confidence_boost = max(0.0, probability - (1 / data.odds))

    return ConsensusResult(
        sport=data.sport,
        league=data.league,
        match=data.match,
        pick=data.pick,
        odds=data.odds,
        model_probability=probability,
        confidence_boost=confidence_boost,
        reason=reason,
    )

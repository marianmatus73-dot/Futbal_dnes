from __future__ import annotations

from core.bayesian_update import bayesian_multiplier
from core.model_stats import load_model_stats


def _safe_int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _combined_weight(
    item: dict | None,
    *,
    min_samples: int,
    bayes_min: float,
    bayes_max: float,
    yield_scale: float,
) -> float:
    if not item:
        return 1.0

    wins = _safe_int(item.get("wins"))
    losses = _safe_int(item.get("losses"))
    total = _safe_int(item.get("total"))
    yield_pct = _safe_float(item.get("yield"))

    bayes_weight = bayesian_multiplier(
        wins=wins,
        losses=losses,
        min_weight=bayes_min,
        max_weight=bayes_max,
    )

    if total < min_samples:
        return bayes_weight

    yield_component = max(
        -0.05,
        min(0.05, yield_pct / yield_scale),
    )

    combined = bayes_weight + yield_component

    return round(
        max(bayes_min, min(bayes_max, combined)),
        4,
    )


def sport_weight(sport: str) -> float:
    stats = load_model_stats()
    item = stats.get("by_sport", {}).get(sport)

    return _combined_weight(
        item,
        min_samples=30,
        bayes_min=0.90,
        bayes_max=1.10,
        yield_scale=400.0,
    )


def bookmaker_weight(bookmaker: str) -> float:
    stats = load_model_stats()
    item = stats.get("by_bookmaker", {}).get(bookmaker)

    return _combined_weight(
        item,
        min_samples=25,
        bayes_min=0.95,
        bayes_max=1.05,
        yield_scale=500.0,
    )


def league_weight(league: str) -> float:
    stats = load_model_stats()
    item = stats.get("by_league", {}).get(league)

    return _combined_weight(
        item,
        min_samples=30,
        bayes_min=0.92,
        bayes_max=1.08,
        yield_scale=450.0,
    )

from __future__ import annotations

from core.model_stats import load_model_stats


def sport_weight(sport: str) -> float:
    stats = load_model_stats()

    by_sport = stats.get("by_sport", {})
    item = by_sport.get(sport)

    if not item:
        return 1.0

    y = float(item.get("yield", 0))

    if y > 20:
        return 1.20

    if y > 10:
        return 1.10

    if y < -10:
        return 0.90

    if y < -20:
        return 0.80

    return 1.0


def bookmaker_weight(bookmaker: str) -> float:
    stats = load_model_stats()

    by_bookmaker = stats.get("by_bookmaker", {})
    item = by_bookmaker.get(bookmaker)

    if not item:
        return 1.0

    y = float(item.get("yield", 0))

    if y > 30:
        return 1.15

    if y > 15:
        return 1.10

    if y < -15:
        return 0.90

    return 1.0


def league_weight(league: str) -> float:
    stats = load_model_stats()

    by_league = stats.get("by_league", {})
    item = by_league.get(league)

    if not item:
        return 1.0

    y = float(item.get("yield", 0))

    if y > 20:
        return 1.15

    if y > 10:
        return 1.05

    if y < -10:
        return 0.90

    return 1.0

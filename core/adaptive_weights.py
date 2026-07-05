from __future__ import annotations

import json
from pathlib import Path


MODEL_STATS = Path("exports/model_stats.json")
ADAPTIVE_WEIGHTS = Path("exports/adaptive_weights.json")


def build_adaptive_weights() -> dict:
    if not MODEL_STATS.exists():
        return {}

    data = json.loads(
        MODEL_STATS.read_text(encoding="utf-8")
    )

    result = {
        "sports": {},
        "bookmakers": {},
        "leagues": {},
    }

    for sport, stats in data.get("by_sport", {}).items():
        profit = float(stats.get("profit", 0))
        yield_pct = float(stats.get("yield", 0))

        weight = 1.0

        if yield_pct > 20:
            weight = 1.20
        elif yield_pct > 10:
            weight = 1.10
        elif yield_pct < -10:
            weight = 0.90
        elif yield_pct < -20:
            weight = 0.80

        result["sports"][sport] = round(weight, 3)

    for bookmaker, stats in data.get("by_bookmaker", {}).items():
        profit = float(stats.get("profit", 0))
        yield_pct = float(stats.get("yield", 0))
        bets = int(stats.get("total", 0))

        weight = 1.0

        if bets >= 5:
            if yield_pct > 20:
                weight = 1.20
            elif yield_pct > 10:
                weight = 1.10
            elif yield_pct < -10:
                weight = 0.90
            elif yield_pct < -20:
                weight = 0.80

        result["bookmakers"][bookmaker] = round(weight, 3)

    for league, stats in data.get("by_league", {}).items():
        yield_pct = float(stats.get("yield", 0))

        weight = 1.0

        if yield_pct > 20:
            weight = 1.15
        elif yield_pct > 10:
            weight = 1.05
        elif yield_pct < -10:
            weight = 0.95
        elif yield_pct < -20:
            weight = 0.85

        result["leagues"][league] = round(weight, 3)

    ADAPTIVE_WEIGHTS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ADAPTIVE_WEIGHTS.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    return result

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json


MODEL_STATS_FILE = Path("exports/model_stats.json")


def default_model_stats() -> dict:
    return {
        "total_bets": 0,
        "wins": 0,
        "losses": 0,
        "winrate": 0.0,
        "yield": 0.0,
        "profit": 0.0,
        "stake_sum": 0.0,
        "open_bets": 0,
        "settled_bets": 0,
        "by_sport": {},
        "by_bookmaker": {},
        "by_league": {},
        "last_update": "",
    }


def save_model_stats(
    total_bets: int,
    wins: int,
    losses: int,
    profit: float,
    yield_pct: float,
    stake_sum: float = 0.0,
    open_bets: int = 0,
    settled_bets: int = 0,
    by_sport: dict | None = None,
    by_bookmaker: dict | None = None,
    by_league: dict | None = None,
) -> None:
    MODEL_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "total_bets": total_bets,
        "wins": wins,
        "losses": losses,
        "winrate": round((wins / total_bets * 100) if total_bets else 0.0, 2),
        "yield": round(yield_pct, 2),
        "profit": round(profit, 2),
        "stake_sum": round(stake_sum, 2),
        "open_bets": open_bets,
        "settled_bets": settled_bets,
        "by_sport": by_sport or {},
        "by_bookmaker": by_bookmaker or {},
        "by_league": by_league or {},
        "last_update": datetime.now(timezone.utc).isoformat(),
    }

    MODEL_STATS_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_model_stats() -> dict:
    if not MODEL_STATS_FILE.exists():
        return default_model_stats()

    try:
        data = json.loads(MODEL_STATS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return default_model_stats()

    defaults = default_model_stats()
    defaults.update(data)
    return defaults

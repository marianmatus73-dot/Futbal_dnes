from __future__ import annotations

from pathlib import Path
import json

MODEL_STATS_FILE = Path("exports/model_stats.json")


def save_model_stats(
    total_bets: int,
    wins: int,
    losses: int,
    profit: float,
    yield_pct: float,
) -> None:
    MODEL_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "total_bets": total_bets,
        "wins": wins,
        "losses": losses,
        "winrate": round(
            (wins / total_bets * 100) if total_bets else 0,
            2,
        ),
        "yield": round(yield_pct, 2),
        "profit": round(profit, 2),
    }

    MODEL_STATS_FILE.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )


def load_model_stats() -> dict:
    if not MODEL_STATS_FILE.exists():
        return {
            "total_bets": 0,
            "wins": 0,
            "losses": 0,
            "winrate": 0.0,
            "yield": 0.0,
            "profit": 0.0,
        }

    return json.loads(
        MODEL_STATS_FILE.read_text(encoding="utf-8")
    )

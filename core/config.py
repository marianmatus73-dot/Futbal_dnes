from __future__ import annotations

import os
from dataclasses import dataclass


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class Settings:
    bank: float = 1000.0
    min_edge: float = 0.08
    max_edge: float = 0.15
    max_odds: float = 7.5
    max_stake_pct: float = 0.01
    kelly_frac: float = 0.015
    odds_api_key: str = ""
    dry_run: bool = False
    db_file: str = "bets.db"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            bank=env_float("AKTUALNY_BANK", 1000.0),
            min_edge=env_float("MIN_EDGE", 0.08),
            max_edge=env_float("MAX_EDGE", 0.15),
            max_odds=env_float("MAX_ODDS", 7.5),
            max_stake_pct=env_float("MAX_STAKE_PCT", 0.01),
            kelly_frac=env_float("KELLY_FRAC", 0.015),
            odds_api_key=os.getenv("ODDS_API_KEY", "").strip(),
            db_file=os.getenv("DB_FILE", "bets.db"),
        )

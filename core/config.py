from __future__ import annotations

import os
from dataclasses import dataclass


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
            bank=float(os.getenv("AKTUALNY_BANK", "1000")),
            min_edge=float(os.getenv("MIN_EDGE", "0.08")),
            max_edge=float(os.getenv("MAX_EDGE", "0.15")),
            max_odds=float(os.getenv("MAX_ODDS", "7.5")),
            max_stake_pct=float(os.getenv("MAX_STAKE_PCT", "0.01")),
            kelly_frac=float(os.getenv("KELLY_FRAC", "0.015")),
            odds_api_key=os.getenv("ODDS_API_KEY", "").strip(),
            db_file=os.getenv("DB_FILE", "bets.db"),
        )

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from core.config import Settings
from core.sport_quant import init_sport_db
from core.sport_settlement import ensure_settlement_columns
from core.types import Bet
from sports.hockey import HockeyModule


class HockeyEventIdentityTests(unittest.TestCase):
    def test_save_is_idempotent_and_persists_external_event_id(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            database = Path(directory) / "bets.db"
            settings = Settings(db_file=str(database))
            init_sport_db(settings)
            ensure_settlement_columns(settings)
            bet = Bet(
                sport="hockey",
                league="icehockey_nhl",
                event="Boston vs Toronto",
                market="h2h",
                selection="Boston",
                odds=1.90,
                prob_model=0.58,
                prob_market=0.55,
                prob_final=0.58,
                edge=0.102,
                stake=1.0,
                bookmaker="Book A",
                start_time="2026-10-01T18:00:00Z",
                external_event_id="nhl-event-123",
            )
            module = HockeyModule()
            module._save_bet(settings, bet)
            module._save_bet(settings, bet)
            module._save_bet(
                settings,
                replace(bet, selection="Toronto", odds=2.10),
            )
            with sqlite3.connect(database) as conn:
                rows = conn.execute(
                    "SELECT external_event_id, home_team, away_team, result "
                    "FROM sport_bets"
                ).fetchall()
            self.assertEqual(rows, [
                ("nhl-event-123", "Boston", "Toronto", "OPEN")
            ])


if __name__ == "__main__":
    unittest.main()


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
from sports.baseball import BaseballModule
from sports.basketball import BasketballModule
from sports.nfl import NFLModule


class MultisportEventIdentityTests(unittest.TestCase):
    def test_one_persisted_selection_per_event_and_market(self) -> None:
        for sport, league, module in (
            ("baseball", "baseball_mlb", BaseballModule()),
            ("basketball", "basketball_nba", BasketballModule()),
            ("nfl", "americanfootball_nfl", NFLModule()),
        ):
            with self.subTest(sport=sport), tempfile.TemporaryDirectory(
                ignore_cleanup_errors=True
            ) as directory:
                database = Path(directory) / "bets.db"
                settings = Settings(db_file=str(database))
                init_sport_db(settings)
                ensure_settlement_columns(settings)
                bet = Bet(
                    sport=sport,
                    league=league,
                    event="Home vs Away",
                    market="h2h",
                    selection="Home",
                    odds=1.90,
                    prob_model=0.58,
                    prob_market=0.55,
                    prob_final=0.58,
                    edge=0.102,
                    stake=1.0,
                    bookmaker="Book A",
                    start_time="2026-10-01T18:00:00Z",
                    external_event_id=f"{sport}-event-123",
                )
                module._save_bet(settings, bet)
                module._save_bet(
                    settings,
                    replace(bet, selection="Away", odds=2.10),
                )
                with sqlite3.connect(database) as conn:
                    rows = conn.execute(
                        "SELECT external_event_id, selection, result "
                        "FROM sport_bets"
                    ).fetchall()
                self.assertEqual(rows, [
                    (f"{sport}-event-123", "Home", "OPEN")
                ])


if __name__ == "__main__":
    unittest.main()


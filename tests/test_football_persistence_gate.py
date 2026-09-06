from __future__ import annotations

import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from core.config import Settings
from core.football_tip_release import ensure_release_columns
from core.sport_quant import init_sport_db
from core.sport_settlement import ensure_settlement_columns
from core.types import Bet
from sports.football import FootballModule


class FootballPersistenceGateTests(unittest.TestCase):
    def test_only_released_list_is_persisted_once_per_event_market(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            database = Path(directory) / "bets.db"
            settings = Settings(db_file=str(database))
            init_sport_db(settings)
            ensure_settlement_columns(settings)
            ensure_release_columns(settings)
            accepted = Bet(
                sport="football", league="Premier League",
                event="Arsenal vs Chelsea", market="h2h",
                selection="Arsenal", odds=1.80, prob_model=0.60,
                prob_market=0.56, prob_final=0.60, edge=0.08,
                stake=1.0, bookmaker="Book A",
                start_time="2026-09-10T18:00:00Z", score=72,
                external_event_id="football-event-1", release_stage="EARLY",
                opening_odds=1.80,
            )
            rejected = replace(
                accepted,
                external_event_id="football-event-2",
                event="Liverpool vs Everton",
            )
            module = FootballModule()
            self.assertEqual(module.persist_released_bets(settings, [accepted]), 1)
            self.assertEqual(module.persist_released_bets(
                settings, [replace(accepted, selection="Chelsea", odds=2.20)]
            ), 0)
            with sqlite3.connect(database) as conn:
                rows = conn.execute(
                    "SELECT external_event_id, selection, release_stage, "
                    "opening_odds FROM sport_bets"
                ).fetchall()
            self.assertEqual(rows, [
                ("football-event-1", "Arsenal", "EARLY", 1.8)
            ])
            self.assertNotEqual(rejected.external_event_id, rows[0][0])


if __name__ == "__main__":
    unittest.main()


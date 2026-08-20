from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.config import Settings
from core.football_tip_release import apply_football_release_policy
from core.sport_context import SportContextDatabase
from core.types import Bet, SportResult


class FootballTipReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db = Path(self.temp.name) / "bets.db"
        self.settings = Settings(db_file=str(self.db))
        with sqlite3.connect(self.db) as conn:
            conn.execute(
                """
                CREATE TABLE sport_bets (
                    id INTEGER PRIMARY KEY, sport TEXT, league TEXT, event TEXT,
                    selection TEXT, start_time TEXT, odds REAL
                )
                """
            )
        SportContextDatabase(self.settings).init_db()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _bet(self, start: datetime, event: str) -> Bet:
        bet = Bet(
            sport="football", league="Test", event=event, market="h2h",
            selection="A", odds=2.0, prob_model=.6, prob_market=.5,
            prob_final=.6, edge=.2, stake=1.0, bookmaker="Book",
            start_time=start.isoformat(), external_event_id=event,
        )
        with sqlite3.connect(self.db) as conn:
            conn.execute(
                "INSERT INTO sport_bets (sport, league, event, selection, start_time, odds) "
                "VALUES ('football', 'Test', ?, 'A', ?, 2.0)",
                (event, start.isoformat()),
            )
        return bet

    def test_early_and_verified_final_are_published(self) -> None:
        now = datetime.now(timezone.utc)
        early = self._bet(now + timedelta(hours=5), "early")
        final = self._bet(now + timedelta(minutes=45), "final")
        context = SportContextDatabase(self.settings)
        with context.connect() as conn:
            conn.execute(
                """
                INSERT INTO sport_context_features (
                    sport, league, event, external_event_id, start_time,
                    lineup_confirmed, source, captured_at, source_hash
                ) VALUES ('football', 'Test', 'final', 'final', ?, 1,
                          'provider', ?, 'context-final')
                """,
                (final.start_time, now.isoformat()),
            )
        result = SportResult(sport="football", mode="scan", bets=[early, final])
        summary = apply_football_release_policy(
            [{"result": result}], self.settings, now=now
        )
        self.assertEqual(summary.early, 1)
        self.assertEqual(summary.final, 1)
        self.assertEqual([bet.release_stage for bet in result.bets], ["EARLY", "FINAL"])

    def test_unverified_close_tip_is_held(self) -> None:
        now = datetime.now(timezone.utc)
        bet = self._bet(now + timedelta(minutes=40), "held")
        result = SportResult(sport="football", mode="scan", bets=[bet])
        summary = apply_football_release_policy(
            [{"result": result}], self.settings, now=now
        )
        self.assertEqual(summary.awaiting_lineup, 1)
        self.assertEqual(result.bets, [])


if __name__ == "__main__":
    unittest.main()

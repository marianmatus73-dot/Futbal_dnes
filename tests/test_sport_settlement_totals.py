import asyncio
import sqlite3
import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.modules.setdefault("aiohttp", SimpleNamespace())

from core.config import Settings
from core.sport_quant import init_sport_db
from core.sport_settlement import settle_sport_bets


class SportSettlementTotalsTests(unittest.TestCase):
    def test_inactive_tournament_key_and_exact_event_id_are_used(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            db = Path(tmp) / "bets.db"
            settings = Settings(db_file=str(db), odds_api_key="test")
            init_sport_db(settings)
            with sqlite3.connect(db) as conn:
                conn.execute("ALTER TABLE sport_bets ADD COLUMN external_event_id TEXT")
                for source_hash, event_id in (("right", "event-123"), ("wrong", "event-999")):
                    conn.execute(
                        """INSERT INTO sport_bets
                        (sport, league, event, home_team, away_team, market,
                         selection, odds, stake, result, source_hash,
                         external_event_id)
                        VALUES ('tennis', 'tennis_atp_finished',
                                'Player A vs Player B', 'Player A', 'Player B',
                                'h2h', 'Player A', 1.8, 1, 'OPEN', ?, ?)""",
                        (source_hash, event_id),
                    )
            scores = [{
                "id": "event-123", "completed": True,
                "home_team": "Player A", "away_team": "Player B",
                "scores": [
                    {"name": "Player A", "score": "2"},
                    {"name": "Player B", "score": "0"},
                ],
            }]
            with (
                patch(
                    "core.sport_settlement.fetch_scores",
                    AsyncMock(return_value=scores),
                ) as fetch,
                patch("core.sport_settlement.update_closing_lines"),
                patch("core.sport_settlement.refresh_bookmaker_stats"),
            ):
                settled = asyncio.run(
                    settle_sport_bets(settings, "tennis", [])
                )
            self.assertEqual(settled, 1)
            self.assertEqual(fetch.await_args.args[1], "tennis_atp_finished")
            with sqlite3.connect(db) as conn:
                results = conn.execute(
                    "SELECT source_hash, result FROM sport_bets ORDER BY source_hash"
                ).fetchall()
            self.assertEqual(results, [("right", "WON"), ("wrong", "OPEN")])

    def test_totals_result_and_score_are_persisted(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            db = Path(tmp) / "bets.db"
            settings = Settings(db_file=str(db), odds_api_key="test")
            init_sport_db(settings)
            with sqlite3.connect(db) as conn:
                conn.execute(
                    """INSERT INTO sport_bets
                    (sport, league, event, home_team, away_team, market,
                     selection, odds, stake, result, source_hash)
                    VALUES ('football', 'soccer_test', 'Home vs Away', 'Home',
                            'Away', 'totals_2.5', 'Over 2.5', 1.9, 10, 'OPEN', 'x')"""
                )
            scores = [{
                "completed": True, "home_team": "Home", "away_team": "Away",
                "scores": [{"name": "Home", "score": "2"}, {"name": "Away", "score": "1"}],
            }]
            with patch("core.sport_settlement.fetch_scores", AsyncMock(return_value=scores)), patch("core.sport_settlement.update_closing_lines"), patch("core.sport_settlement.refresh_bookmaker_stats"):
                settled = asyncio.run(settle_sport_bets(settings, "football", ["soccer_test"]))
            self.assertEqual(settled, 1)
            with sqlite3.connect(db) as conn:
                row = conn.execute("SELECT result, final_score, home_goals, away_goals FROM sport_bets").fetchone()
            self.assertEqual(row, ("WON", "2-1", 2, 1))


if __name__ == "__main__":
    unittest.main()


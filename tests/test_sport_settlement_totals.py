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

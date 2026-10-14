from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from core.config import Settings
from core.sport_walkforward import walkforward_report


class SportWalkforwardTests(unittest.TestCase):
    def test_separates_sport_market_and_odds_tier(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            db = Path(directory) / "bets.db"
            with sqlite3.connect(db) as conn:
                conn.execute(
                    "CREATE TABLE sport_bets (sport TEXT, market TEXT, odds REAL, "
                    "prob_final REAL, result TEXT, start_time TEXT, tracking_mode TEXT)"
                )
                conn.executemany(
                    "INSERT INTO sport_bets VALUES (?,?,?,?,?,?,?)",
                    [
                        ("football", "h2h", 1.40, .75, "WON", "2026-01-01", "LIVE"),
                        ("football", "h2h", 2.10, .55, "LOST", "2026-01-02", "LIVE"),
                        ("football", "totals_2.5", 2.70, .45, "WON", "2026-01-03", "LIVE"),
                        ("football", "h2h", 1.30, .80, "WON", "2026-01-04", "LOW_ODDS_WATCH"),
                    ],
                )
            report = walkforward_report(
                Settings(db_file=str(db)), min_samples=2,
                export_dir=Path(directory) / "exports",
            )
            self.assertEqual(report["football"]["samples"], 3)
            self.assertEqual(report["football/market=h2h"]["samples"], 2)
            self.assertEqual(report["football/tier=SAFE_1.20_1.60"]["samples"], 1)
            self.assertEqual(report["football/tier=VALUE_1.61_2.50"]["samples"], 1)
            self.assertEqual(report["football/tier=EXPERIMENTAL_2.51_PLUS"]["samples"], 1)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from core.config import Settings
from core.professional_model_table import build_professional_model_table


class ProfessionalModelTableTests(unittest.TestCase):
    def test_football_low_odds_only_counts_settled_h2h_in_exact_band(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            db = Path(directory) / "bets.db"
            with sqlite3.connect(db) as conn:
                conn.execute("""CREATE TABLE sport_bets (
                    sport TEXT, league TEXT, market TEXT, odds REAL,
                    stake REAL, profit REAL, clv_pct REAL, prob_final REAL,
                    result TEXT, start_time TEXT, settled_at TEXT)""")
                conn.executemany("INSERT INTO sport_bets VALUES (?,?,?,?,?,?,?,?,?,?,?)", [
                    ("football", "Liga A", "h2h", 1.20, 1, .2, None, .85, "WON", "2026-09-01", "2026-09-02"),
                    ("football", "Liga A", "h2h", 1.60, 1, -1, None, .65, "LOST", "2026-09-01", "2026-09-02"),
                    ("football", "Liga A", "h2h", 1.61, 1, .61, None, .65, "WON", "2026-09-01", "2026-09-02"),
                    ("football", "Liga A", "totals_2.5", 1.50, 1, .5, None, .70, "WON", "2026-09-01", "2026-09-02"),
                    ("football", "Liga A", "h2h", 1.40, 1, 0, None, .70, "OPEN", "2026-09-01", None),
                ])
            payload = build_professional_model_table(Settings(db_file=str(db)), export_dir=Path(directory) / "exports")
            band = payload["sports"]["football"]["low_odds_1_20_1_60"]
            self.assertEqual((band["settled"], band["wins"], band["losses"]), (2, 1, 1))
            self.assertEqual(band["hit_rate_pct"], 50.0)
            self.assertEqual(band["profit"], -0.8)
            self.assertEqual(band["yield_pct"], -40.0)

    def test_metrics_periods_dimensions_and_drawdown(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            db = Path(directory) / "bets.db"
            exports = Path(directory) / "exports"
            with sqlite3.connect(db) as conn:
                conn.execute(
                    """
                    CREATE TABLE sport_bets (
                        sport TEXT, league TEXT, market TEXT, odds REAL,
                        stake REAL, profit REAL, clv_pct REAL, prob_final REAL,
                        result TEXT, start_time TEXT, settled_at TEXT
                    )
                    """
                )
                conn.executemany(
                    "INSERT INTO sport_bets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        ("football", "Liga A", "h2h", 2.0, 1.0, 1.0, .05, .70,
                         "WON", "2026-08-01T10:00:00Z", "2026-08-01T12:00:00Z"),
                        ("football", "Liga A", "h2h", 1.8, 1.0, -1.0, -.02, .60,
                         "LOST", "2026-08-02T10:00:00Z", "2026-08-02T12:00:00Z"),
                        ("football", "Liga B", "totals", 2.2, 1.0, -1.0, None, .55,
                         "LOST", "2026-08-03T10:00:00Z", "2026-08-03T12:00:00Z"),
                    ],
                )

            payload = build_professional_model_table(
                Settings(db_file=str(db)), export_dir=exports
            )
            football = payload["sports"]["football"]
            self.assertEqual(football["all_time"]["settled"], 3)
            self.assertEqual(football["all_time"]["profit"], -1.0)
            self.assertEqual(football["all_time"]["max_drawdown"], 2.0)
            self.assertEqual(football["all_time"]["clv_samples"], 2)
            self.assertIsNotNone(football["all_time"]["brier_score"])
            self.assertIn("Liga A", football["by_league"])
            self.assertIn("1.50-1.99", football["by_odds"])
            self.assertIn("totals", football["by_market"])
            self.assertTrue((exports / "professional_model_table.json").exists())
            self.assertTrue((exports / "professional_model_table.csv").exists())
            self.assertTrue((exports / "professional_model_table.html").exists())


if __name__ == "__main__":
    unittest.main()


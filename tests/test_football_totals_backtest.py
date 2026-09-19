import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from core.football_totals_backtest import evaluate_totals


class FootballTotalsBacktestTests(unittest.TestCase):
    def test_excludes_post_match_and_unsettled_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "bets.db"
            with closing(sqlite3.connect(db)) as conn:
                conn.execute("""CREATE TABLE sport_bets (
                    sport TEXT, market TEXT, selection TEXT, odds REAL,
                    prob_final REAL, stake REAL, start_time TEXT,
                    created_at TEXT, result TEXT)""")
                conn.executemany("INSERT INTO sport_bets VALUES (?,?,?,?,?,?,?,?,?)", [
                    ("football", "totals_2.5", "Under 2.5", 1.8, .6, 10, "2026-01-02T12:00:00Z", "2026-01-01T12:00:00Z", "WON"),
                    ("football", "totals_2.5", "Over 2.5", 2.0, .5, 10, "2026-01-02T12:00:00Z", "2026-01-01T12:00:00Z", "LOST"),
                    ("football", "totals_2.5", "Over 2.5", 2.0, .5, 10, "2026-01-02T12:00:00Z", "2026-01-03T12:00:00Z", "WON"),
                    ("football", "totals_2.5", "Over 2.5", 2.0, .5, 10, "2026-01-04T12:00:00Z", "2026-01-01T12:00:00Z", "OPEN"),
                ])
                conn.commit()
            result = evaluate_totals(db, min_samples=2)
            self.assertEqual(result["sample"], 2)
            self.assertEqual(result["excluded"], {"unsettled": 1, "not_pre_match": 1, "invalid_values": 0})
            self.assertEqual(result["net_profit"], -2.0)
            self.assertEqual(result["yield_pct"], -10.0)

    def test_empty_sample_does_not_claim_performance(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(evaluate_totals(Path(tmp) / "missing.db")["status"], "NO_DATABASE")

    def test_exported_csv_without_totals_is_insufficient(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "history.csv"
            file.write_text(
                "sport,market,selection,odds,prob_final,stake,start_time,created_at,result\n"
                "football,h2h,Home,1.8,0.6,10,2026-01-02T12:00:00Z,2026-01-01T12:00:00Z,WON\n",
                encoding="utf-8",
            )
            result = evaluate_totals(file)
            self.assertEqual(result["status"], "INSUFFICIENT_SAMPLE")
            self.assertEqual(result["sample"], 0)


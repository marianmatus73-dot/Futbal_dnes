from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.config import Settings
from core.settlement_integrity import build_settlement_integrity


class SettlementIntegrityTests(unittest.TestCase):
    def test_flags_overdue_and_incomplete_settlement(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            db = Path(directory) / "bets.db"
            old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
            with sqlite3.connect(db) as conn:
                conn.execute(
                    "CREATE TABLE sport_bets (sport TEXT, result TEXT, start_time TEXT, "
                    "profit REAL, final_score TEXT, closing_odds REAL, tracking_mode TEXT)"
                )
                conn.executemany(
                    "INSERT INTO sport_bets VALUES (?,?,?,?,?,?,?)",
                    [
                        ("football", "WON", old, .8, "2-1", 1.7, "LIVE"),
                        ("football", "LOST", old, None, "", None, "LIVE"),
                        ("football", "OPEN", old, None, None, None, "LIVE"),
                        ("football", "UNRESOLVED", old, 0, "", None, "LIVE"),
                        ("football", "VOID", old, 0, "", None, "LIVE"),
                    ],
                )
            payload = build_settlement_integrity(
                Settings(db_file=str(db)), Path(directory) / "exports"
            )
            football = payload["sports"][0]
            self.assertEqual(football["status"], "BLOCKED")
            self.assertEqual(football["overdue_open"], 1)
            self.assertEqual(football["open"], 1)
            self.assertEqual(football["unresolved_excluded"], 1)
            self.assertEqual(football["unknown_result"], 0)
            self.assertEqual(football["settled_missing_profit"], 1)
            self.assertEqual(football["settled_missing_final_score"], 1)
            self.assertEqual(football["closing_coverage"], round(1 / 3, 4))

    def test_unresolved_and_scoreless_void_are_not_active_failures(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            db = Path(directory) / "bets.db"
            old = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
            with sqlite3.connect(db) as conn:
                conn.execute(
                    "CREATE TABLE sport_bets (sport TEXT, result TEXT, start_time TEXT, "
                    "profit REAL, final_score TEXT, closing_odds REAL, tracking_mode TEXT)"
                )
                conn.executemany(
                    "INSERT INTO sport_bets VALUES (?,?,?,?,?,?,?)",
                    [
                        ("football", "UNRESOLVED", old, 0, "", None, "LIVE"),
                        ("football", "VOID", old, 0, "", None, "LIVE"),
                    ],
                )
            payload = build_settlement_integrity(
                Settings(db_file=str(db)), Path(directory) / "exports"
            )
            football = payload["sports"][0]
            self.assertEqual(football["open"], 0)
            self.assertEqual(football["overdue_open"], 0)
            self.assertEqual(football["unresolved_excluded"], 1)
            self.assertEqual(football["settled_missing_final_score"], 0)
            self.assertEqual(football["status"], "WARNING")


if __name__ == "__main__":
    unittest.main()

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from core.config import Settings
from core.mobile_history import export_mobile_tip_history


class MobileHistoryTests(unittest.TestCase):
    def test_exports_at_most_five_unique_real_analyses_per_sport(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            db = root / "bets.db"
            with sqlite3.connect(db) as conn:
                conn.execute(
                    """CREATE TABLE sport_bets (
                    id INTEGER PRIMARY KEY, sport TEXT, league TEXT, event TEXT,
                    selection TEXT, odds REAL, market TEXT, result TEXT,
                    start_time TEXT, created_at TEXT, settled_at TEXT,
                    final_score TEXT, home_goals INTEGER, away_goals INTEGER)"""
                )
                for index in range(7):
                    conn.execute(
                        "INSERT INTO sport_bets VALUES (?, 'tennis', 'ATP', ?, ?, 1.8, 'h2h', ?, ?, ?, NULL, ?, NULL, NULL)",
                        (index + 1, f"A{index} vs B{index}", f"A{index}", "WON" if index == 6 else "OPEN", f"2026-08-{20 + index}T12:00:00Z", f"2026-08-{20 + index}T09:00:00Z", "2-0" if index == 6 else None),
                    )
            path = export_mobile_tip_history(Settings(db_file=str(db)), export_dir=root / "exports")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["sports"]["tennis"]), 5)
            self.assertEqual(payload["sports"]["tennis"][0]["result"], "WON")
            self.assertEqual(payload["sports"]["tennis"][0]["final_score"], "2-0")


if __name__ == "__main__":
    unittest.main()

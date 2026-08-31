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
                    final_score TEXT, home_goals INTEGER, away_goals INTEGER,
                    prob_final REAL, edge REAL, stake REAL)"""
                )
                for index in range(7):
                    conn.execute(
                        """INSERT INTO sport_bets (
                        id, sport, league, event, selection, odds, market,
                        result, start_time, created_at, settled_at, final_score,
                        home_goals, away_goals, prob_final, edge, stake
                        ) VALUES (?, 'tennis', 'ATP', ?, ?, 1.8, 'h2h', ?, ?, ?,
                                  NULL, ?, NULL, NULL, .60, .08, 1.0)""",
                        (index + 1, f"A{index} vs B{index}", f"A{index}", "WON" if index == 6 else "OPEN", f"2026-08-{20 + index}T12:00:00Z", f"2026-08-{20 + index}T09:00:00Z", "2-0" if index == 6 else None),
                    )
                conn.executemany(
                    """INSERT INTO sport_bets (
                    id, sport, league, event, selection, odds, market, result,
                    start_time, created_at, prob_final, edge, stake
                    ) VALUES (?, 'baseball', 'MLB', 'A vs B', ?, ?, 'h2h',
                              'OPEN', '2026-09-01T12:00:00Z',
                              '2026-08-31T09:00:00Z', ?, ?, 1.0)""",
                    [
                        (100, "A", 2.10, .51, .04),
                        (101, "B", 1.80, .64, .15),
                    ],
                )
            path = export_mobile_tip_history(Settings(db_file=str(db)), export_dir=root / "exports")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["sports"]["tennis"]), 5)
            self.assertEqual(payload["sports"]["tennis"][0]["result"], "WON")
            self.assertEqual(payload["sports"]["tennis"][0]["final_score"], "2-0")
            self.assertEqual(len(payload["sports"]["baseball"]), 1)
            self.assertEqual(payload["sports"]["baseball"][0]["pick"], "B")


if __name__ == "__main__":
    unittest.main()


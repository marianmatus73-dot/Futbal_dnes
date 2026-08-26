import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from core.config import Settings
from core.mobile_performance import export_mobile_performance


class MobilePerformanceTests(unittest.TestCase):
    def test_exports_bankroll_and_drawdown_only_from_settled_results(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            db = root / "bets.db"
            with sqlite3.connect(db) as conn:
                conn.execute("CREATE TABLE sport_bets (id INTEGER, sport TEXT, result TEXT, profit REAL, clv_pct REAL, settled_at TEXT)")
                conn.executemany("INSERT INTO sport_bets VALUES (?,?,?,?,?,?)", [
                    (1, "football", "WON", 10, 2.5, "2026-08-01T10:00:00Z"),
                    (2, "football", "LOST", -20, -1.0, "2026-08-02T10:00:00Z"),
                    (3, "tennis", "UNRESOLVED", 999, 99, "2026-08-03T10:00:00Z"),
                ])
            path = export_mobile_performance(Settings(db_file=str(db), bank=1000), export_dir=root / "exports")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["current_bankroll"], 990)
            self.assertEqual(len(payload["points"]), 2)
            self.assertLess(payload["points"][-1]["drawdown_pct"], 0)


if __name__ == "__main__":
    unittest.main()

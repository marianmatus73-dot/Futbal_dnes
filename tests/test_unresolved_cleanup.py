import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

sys.modules.setdefault("aiohttp", SimpleNamespace())

from core.config import Settings
from core.sport_quant import init_sport_db
from core.sport_settlement import expire_historical_unresolved


class UnresolvedCleanupTests(unittest.TestCase):
    def test_only_old_open_rows_are_closed_without_profit(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            db = Path(tmp) / "bets.db"
            settings = Settings(db_file=str(db))
            init_sport_db(settings)
            old = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
            future = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
            with sqlite3.connect(db) as conn:
                for source, start, result in (("old", old, "OPEN"), ("future", future, "OPEN"), ("won", old, "WON")):
                    conn.execute(
                        """INSERT INTO sport_bets
                        (sport, league, event, market, selection, odds, stake,
                         start_time, source_hash, result)
                        VALUES ('tennis','test',?,'h2h','A',2.0,1.0,?,?,?)""",
                        (source, start, source, result),
                    )
            self.assertEqual(expire_historical_unresolved(settings, older_than_days=7), 1)
            with sqlite3.connect(db) as conn:
                rows = dict(conn.execute("SELECT event, result FROM sport_bets"))
            self.assertEqual(rows, {"old": "UNRESOLVED", "future": "OPEN", "won": "WON"})


if __name__ == "__main__":
    unittest.main()

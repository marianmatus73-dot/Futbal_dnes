from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from core.config import Settings
from core.operational_health import build_operational_health


class OperationalHealthTests(unittest.TestCase):
    def test_exports_health_funnel_and_xg_free_modes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            db = root / "bets.db"
            with closing(sqlite3.connect(db)) as conn:
                conn.execute(
                    "CREATE TABLE sport_bets (sport TEXT, result TEXT, "
                    "external_event_id TEXT)"
                )
                conn.execute(
                    "CREATE TABLE sport_odds_snapshots (sport TEXT, captured_at TEXT)"
                )
                conn.execute(
                    "INSERT INTO sport_bets VALUES ('football','OPEN','event-1')"
                )
                conn.execute(
                    "INSERT INTO sport_bets VALUES ('tennis','OPEN','')"
                )
                conn.execute(
                    "INSERT INTO sport_odds_snapshots VALUES ('football',?)",
                    (datetime.now(timezone.utc).isoformat(),),
                )
                conn.commit()
            summary = SimpleNamespace(
                candidates=8, accepted=2, rejected=6,
                rejected_reasons={"edge": 4, "odds": 2},
            )
            payload = build_operational_health(
                Settings(db_file=str(db)), summary, export_dir=root / "exports"
            )

            self.assertEqual(payload["funnel"]["accepted"], 2)
            self.assertFalse(payload["football_without_paid_xg"]["paid_xg_required"])
            self.assertEqual(payload["football_without_paid_xg"]["double_chance"], "LIVE")
            self.assertEqual(payload["football_without_paid_xg"]["totals"], "WATCH_ONLY_WITHOUT_XG")
            football = next(row for row in payload["sports"] if row["sport"] == "football")
            tennis = next(row for row in payload["sports"] if row["sport"] == "tennis")
            self.assertEqual(football["snapshots_24h"], 1)
            self.assertEqual(tennis["status"], "ATTENTION")
            self.assertTrue((root / "exports" / "operational_health.json").exists())


if __name__ == "__main__":
    unittest.main()

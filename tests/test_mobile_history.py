import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
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
                        (index + 1, f"A{index} vs B{index}", f"A{index}", "WON" if index == 6 else "LOST", f"2026-08-31T{10 + index:02d}:00:00Z", f"2026-08-31T{7 + index:02d}:00:00Z", "2-0" if index == 6 else None),
                    )
                conn.executemany(
                    """INSERT INTO sport_bets (
                    id, sport, league, event, selection, odds, market, result,
                    start_time, created_at, prob_final, edge, stake
                    ) VALUES (?, 'baseball', 'MLB', 'A vs B', ?, ?, 'h2h',
                              'OPEN', '2026-08-31T20:00:00Z',
                              '2026-08-31T09:00:00Z', ?, ?, 1.0)""",
                    [
                        (100, "A", 2.10, .51, .04),
                        (101, "B", 1.80, .64, .15),
                    ],
                )
                conn.execute(
                    """INSERT INTO sport_bets (
                    id, sport, league, event, selection, odds, market, result,
                    start_time, created_at, prob_final, edge, stake
                    ) VALUES (200, 'football', 'Old league', 'Old A vs Old B',
                              'Old A', 2.0, 'h2h', 'OPEN',
                              '2024-09-27T18:45:00Z', '2024-09-27T09:00:00Z',
                              .55, .05, 1.0)"""
                )
                conn.execute(
                    """CREATE TABLE sport_shadow_candidates (
                    id INTEGER PRIMARY KEY, created_at TEXT, sport TEXT,
                    league TEXT, external_event_id TEXT, event TEXT,
                    home_team TEXT, away_team TEXT, market TEXT,
                    selection TEXT, bookmaker TEXT, odds REAL,
                    probability REAL, start_time TEXT, model_version TEXT,
                    result TEXT, profit_units REAL, final_score TEXT,
                    settled_at TEXT)"""
                )
                conn.executemany(
                    """INSERT INTO sport_shadow_candidates (
                    id, created_at, sport, league, external_event_id, event,
                    market, selection, bookmaker, odds, probability,
                    start_time, model_version, result, final_score, settled_at
                    ) VALUES (?, ?, 'handball', 'Bundesliga', ?, ?, 'h2h', ?,
                              'Example Book', ?, ?, ?, 'shadow-v1', ?, ?, ?)""",
                    [
                        (300, "2026-08-31T08:00:00Z", "hb-1", "Kiel vs Berlin", "Kiel", 1.55, .70, "2026-08-31T18:00:00Z", "OPEN", None, None),
                        (301, "2026-08-30T08:00:00Z", "hb-2", "Flensburg vs Leipzig", "Flensburg", 1.70, .65, "2026-08-30T18:00:00Z", "WON", "31-27", "2026-08-30T20:00:00Z"),
                    ],
                )
            path = export_mobile_tip_history(
                Settings(db_file=str(db)),
                export_dir=root / "exports",
                now=datetime(2026, 8, 31, 15, 30, tzinfo=timezone.utc),
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["sports"]["tennis"]), 5)
            self.assertEqual(payload["sports"]["tennis"][0]["result"], "WON")
            self.assertEqual(payload["sports"]["tennis"][0]["final_score"], "2-0")
            self.assertEqual(len(payload["sports"]["baseball"]), 1)
            self.assertEqual(payload["sports"]["baseball"][0]["pick"], "B")
            self.assertNotIn("football", payload["sports"])
            self.assertEqual(len(payload["results_sports"]["tennis"]), 5)
            self.assertTrue(all(item["result"] in {"WON", "LOST", "VOID"} for item in payload["results_sports"]["tennis"]))
            self.assertEqual(payload["sports"]["handball"][0]["match"], "Kiel vs Berlin")
            self.assertEqual(payload["sports"]["handball"][0]["tracking_mode"], "SHADOW")
            self.assertTrue(payload["sports"]["handball"][0]["observation_only"])
            self.assertEqual(payload["results_sports"]["handball"][0]["result"], "WON")
            self.assertEqual(payload["results_sports"]["handball"][0]["final_score"], "31-27")


if __name__ == "__main__":
    unittest.main()


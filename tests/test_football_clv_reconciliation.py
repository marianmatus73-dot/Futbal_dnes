from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from core.config import Settings
from core.football_market import FootballMarketDatabase


class FootballCLVReconciliationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self.temp_dir.name) / "bets.db"
        self.settings = Settings(db_file=str(self.db_path))
        self.market = FootballMarketDatabase(self.settings)
        self.market.init_db()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE sport_bets (
                    id INTEGER PRIMARY KEY, sport TEXT, league TEXT, event TEXT,
                    selection TEXT, bookmaker TEXT, odds REAL, start_time TEXT,
                    external_event_id TEXT, closing_odds REAL, clv_pct REAL
                )
                """
            )
            conn.execute(
                """
                INSERT INTO sport_bets VALUES
                (1, 'football', 'Test', 'A vs B', 'A', 'Book A', 2.10,
                 '2026-08-20T20:00:00+00:00', 'event-1', NULL, NULL)
                """
            )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _insert_closing(self, *, odds: float, captured_at: str, source: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO football_market_closing (
                    sport_key, league, event, home_team, away_team,
                    commence_time, external_event_id, selection, bookmaker,
                    closing_odds, closing_probability, captured_at,
                    source_hash, created_at
                ) VALUES (?, 'Test', 'A vs B', 'A', 'B', ?, ?, 'A',
                          'Book A', ?, ?, ?, ?, ?)
                """,
                (
                    "soccer_test", "2026-08-20T20:00:00+00:00", "event-1",
                    odds, 1.0 / odds, captured_at, source, captured_at,
                ),
            )

    def test_reconciles_by_event_id_and_preserves_opening_odds(self) -> None:
        self._insert_closing(
            odds=1.90,
            captured_at="2026-08-20T19:55:00+00:00",
            source="closing-before",
        )

        self.assertEqual(self.market.reconcile_closing_lines(), 1)
        with sqlite3.connect(self.db_path) as conn:
            odds, closing, clv = conn.execute(
                "SELECT odds, closing_odds, clv_pct FROM sport_bets WHERE id=1"
            ).fetchone()
            audit_count = conn.execute(
                "SELECT COUNT(*) FROM football_clv_audit WHERE bet_id=1"
            ).fetchone()[0]

        self.assertEqual(odds, 2.10)
        self.assertEqual(closing, 1.90)
        self.assertAlmostEqual(clv, 2.10 / 1.90 - 1.0, places=5)
        self.assertEqual(audit_count, 1)
        self.assertEqual(self.market.reconcile_closing_lines(), 0)

    def test_ignores_snapshot_captured_after_kickoff(self) -> None:
        self._insert_closing(
            odds=1.80,
            captured_at="2026-08-20T20:01:00+00:00",
            source="closing-after",
        )
        self.assertEqual(self.market.reconcile_closing_lines(), 0)
        metrics = self.market.clv_metrics()
        self.assertEqual(metrics.eligible_bets, 1)
        self.assertEqual(metrics.closing_odds_samples, 0)
        self.assertEqual(metrics.clv_ready, 0)

    def test_market_metrics_use_real_rows_without_double_counting(self) -> None:
        event = {
            "id": "event-market-1",
            "home_team": "A",
            "away_team": "B",
            "commence_time": "2026-08-20T20:00:00+00:00",
            "bookmakers": [{
                "title": "Book A",
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": "A", "price": 2.0},
                        {"name": "B", "price": 2.1},
                    ],
                }],
            }],
        }
        self.assertEqual(self.market.save_event_snapshot(
            sport_key="soccer_test", league="Test", event=event,
        ), 2)
        self.assertEqual(self.market.save_closing_snapshot(
            sport_key="soccer_test", league="Test", event=event,
        ), 2)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE football_market_snapshots_v14 (id INTEGER)"
            )
            conn.executemany(
                "INSERT INTO football_market_snapshots_v14 VALUES (?)",
                [(1,), (2,), (3,)],
            )

        metrics = self.market.market_metrics()
        self.assertTrue(metrics.available)
        self.assertEqual(metrics.live_snapshots, 2)
        self.assertEqual(metrics.legacy_snapshots, 3)
        self.assertEqual(metrics.closing_snapshots, 2)
        self.assertEqual(metrics.total_snapshots, 2)


if __name__ == "__main__":
    unittest.main()


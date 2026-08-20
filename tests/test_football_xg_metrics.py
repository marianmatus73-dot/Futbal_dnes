from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from core.config import Settings
from core.football_dataset_v15 import FootballDatasetV15
from core.football_xg import FootballXGDatabase


class FootballXGMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self.temp_dir.name) / "bets.db"
        self.settings = Settings(db_file=str(self.db_path))
        self.xg = FootballXGDatabase(self.settings)
        self.xg.init_db()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_metrics_count_only_persisted_xg_and_dataset_coverage(self) -> None:
        self.assertFalse(self.xg.metrics().available)
        self.assertTrue(self.xg.update_after_match(
            league="Test", home_team="A", away_team="B",
            home_xg=1.7, away_xg=0.8,
            played_at="2026-08-20T20:00:00+00:00",
            source="provider", source_hash="real-xg-1",
        ))
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE football_dataset_v15 (id INTEGER, has_xg INTEGER)"
            )
            conn.executemany(
                "INSERT INTO football_dataset_v15 VALUES (?, ?)",
                [(1, 1), (2, 0)],
            )

        metrics = self.xg.metrics()
        self.assertTrue(metrics.available)
        self.assertEqual(metrics.history_rows, 1)
        self.assertEqual(metrics.rated_teams, 2)
        self.assertEqual(metrics.dataset_samples, 1)
        self.assertEqual(metrics.dataset_total, 2)
        self.assertEqual(metrics.dataset_coverage_pct, 50.0)

    def test_dataset_rejects_unrelated_old_head_to_head_xg(self) -> None:
        self.assertTrue(self.xg.update_after_match(
            league="Test", home_team="A", away_team="B",
            home_xg=1.7, away_xg=0.8,
            played_at="2026-08-20T20:00:00+00:00",
            source="provider", source_hash="real-xg-for-join",
        ))
        dataset = FootballDatasetV15(self.settings)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            old = dataset._xg_row(
                conn, league="Test", home_team="A", away_team="B",
                commence_time="2026-09-20T20:00:00+00:00",
            )
            exact = dataset._xg_row(
                conn, league="Test", home_team="A", away_team="B",
                commence_time="2026-08-20T20:00:00+00:00",
            )
        self.assertIsNone(old)
        self.assertIsNotNone(exact)
        self.assertEqual(exact["home_xg"], 1.7)


if __name__ == "__main__":
    unittest.main()


from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from core.config import Settings
from core.football_pipeline_metrics import load_football_pipeline_metrics


class FootballPipelineMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self.temp_dir.name) / "bets.db"
        self.settings = Settings(db_file=str(self.db_path))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_empty_database_is_building_without_fake_counts(self) -> None:
        self.assertEqual(load_football_pipeline_metrics(self.settings).matches, 0)

    def test_all_stage_counts_share_the_same_dataset_population(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE football_dataset_v15 (
                    source_hash TEXT, sport_key TEXT, event TEXT,
                    commence_time TEXT, has_closing_line INTEGER,
                    clv_probability REAL, clv_odds_ratio REAL
                )
                """
            )
            conn.executemany(
                "INSERT INTO football_dataset_v15 VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    ("key-1", "soccer_test", "A vs B", "2026-08-20", 1, .03, None),
                    ("key-2", "soccer_test", "C vs D", "2026-08-21", 0, None, None),
                    ("", "", "", "", 0, None, None),
                ],
            )
            conn.execute(
                "CREATE TABLE football_explainability_v15 (source_hash TEXT)"
            )
            conn.executemany(
                "INSERT INTO football_explainability_v15 VALUES (?)",
                [("key-1",), ("key-2",)],
            )

        metrics = load_football_pipeline_metrics(self.settings)
        self.assertEqual(metrics.matches, 3)
        self.assertEqual(metrics.resolved_matches, 2)
        self.assertEqual(metrics.keys_created, 2)
        self.assertEqual(metrics.joins_completed, 1)
        self.assertEqual(metrics.closing_written, 1)
        self.assertEqual(metrics.clv_ready, 1)
        self.assertEqual(metrics.explainability_rows, 2)
        for count in (
            metrics.resolved_matches, metrics.keys_created,
            metrics.joins_completed, metrics.closing_written, metrics.clv_ready,
        ):
            self.assertLessEqual(count, metrics.matches)


if __name__ == "__main__":
    unittest.main()


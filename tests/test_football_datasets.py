from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from core.config import Settings
from core.football_dataset_v15 import FootballDatasetV15
from core.football_postmatch_dataset_v14 import FootballPostmatchDatasetV14
from core.football_trainer import ensure_feature_history_table


class FootballDatasetPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db = Path(self.temp.name) / "bets.db"
        self.settings = Settings(db_file=str(self.db))
        ensure_feature_history_table(self.db)
        FootballPostmatchDatasetV14(self.settings)
        FootballDatasetV15(self.settings)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _feature(self) -> None:
        with sqlite3.connect(self.db) as conn:
            conn.execute(
                """
                INSERT INTO football_feature_history (
                    sport_key, league, event, selection, bookmaker,
                    commence_time, odds, market_selection_probability,
                    model_consensus_probability, result, source_hash,
                    created_at
                ) VALUES (
                    'soccer_test', 'Test', 'A vs B', 'A', 'Book',
                    '2026-08-20T20:00:00+00:00', 2.10, .48, .60,
                    'WON', 'feature-1', '2026-08-20T10:00:00+00:00'
                )
                """
            )

    def _closing(self) -> None:
        with sqlite3.connect(self.db) as conn:
            conn.execute(
                """
                CREATE TABLE football_market_closing (
                    id INTEGER PRIMARY KEY, sport_key TEXT, event TEXT,
                    selection TEXT, commence_time TEXT, bookmaker TEXT,
                    closing_odds REAL, closing_probability REAL,
                    captured_at TEXT
                )
                """
            )
            conn.execute(
                """
                INSERT INTO football_market_closing VALUES (
                    1, 'soccer_test', 'A vs B', 'A',
                    '2026-08-20T20:00:00+00:00', 'Closing Book',
                    1.95, 0.5128205128, '2026-08-20T19:50:00+00:00'
                )
                """
            )

    def test_settled_features_build_both_datasets(self) -> None:
        self._feature()
        postmatch = FootballPostmatchDatasetV14(self.settings).rebuild()
        dataset = FootballDatasetV15(self.settings).rebuild()
        self.assertEqual(postmatch.total_rows, 1)
        self.assertEqual(dataset.total_rows, 1)
        self.assertEqual(dataset.training_ready, 1)

    def test_compact_closing_row_carries_bookmaker_into_datasets(self) -> None:
        self._feature()
        self._closing()
        postmatch = FootballPostmatchDatasetV14(self.settings).rebuild()
        dataset = FootballDatasetV15(self.settings).rebuild()
        self.assertEqual(postmatch.missing_closing_line, 0)
        self.assertEqual(dataset.with_closing, 1)
        with sqlite3.connect(self.db) as conn:
            bookmaker = conn.execute(
                "SELECT closing_bookmaker FROM football_postmatch_dataset_v14"
            ).fetchone()[0]
        self.assertEqual(bookmaker, "Closing Book")


if __name__ == "__main__":
    unittest.main()

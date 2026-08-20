from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from core.config import Settings
from core.football_dataset_v15 import FootballDatasetV15
from core.football_elo import FootballEloDatabase
from core.football_team_form import FootballFormDatabase


class FootballEloFormMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self.temp_dir.name) / "bets.db"
        self.settings = Settings(db_file=str(self.db_path))
        self.elo = FootballEloDatabase(self.settings)
        self.form = FootballFormDatabase(self.settings)
        self.elo.init_db()
        self.form.init_db()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_availability_requires_real_match_history(self) -> None:
        self.assertFalse(self.elo.metrics().available)
        self.assertFalse(self.form.metrics().available)

        elo_result = self.elo.update_after_match(
            league="Test", home_team="A", away_team="B",
            home_goals=2, away_goals=1,
            played_at="2026-08-20T20:00:00+00:00",
            source_hash="elo-real-1",
        )
        form_result = self.form.update_after_match(
            league="Test", home_team="A", away_team="B",
            home_goals=2, away_goals=1,
            played_at="2026-08-20T20:00:00+00:00",
            source_hash="form-real-1",
        )
        self.assertTrue(elo_result.inserted)
        self.assertTrue(form_result)
        self.assertTrue(self.elo.metrics().available)
        self.assertTrue(self.form.metrics().available)

    def test_v15_dataset_reads_production_overall_elo_column(self) -> None:
        self.elo.update_after_match(
            league="Test", home_team="A", away_team="B",
            home_goals=2, away_goals=1, source_hash="elo-dataset-1",
        )
        dataset = FootballDatasetV15(self.settings)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            home, away = dataset._elo_context(
                conn, league="Test", home_team="A", away_team="B"
            )
        self.assertIsNotNone(home)
        self.assertIsNotNone(away)
        self.assertNotEqual(home, away)


if __name__ == "__main__":
    unittest.main()


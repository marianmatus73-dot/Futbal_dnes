from __future__ import annotations

import unittest
import sqlite3
import tempfile
from pathlib import Path

from core.api_sports_handball import (
    league_allowed,
    normalize_completed_game,
    normalize_game,
    select_balanced_games,
)
from core.config import Settings
from sports.handball import ensure_handball_learning_tables


class ApiSportsHandballTests(unittest.TestCase):
    def test_learning_tables_exist_before_history_restore(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            database = Path(directory) / "bets.db"
            ensure_handball_learning_tables(Settings(db_file=str(database)))
            with sqlite3.connect(database) as conn:
                names = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
            self.assertIn("sport_learning_observations", names)
            self.assertIn("sport_shadow_candidates", names)

    def test_normalizes_match_winner_odds(self) -> None:
        game = {
            "id": 42,
            "date": "2026-09-25T18:00:00+00:00",
            "league": {"id": 7, "name": "Champions League"},
            "teams": {"home": {"name": "Kiel"}, "away": {"name": "Barcelona"}},
        }
        odds = [{"bookmakers": [{"name": "Book", "bets": [{
            "name": "Match Winner",
            "values": [
                {"value": "Home", "odd": "1.80"},
                {"value": "Draw", "odd": "8.00"},
                {"value": "Away", "odd": "2.20"},
            ],
        }]}]}]
        result = normalize_game(game, odds)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["id"], "apisports:42")
        outcomes = result["bookmakers"][0]["markets"][0]["outcomes"]
        self.assertEqual([item["name"] for item in outcomes], ["Kiel", "Draw", "Barcelona"])

    def test_filters_priority_leagues_case_insensitively(self) -> None:
        game = {"league": {"name": "Liga ASOBAL"}}
        self.assertTrue(league_allowed(game, {"liga asobal"}))
        self.assertFalse(league_allowed(game, {"bundesliga"}))

    def test_accepts_provider_league_name_suffix(self) -> None:
        game = {"league": {"name": "Bundesliga - Germany"}}
        self.assertTrue(league_allowed(game, {"bundesliga"}))

    def test_balanced_selection_does_not_let_first_league_take_all_slots(self) -> None:
        games = [
            {"id": index, "date": f"2026-11-{index:02d}", "league": {"name": "League A"}}
            for index in range(1, 7)
        ] + [
            {"id": 100 + index, "date": f"2026-11-{index:02d}", "league": {"name": "League B"}}
            for index in range(1, 4)
        ]
        selected = select_balanced_games(
            games,
            league_names=set(),
            per_league_limit=2,
            total_limit=4,
        )
        leagues = [(game.get("league") or {}).get("name") for game in selected]
        self.assertEqual(leagues.count("League A"), 2)
        self.assertEqual(leagues.count("League B"), 2)

    def test_balanced_selection_prefers_new_events(self) -> None:
        games = [
            {"id": 1, "date": "2026-11-13", "league": {"name": "League A"}},
            {"id": 2, "date": "2026-11-14", "league": {"name": "League A"}},
        ]
        selected = select_balanced_games(
            games,
            league_names=set(),
            per_league_limit=1,
            total_limit=1,
            known_event_ids={"apisports:1"},
        )
        self.assertEqual(selected[0]["id"], 2)

    def test_normalizes_finished_game_for_settlement(self) -> None:
        game = {
            "id": 42,
            "status": {"short": "FT"},
            "teams": {"home": {"name": "Kiel"}, "away": {"name": "Barcelona"}},
            "scores": {"home": 31, "away": 29},
        }
        self.assertEqual(normalize_completed_game(game), {
            "id": "apisports:42",
            "completed": True,
            "home_team": "Kiel",
            "away_team": "Barcelona",
            "scores": [
                {"name": "Kiel", "score": 31},
                {"name": "Barcelona", "score": 29},
            ],
        })

    def test_does_not_treat_scheduled_game_as_finished(self) -> None:
        game = {
            "id": 43,
            "status": {"short": "NS"},
            "teams": {"home": {"name": "Kiel"}, "away": {"name": "Berlin"}},
            "scores": {"home": 0, "away": 0},
        }
        self.assertIsNone(normalize_completed_game(game))


if __name__ == "__main__":
    unittest.main()

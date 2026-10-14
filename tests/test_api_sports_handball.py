from __future__ import annotations

import unittest

from core.api_sports_handball import league_allowed, normalize_completed_game, normalize_game


class ApiSportsHandballTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

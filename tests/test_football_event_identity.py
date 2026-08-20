from __future__ import annotations

import unittest
from datetime import datetime, timezone

from core.event_time import is_closing_window
from core.football_settlement import CompletedFootballGame, FootballSettlementEngine, OpenFootballBet


class FootballEventIdentityTests(unittest.TestCase):
    def test_closing_window_is_time_bounded(self) -> None:
        now = datetime(2026, 8, 20, 10, 0, tzinfo=timezone.utc)
        self.assertTrue(is_closing_window("2026-08-20T20:00:00Z", captured_at=now))
        self.assertFalse(is_closing_window("2026-08-21T10:01:00Z", captured_at=now))
        self.assertFalse(is_closing_window("2026-08-20T09:59:00Z", captured_at=now))

    def test_external_event_id_wins_over_team_alias_matching(self) -> None:
        engine = object.__new__(FootballSettlementEngine)
        bet = OpenFootballBet(
            bet_id=1, source_hash="hash", sport_key="soccer_test", league="Test",
            event="Old A vs Old B", market="h2h", selection="Old A",
            start_time="2026-08-20T10:00:00Z", home_team="Old A", away_team="Old B",
            external_event_id="event-123",
        )
        game = CompletedFootballGame(
            event_id="event-123", sport_key="soccer_test",
            home_team="Renamed A", away_team="Renamed B",
            commence_time="2026-08-20T10:00:00Z", home_goals=2, away_goals=1,
            last_update="2026-08-20T12:00:00Z",
        )
        self.assertIs(engine._match_game(bet, [game]), game)


if __name__ == "__main__":
    unittest.main()

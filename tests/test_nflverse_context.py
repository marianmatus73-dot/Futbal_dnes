from __future__ import annotations

import unittest

from core.nflverse_context import build_signal, load_schedule_rows


CSV = """game_id,season,gameday,away_team,away_score,home_team,home_score,away_rest,home_rest
1,2026,2026-09-01,BUF,10,KC,30,7,7
2,2026,2026-09-08,DEN,17,KC,27,7,7
3,2026,2026-09-15,KC,24,LV,14,7,7
4,2026,2026-09-01,NE,28,BUF,14,7,7
5,2026,2026-09-08,BUF,13,MIA,24,7,7
6,2026,2026-09-15,NYJ,21,BUF,20,7,7
7,2026,2026-09-25,BUF,,KC,,6,10
"""


class NFLVerseContextTests(unittest.TestCase):
    def test_form_and_rest_create_bounded_selection_adjustment(self) -> None:
        rows = load_schedule_rows(CSV)
        signal = build_signal(
            rows,
            home_name="Kansas City Chiefs",
            away_name="Buffalo Bills",
            selection="Kansas City Chiefs",
            commence_time="2026-09-25T20:00:00Z",
        )
        self.assertTrue(signal.available)
        self.assertGreater(signal.adjustment, 0)
        self.assertLessEqual(signal.adjustment, 0.04)
        self.assertEqual(signal.home_rest, 10)
        self.assertEqual(signal.away_rest, 6)

    def test_away_selection_reverses_adjustment(self) -> None:
        rows = load_schedule_rows(CSV)
        home = build_signal(
            rows,
            home_name="Kansas City Chiefs",
            away_name="Buffalo Bills",
            selection="Kansas City Chiefs",
            commence_time="2026-09-25T20:00:00Z",
        )
        away = build_signal(
            rows,
            home_name="Kansas City Chiefs",
            away_name="Buffalo Bills",
            selection="Buffalo Bills",
            commence_time="2026-09-25T20:00:00Z",
        )
        self.assertAlmostEqual(home.adjustment, -away.adjustment)

    def test_insufficient_history_is_not_applied(self) -> None:
        signal = build_signal(
            load_schedule_rows(CSV.splitlines()[0] + "\n" + CSV.splitlines()[1]),
            home_name="Kansas City Chiefs",
            away_name="Buffalo Bills",
            selection="Kansas City Chiefs",
            commence_time="2026-09-25T20:00:00Z",
        )
        self.assertFalse(signal.available)
        self.assertEqual(signal.adjustment, 0.0)


if __name__ == "__main__":
    unittest.main()

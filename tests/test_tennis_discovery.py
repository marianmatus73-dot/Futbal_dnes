from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.modules.setdefault("aiohttp", SimpleNamespace())

from core.config import Settings
from sports.tennis import TennisModule


class TennisDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        os.environ.pop("TENNIS_SPORT_KEYS", None)

    async def test_auto_discovery_uses_new_active_tournaments(self) -> None:
        active = {"tennis_atp_new_event", "tennis_wta_new_event"}
        with (
            patch("sports.tennis.init_sport_db"),
            patch("sports.tennis.discover_active_sport_keys", AsyncMock(return_value=active)),
            patch("sports.tennis.settle_sport_bets", AsyncMock(return_value=0)),
            patch("sports.tennis.update_closing_lines", return_value=0),
            patch("sports.tennis.refresh_bookmaker_stats"),
            patch("sports.tennis.fetch_odds", AsyncMock(return_value=[])) as fetch,
        ):
            result = await TennisModule().scan(Settings(odds_api_key="key"))

        self.assertEqual(fetch.await_count, 2)
        requested = {call.args[1] for call in fetch.await_args_list}
        self.assertEqual(requested, active)
        self.assertIn("Events: 0", result.message)


if __name__ == "__main__":
    unittest.main()

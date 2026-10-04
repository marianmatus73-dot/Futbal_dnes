from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.model_governance import build_model_governance
from core.mobile_notifications import build_notifications
from core.rejection_explanations import explain_rejection


class ModelGovernanceTests(unittest.TestCase):
    def test_modes_coach_and_source_quality_are_exported_without_auto_apply(self) -> None:
        table = {"sports": {"football": {
            "all_time": {"settled": 120, "yield_pct": 4, "average_clv_pct": 1.2},
            "by_market": {
                "h2h": {"settled": 80, "yield_pct": 3, "average_clv_pct": 1},
                "totals_2.5": {"settled": 20, "yield_pct": -2},
            },
            "by_league": {
                "Good": {"settled": 60, "yield_pct": 2, "average_clv_pct": 1},
                "Weak": {"settled": 40, "yield_pct": -15, "average_clv_pct": -3},
            },
        }}}
        health = {"sports": [{"sport": "football", "snapshots_24h": 8, "missing_event_id": 0, "settled_bets": 120}]}
        with tempfile.TemporaryDirectory() as folder:
            payload = build_model_governance(table, health, folder)
            self.assertEqual(payload["football_market_modes"][0]["mode"], "LIVE")
            self.assertEqual(next(x for x in payload["football_league_trust"] if x["league"] == "Weak")["mode"], "BLOCKED")
            self.assertTrue(all(not item["auto_apply"] for item in payload["weekly_model_coach"]))
            self.assertFalse(any(item["auto_promote"] for item in payload["champion_challenger"]))
            self.assertEqual(payload["source_quality"][0]["status"], "GOOD")
            json.loads((Path(folder) / "model_governance.json").read_text(encoding="utf-8"))

    def test_notifications_only_emit_actionable_changes(self) -> None:
        previous = {"selected": [{"sport": "football", "event": "A-B", "match": "A-B", "market": "h2h", "selection": "A", "pick": "A", "release_stage": "EARLY", "lineup_verified": False}]}
        current = {"selected": [{"sport": "football", "event": "A-B", "match": "A-B", "market": "h2h", "selection": "A", "pick": "A", "release_stage": "FINAL", "lineup_verified": True, "opening_odds": 2.0, "odds": 1.85}]}
        types = {item["type"] for item in build_notifications(previous, current)}
        self.assertEqual(types, {"FINAL", "LINEUP", "ODDS_MOVE"})

    def test_rejection_explanation_is_human_readable(self) -> None:
        item = explain_rejection("league CLV below minimum")
        self.assertEqual(item["category"], "LIGA")
        self.assertIn("closing", item["explanation"])


if __name__ == "__main__":
    unittest.main()

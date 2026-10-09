from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.pro_tipper import build_pro_tip, filter_value_tips
from core.tip_card import (
    build_low_odds_watch,
    save_latest_rejected_candidates,
    save_latest_tip_card,
)


class TipCardTests(unittest.TestCase):
    def test_low_odds_watch_is_separate_and_deduplicates_opposite_sides(self) -> None:
        accepted = build_pro_tip(
            sport="football", league="test", match="A vs B", pick="A",
            odds=1.50, model_probability=0.72, model_score=78,
        )
        rejected = {
            "sport": "football", "league": "test", "event": "A vs B",
            "selection": "B", "market": "h2h", "odds": 1.55,
            "prob_final": 0.67, "prob_market": 0.64, "score": 70,
            "rejection_reason": "conservative edge below sport minimum",
        }
        rows = build_low_odds_watch([accepted], [rejected])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["pick"], "A")
        self.assertEqual(rows[0]["watch_status"], "PASSED_PRO_FILTER")

    def test_low_odds_watch_keeps_rejected_observation_without_publishing(self) -> None:
        rows = build_low_odds_watch([], [{
            "sport": "football", "league": "test", "event": "C vs D",
            "selection": "C", "market": "h2h", "odds": 1.40,
            "prob_final": 0.74, "prob_market": 0.71, "score": 68,
            "rejection_reason": "confidence below sport minimum",
        }])
        self.assertEqual(rows[0]["decision"], "REJECT")
        self.assertEqual(rows[0]["watch_status"], "WATCH_ONLY")

    def test_low_odds_watch_fills_three_daily_rows_from_scan_audit(self) -> None:
        audit = [
            {
                "league": "league", "event": f"Home {index} vs Away {index}",
                "selection": f"Home {index}", "bookmaker": "Book",
                "odds": 1.30 + index * 0.05, "prob_market": 0.70 - index * 0.02,
                "edge": -0.01, "decision": "BLOCK",
                "reason": f"edge below minimum; final={0.76 - index * 0.02:.2f}",
            }
            for index in range(4)
        ]
        rows = build_low_odds_watch([], [], audit_candidates=audit, limit=3)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["not_official_tip"] for row in rows))
        self.assertTrue(all(row["stake_units"] == 0.10 for row in rows))
        self.assertEqual(rows[0]["match"], "Home 0 vs Away 0")

    def test_low_odds_watch_keeps_only_one_side_per_match_from_audit(self) -> None:
        rows = build_low_odds_watch([], [], audit_candidates=[
            {"league": "league", "event": "A vs B", "selection": "A",
             "bookmaker": "Book", "odds": 1.40, "prob_market": 0.68,
             "reason": "edge below minimum; final=0.74"},
            {"league": "league", "event": "A vs B", "selection": "1X",
             "bookmaker": "Book", "odds": 1.30, "prob_market": 0.73,
             "reason": "double chance evidence or edge gate"},
        ], limit=3)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["selection"], "A")

    def test_value_filter_uses_expected_return_for_lower_odds(self) -> None:
        tip = build_pro_tip(
            sport="football",
            league="test",
            match="A vs B",
            pick="A",
            odds=1.50,
            model_probability=0.70,
            model_score=75,
        )
        self.assertLess(tip.edge, 0.04)
        self.assertAlmostEqual(tip.model_probability * tip.odds - 1.0, 0.05)
        self.assertEqual(filter_value_tips([tip]), [tip])

    def test_card_contains_complete_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tip = build_pro_tip(
                sport="football",
                league="test",
                match="A vs B",
                pick="A",
                odds=2.0,
                model_probability=0.60,
            )
            path = save_latest_tip_card(
                [tip], [tip], export_dir=Path(temp), top_limit=5
            )
            card = json.loads(path.read_text(encoding="utf-8"))
            for candidate in card["selected"] + card["rejected_sample"]:
                for field in (
                    "sport", "event", "selection", "odds",
                    "model_probability", "market_probability", "edge",
                    "confidence",
                    "bookmaker_weight", "bookmaker_samples", "bookmaker_label",
                ):
                    self.assertNotIn(candidate.get(field), (None, ""))

    def test_empty_run_replaces_stale_card(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            export_dir = Path(temp)
            stale = export_dir / "latest_tip_card.json"
            stale.write_text('{"selected":[{"stale":true}]}', encoding="utf-8")
            save_latest_tip_card([], [], export_dir=export_dir, top_limit=5)
            card = json.loads(stale.read_text(encoding="utf-8"))
            self.assertFalse(card["publishable"])
            self.assertEqual(card["selected"], [])
            self.assertEqual(card["rejected_sample"], [])

    def test_complete_rejected_export_keeps_risk_and_selection_items(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tip = build_pro_tip(
                sport="football", league="test", match="A vs B",
                pick="A", odds=2.0, model_probability=0.60,
            )
            path = save_latest_rejected_candidates(
                [{"sport": "tennis", "event": "C vs D",
                  "rejection_reason": "confidence below sport minimum"}],
                [tip],
                export_dir=Path(temp),
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["total"], 2)
            self.assertEqual(len(payload["candidates"]), 2)
            self.assertEqual(
                payload["candidates"][1]["rejection_stage"],
                "VALUE_OR_TOP_SELECTION",
            )


if __name__ == "__main__":
    unittest.main()


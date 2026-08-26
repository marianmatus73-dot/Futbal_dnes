from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.pro_tipper import build_pro_tip
from core.tip_card import save_latest_tip_card


class TipCardTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

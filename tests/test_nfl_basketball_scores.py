from __future__ import annotations

import unittest

from sports.basketball import basketball_selection_score
from sports.nfl import nfl_selection_score


class SportSelectionScoreTests(unittest.TestCase):
    def test_nfl_uses_selection_probability_not_edge(self) -> None:
        self.assertAlmostEqual(nfl_selection_score(0.71), 71.0)
        self.assertAlmostEqual(nfl_selection_score(0.12), 12.0)
        self.assertEqual(nfl_selection_score(1.5), 99.0)

    def test_basketball_uses_selection_probability_not_edge(self) -> None:
        self.assertAlmostEqual(basketball_selection_score(0.69), 69.0)
        self.assertAlmostEqual(basketball_selection_score(0.14), 14.0)
        self.assertEqual(basketball_selection_score(-0.1), 1.0)


if __name__ == "__main__":
    unittest.main()

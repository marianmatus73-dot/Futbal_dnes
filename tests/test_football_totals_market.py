from __future__ import annotations

import unittest

from core.market import best_total_prices, consensus_totals
from core.football_totals_confidence import totals_publication_confidence


def bookmaker(title: str, over: float, under: float) -> dict:
    return {
        "title": title,
        "markets": [{
            "key": "totals",
            "outcomes": [
                {"name": "Over", "point": 2.5, "price": over},
                {"name": "Under", "point": 2.5, "price": under},
            ],
        }],
    }


class FootballTotalsMarketTests(unittest.TestCase):
    def test_missing_xg_does_not_become_false_one_percent_confidence(self) -> None:
        self.assertIsNone(totals_publication_confidence(.10, 0.0))
        self.assertIsNone(totals_publication_confidence(.10, .29))
        self.assertEqual(totals_publication_confidence(.10, .60), 70.0)

    def test_consensus_removes_margin_and_best_prices_are_kept(self) -> None:
        books = [
            bookmaker("A", 1.90, 1.90),
            bookmaker("B", 2.00, 1.82),
            bookmaker("C", 1.95, 1.87),
        ]
        consensus = consensus_totals(books, min_books=3)
        self.assertAlmostEqual(sum(consensus.values()), 1.0, places=6)
        self.assertEqual(best_total_prices(books)["Over"], ("B", 2.00))
        self.assertEqual(best_total_prices(books)["Under"], ("A", 1.90))


if __name__ == "__main__":
    unittest.main()


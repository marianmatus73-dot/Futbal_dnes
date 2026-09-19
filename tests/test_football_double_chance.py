from __future__ import annotations

import unittest

from core.football_double_chance import (
    best_double_chance_prices,
    double_chance_probabilities,
    double_chance_won,
)


class FootballDoubleChanceTests(unittest.TestCase):
    def test_combines_exclusive_outcomes(self) -> None:
        probabilities = double_chance_probabilities(.45, .25, .30)
        self.assertAlmostEqual(probabilities["1X"], .70)
        self.assertAlmostEqual(probabilities["X2"], .55)
        self.assertAlmostEqual(probabilities["12"], .75)

    def test_requires_real_quotes_from_two_bookmakers(self) -> None:
        def book(title: str, price: float) -> dict:
            return {"title": title, "markets": [{
                "key": "double_chance", "outcomes": [
                    {"name": "Home or Draw", "price": price},
                    {"name": "Away or Draw", "price": 1.40},
                ],
            }]}

        prices = best_double_chance_prices(
            [book("A", 1.51), book("B", 1.54)], "Home", "Away"
        )
        self.assertEqual(prices["1X"], ("B", 1.54))
        self.assertNotIn("12", prices)
        self.assertEqual(
            best_double_chance_prices([book("A", 1.51)], "Home", "Away"),
            {},
        )

    def test_settles_all_three_outcomes(self) -> None:
        self.assertTrue(double_chance_won("1X", 1, 1))
        self.assertFalse(double_chance_won("12", 1, 1))
        self.assertTrue(double_chance_won("X2", 0, 2))
        self.assertFalse(double_chance_won("1X", 0, 2))
        self.assertIsNone(double_chance_won("unknown", 1, 0))


if __name__ == "__main__":
    unittest.main()


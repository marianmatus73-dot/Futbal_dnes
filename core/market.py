from __future__ import annotations

from collections import defaultdict


def no_vig_probs(prices: dict[str, float]) -> dict[str, float]:
    inv = {k: 1.0 / v for k, v in prices.items() if v and v > 1.01}
    total = sum(inv.values())

    if total <= 0:
        return {}

    return {k: v / total for k, v in inv.items()}


def consensus_h2h(bookmakers: list[dict], min_books: int = 3) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)

    for book in bookmakers:
        prices: dict[str, float] = {}

        for market in book.get("markets", []):
            if market.get("key") != "h2h":
                continue

            for outcome in market.get("outcomes", []):
                name = str(outcome.get("name", ""))
                price = float(outcome.get("price", 0) or 0)

                if price > 1.01:
                    prices[name] = price

        probs = no_vig_probs(prices)

        for name, prob in probs.items():
            buckets[name].append(prob)

    return {
        name: sum(vals) / len(vals)
        for name, vals in buckets.items()
        if len(vals) >= min_books
    }


def best_outlier_prices(bookmakers: list[dict]) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []

    for book in bookmakers:
        bookmaker = str(book.get("title", ""))

        for market in book.get("markets", []):
            if market.get("key") != "h2h":
                continue

            for outcome in market.get("outcomes", []):
                name = str(outcome.get("name", ""))
                price = float(outcome.get("price", 0) or 0)

                if price > 1.01:
                    rows.append((bookmaker, name, price))

    return rows

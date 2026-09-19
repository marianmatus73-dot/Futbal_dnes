from __future__ import annotations

from typing import Any


def double_chance_probabilities(
    home: float, draw: float, away: float,
) -> dict[str, float]:
    """Combine mutually exclusive full-time 1X2 probabilities."""
    values = [float(home), float(draw), float(away)]
    total = sum(values)
    if any(value < 0 for value in values) or total <= 0:
        return {}
    home, draw, away = (value / total for value in values)
    return {"1X": home + draw, "X2": draw + away, "12": home + away}


def best_double_chance_prices(
    bookmakers: list[dict[str, Any]],
    home_team: str,
    away_team: str,
    *,
    min_books: int = 2,
) -> dict[str, tuple[str, float]]:
    """Return real quoted prices; never synthesize a double-chance price."""
    allowed = {
        frozenset((home_team.casefold(), "draw")): "1X",
        frozenset((away_team.casefold(), "draw")): "X2",
        frozenset((home_team.casefold(), away_team.casefold())): "12",
    }
    best: dict[str, tuple[str, float]] = {}
    contributors: dict[str, set[str]] = {}
    for book in bookmakers:
        bookmaker = str(book.get("title") or book.get("key") or "").strip()
        if not bookmaker:
            continue
        for market in book.get("markets", []):
            if market.get("key") != "double_chance":
                continue
            for outcome in market.get("outcomes", []):
                name = str(outcome.get("name", ""))
                sides = frozenset(part.strip().casefold() for part in name.split(" or "))
                selection = allowed.get(sides)
                try:
                    price = float(outcome.get("price", 0))
                except (TypeError, ValueError):
                    continue
                if not selection or price <= 1.01:
                    continue
                contributors.setdefault(selection, set()).add(bookmaker)
                if selection not in best or price > best[selection][1]:
                    best[selection] = (bookmaker, price)
    return {
        selection: quote
        for selection, quote in best.items()
        if len(contributors.get(selection, set())) >= min_books
    }


def double_chance_won(selection: str, home_goals: int, away_goals: int) -> bool | None:
    if selection not in {"1X", "X2", "12"}:
        return None
    result = "1" if home_goals > away_goals else "2" if away_goals > home_goals else "X"
    return result in selection


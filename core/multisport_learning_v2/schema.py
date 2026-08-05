from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from core.sqlite_helpers import connect, table_columns


SUPPORTED_SPORTS = (
    "baseball",
    "basketball",
    "tennis",
    "hockey",
    "mma",
    "nfl",
)


@dataclass(frozen=True)
class SportBetSchema:
    sport: str
    result: str
    stake: str | None
    odds: str | None
    profit: str | None
    model_probability: str | None
    market_probability: str | None
    confidence: str | None
    league: str | None
    bookmaker: str | None


def first_present(
    available: set[str],
    candidates: Iterable[str],
) -> str | None:
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def detect_schema(database: str | Path) -> SportBetSchema:
    with connect(database) as conn:
        available = set(table_columns(conn, "sport_bets"))

    sport = first_present(
        available,
        ("sport", "sport_name", "category"),
    )
    result = first_present(
        available,
        ("result", "bet_result", "outcome", "status"),
    )

    missing: list[str] = []
    if sport is None:
        missing.append("sport")
    if result is None:
        missing.append("result")

    if missing:
        raise RuntimeError(
            "sport_bets is missing required columns: "
            + ", ".join(missing)
        )

    return SportBetSchema(
        sport=sport,
        result=result,
        stake=first_present(
            available,
            ("stake", "stake_amount", "amount"),
        ),
        odds=first_present(
            available,
            ("odds", "price", "decimal_odds"),
        ),
        profit=first_present(
            available,
            ("profit", "profit_loss", "pnl"),
        ),
        model_probability=first_present(
            available,
            (
                "model_probability",
                "probability",
                "predicted_probability",
            ),
        ),
        market_probability=first_present(
            available,
            ("market_probability", "implied_probability"),
        ),
        confidence=first_present(
            available,
            ("confidence", "confidence_score"),
        ),
        league=first_present(
            available,
            ("league", "competition", "sport_key"),
        ),
        bookmaker=first_present(
            available,
            ("bookmaker", "sportsbook", "book"),
        ),
    )

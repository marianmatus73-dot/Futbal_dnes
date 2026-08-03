from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


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
    table: str
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
    created_at: str | None
    settled_at: str | None


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone() is not None


def columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if not table_exists(conn, table):
        return set()
    return {
        str(row[1])
        for row in conn.execute(
            f"PRAGMA table_info({quote_identifier(table)})"
        ).fetchall()
    }


def first_present(available: set[str], candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def detect_sport_bets_schema(
    database: str | Path,
    table: str = "sport_bets",
) -> SportBetSchema:
    with sqlite3.connect(database) as conn:
        available = columns(conn, table)

    required = {
        "sport": first_present(
            available,
            ("sport", "sport_name", "category"),
        ),
        "result": first_present(
            available,
            ("result", "status", "bet_result", "outcome"),
        ),
    }

    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RuntimeError(
            f"{table} is missing required columns: {', '.join(missing)}"
        )

    return SportBetSchema(
        table=table,
        sport=required["sport"],
        result=required["result"],
        stake=first_present(
            available,
            ("stake", "stake_amount", "bet_amount", "amount"),
        ),
        odds=first_present(
            available,
            ("odds", "price", "decimal_odds"),
        ),
        profit=first_present(
            available,
            ("profit", "profit_loss", "pnl", "net_profit"),
        ),
        model_probability=first_present(
            available,
            (
                "model_probability",
                "probability",
                "predicted_probability",
                "selection_probability",
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
        created_at=first_present(
            available,
            ("created_at", "placed_at", "event_time"),
        ),
        settled_at=first_present(
            available,
            ("settled_at", "updated_at"),
        ),
    )

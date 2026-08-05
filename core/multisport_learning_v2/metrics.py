from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from core.sqlite_helpers import connect, quote_identifier
from .schema import SportBetSchema


def as_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def expression(column: str | None, alias: str) -> str:
    if column is None:
        return f"NULL AS {quote_identifier(alias)}"

    return (
        f"{quote_identifier(column)} "
        f"AS {quote_identifier(alias)}"
    )


def load_rows(
    database: str | Path,
    schema: SportBetSchema,
    sport: str,
):
    selected = [
        expression(schema.result, "result"),
        expression(schema.stake, "stake"),
        expression(schema.odds, "odds"),
        expression(schema.profit, "profit"),
        expression(
            schema.model_probability,
            "model_probability",
        ),
        expression(
            schema.market_probability,
            "market_probability",
        ),
        expression(schema.confidence, "confidence"),
        expression(schema.league, "league"),
        expression(schema.bookmaker, "bookmaker"),
    ]

    sql = (
        f"SELECT {', '.join(selected)} "
        f"FROM sport_bets "
        f"WHERE LOWER({quote_identifier(schema.sport)})=?"
    )

    with connect(database) as conn:
        return conn.execute(
            sql,
            (sport.lower(),),
        ).fetchall()


def calculate(rows) -> dict[str, Any]:
    wins = 0
    losses = 0
    opened = 0
    profit = 0.0
    stake_sum = 0.0
    brier = 0.0
    log_loss = 0.0
    probability_samples = 0
    confidence_values: list[float] = []
    leagues: set[str] = set()
    bookmakers: set[str] = set()

    for row in rows:
        result = str(row["result"] or "").upper()

        if result in {"WIN", "WON", "SUCCESS"}:
            normalized = "WIN"
            wins += 1
        elif result in {"LOSS", "LOST", "FAIL"}:
            normalized = "LOSS"
            losses += 1
        else:
            normalized = "OPEN"
            opened += 1

        stake = as_float(row["stake"])
        odds = as_float(row["odds"])
        explicit_profit = as_float(row["profit"])

        if stake is not None:
            stake_sum += stake

        if explicit_profit is not None:
            profit += explicit_profit
        elif (
            normalized == "WIN"
            and stake is not None
            and odds is not None
        ):
            profit += stake * max(odds - 1.0, 0.0)
        elif normalized == "LOSS" and stake is not None:
            profit -= stake

        probability = as_float(row["model_probability"])
        if (
            probability is not None
            and normalized in {"WIN", "LOSS"}
        ):
            probability = min(
                max(probability, 1e-6),
                1 - 1e-6,
            )
            target = 1.0 if normalized == "WIN" else 0.0
            brier += (probability - target) ** 2
            log_loss += -(
                target * math.log(probability)
                + (1 - target)
                * math.log(1 - probability)
            )
            probability_samples += 1

        confidence = as_float(row["confidence"])
        if confidence is not None:
            confidence_values.append(
                confidence / 100
                if confidence > 1
                else confidence
            )

        if row["league"]:
            leagues.add(str(row["league"]))

        if row["bookmaker"]:
            bookmakers.add(str(row["bookmaker"]))

    settled = wins + losses
    win_rate = wins / settled if settled else None
    yield_value = profit / stake_sum if stake_sum else None

    maturity = (
        "READY"
        if settled >= 150
        else "DEVELOPING"
        if settled >= 50
        else "EARLY"
        if settled > 0
        else "EMPTY"
    )

    quality = round(
        (
            min(settled / 150, 1.0)
            + (1.0 if probability_samples else 0.0)
            + (1.0 if stake_sum else 0.0)
            + min(len(leagues) / 5, 1.0)
        )
        / 4,
        4,
    )

    return {
        "total_bets": len(rows),
        "settled_bets": settled,
        "open_bets": opened,
        "wins": wins,
        "losses": losses,
        "win_rate": (
            round(win_rate, 4)
            if win_rate is not None
            else None
        ),
        "profit": round(profit, 4),
        "stake_sum": round(stake_sum, 4),
        "yield": (
            round(yield_value, 4)
            if yield_value is not None
            else None
        ),
        "probability_samples": probability_samples,
        "brier_score": (
            round(brier / probability_samples, 6)
            if probability_samples
            else None
        ),
        "log_loss": (
            round(log_loss / probability_samples, 6)
            if probability_samples
            else None
        ),
        "average_confidence": (
            round(
                sum(confidence_values)
                / len(confidence_values),
                4,
            )
            if confidence_values
            else None
        ),
        "leagues": len(leagues),
        "bookmakers": len(bookmakers),
        "data_quality": quality,
        "maturity": maturity,
        "ai_health": (
            "GOOD"
            if maturity == "READY" and quality >= 0.75
            else "BUILDING"
            if settled
            else "EMPTY"
        ),
    }

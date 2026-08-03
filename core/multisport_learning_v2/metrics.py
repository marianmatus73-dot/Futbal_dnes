from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Any

from .schema import SportBetSchema, quote_identifier


WIN_VALUES = {"WIN", "WON", "SUCCESS"}
LOSS_VALUES = {"LOSS", "LOST", "FAIL"}
OPEN_VALUES = {"OPEN", "PENDING", "UNSETTLED", ""}


def _normalized_result(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if raw in WIN_VALUES:
        return "WIN"
    if raw in LOSS_VALUES:
        return "LOSS"
    return "OPEN"


def _to_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _select_expression(column: str | None, alias: str) -> str:
    if column is None:
        return f"NULL AS {quote_identifier(alias)}"
    return f"{quote_identifier(column)} AS {quote_identifier(alias)}"


def load_sport_rows(
    database: str | Path,
    schema: SportBetSchema,
    sport: str,
) -> list[sqlite3.Row]:
    selected = [
        _select_expression(schema.result, "result"),
        _select_expression(schema.stake, "stake"),
        _select_expression(schema.odds, "odds"),
        _select_expression(schema.profit, "profit"),
        _select_expression(schema.model_probability, "model_probability"),
        _select_expression(schema.market_probability, "market_probability"),
        _select_expression(schema.confidence, "confidence"),
        _select_expression(schema.league, "league"),
        _select_expression(schema.bookmaker, "bookmaker"),
    ]
    sql = (
        f"SELECT {', '.join(selected)} "
        f"FROM {quote_identifier(schema.table)} "
        f"WHERE LOWER({quote_identifier(schema.sport)})=?"
    )

    with sqlite3.connect(database) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(sql, (sport.lower(),)).fetchall()


def calculate_metrics(rows: list[sqlite3.Row]) -> dict[str, Any]:
    total = len(rows)
    wins = 0
    losses = 0
    open_count = 0
    stake_sum = 0.0
    profit_sum = 0.0
    known_profit = 0
    brier_total = 0.0
    log_loss_total = 0.0
    probability_samples = 0
    confidence_values: list[float] = []
    leagues: set[str] = set()
    bookmakers: set[str] = set()

    for row in rows:
        result = _normalized_result(row["result"])
        if result == "WIN":
            wins += 1
        elif result == "LOSS":
            losses += 1
        else:
            open_count += 1

        stake = _to_float(row["stake"])
        odds = _to_float(row["odds"])
        explicit_profit = _to_float(row["profit"])

        if stake is not None:
            stake_sum += stake

        if explicit_profit is not None:
            profit_sum += explicit_profit
            known_profit += 1
        elif result in {"WIN", "LOSS"} and stake is not None:
            if result == "WIN" and odds is not None and odds > 1:
                profit_sum += stake * (odds - 1.0)
                known_profit += 1
            elif result == "LOSS":
                profit_sum -= stake
                known_profit += 1

        probability = _to_float(row["model_probability"])
        if probability is not None and result in {"WIN", "LOSS"}:
            probability = min(max(probability, 1e-6), 1 - 1e-6)
            target = 1.0 if result == "WIN" else 0.0
            brier_total += (probability - target) ** 2
            log_loss_total += -(
                target * math.log(probability)
                + (1.0 - target) * math.log(1.0 - probability)
            )
            probability_samples += 1

        confidence = _to_float(row["confidence"])
        if confidence is not None:
            if confidence > 1.0:
                confidence /= 100.0
            confidence_values.append(confidence)

        if row["league"]:
            leagues.add(str(row["league"]))
        if row["bookmaker"]:
            bookmakers.add(str(row["bookmaker"]))

    settled = wins + losses
    win_rate = wins / settled if settled else None
    yield_value = profit_sum / stake_sum if stake_sum > 0 else None

    if settled >= 150:
        maturity = "READY"
    elif settled >= 50:
        maturity = "DEVELOPING"
    elif settled > 0:
        maturity = "EARLY"
    else:
        maturity = "EMPTY"

    quality_components = [
        min(settled / 150.0, 1.0),
        1.0 if probability_samples else 0.0,
        1.0 if known_profit else 0.0,
        min(len(leagues) / 5.0, 1.0),
    ]
    data_quality = round(sum(quality_components) / len(quality_components), 4)

    health = (
        "GOOD"
        if maturity == "READY" and data_quality >= 0.75
        else "BUILDING"
        if settled > 0
        else "EMPTY"
    )

    return {
        "total_bets": total,
        "settled_bets": settled,
        "open_bets": open_count,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "profit": round(profit_sum, 4),
        "stake_sum": round(stake_sum, 4),
        "yield": round(yield_value, 4) if yield_value is not None else None,
        "probability_samples": probability_samples,
        "brier_score": (
            round(brier_total / probability_samples, 6)
            if probability_samples
            else None
        ),
        "log_loss": (
            round(log_loss_total / probability_samples, 6)
            if probability_samples
            else None
        ),
        "average_confidence": (
            round(sum(confidence_values) / len(confidence_values), 4)
            if confidence_values
            else None
        ),
        "leagues": len(leagues),
        "bookmakers": len(bookmakers),
        "data_quality": data_quality,
        "maturity": maturity,
        "ai_health": health,
    }

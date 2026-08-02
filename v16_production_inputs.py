"""
Production input adapter for the integrated V16 cycle.

Reads the latest real settled result from SQLite and derives runtime health
from the actual sport-module execution results.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class V16ProductionInputs:
    previous_result: str | None
    previous_profit: float
    runtime_health: float
    latency_ms: int
    execution_ready: bool
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "previous_result": self.previous_result,
            "previous_profit": self.previous_profit,
            "runtime_health": self.runtime_health,
            "latency_ms": self.latency_ms,
            "execution_ready": self.execution_ready,
            "source": self.source,
        }


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {
            str(row[1])
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
    except sqlite3.Error:
        return set()


def _latest_settled_feedback(database: Path) -> tuple[str | None, float]:
    if not database.exists():
        return None, 0.0

    with sqlite3.connect(database) as conn:
        columns = _table_columns(conn, "sport_bets")
        if not columns or "result" not in columns:
            return None, 0.0

        order_column = next(
            (
                name
                for name in ("settled_at", "updated_at", "created_at", "id")
                if name in columns
            ),
            "rowid",
        )

        selected = ["result"]
        for name in ("profit", "profit_loss", "pnl", "stake", "odds"):
            if name in columns:
                selected.append(name)

        sql = (
            f"SELECT {', '.join(selected)} FROM sport_bets "
            "WHERE UPPER(COALESCE(result, '')) IN ('WON', 'WIN', 'LOST', 'LOSS') "
            f"ORDER BY {order_column} DESC LIMIT 1"
        )
        row = conn.execute(sql).fetchone()
        if row is None:
            return None, 0.0

        values = dict(zip(selected, row))
        raw_result = str(values.get("result", "")).upper()
        result = "WIN" if raw_result in {"WON", "WIN"} else "LOSS"

        for key in ("profit", "profit_loss", "pnl"):
            value = values.get(key)
            if value not in (None, ""):
                try:
                    return result, float(value)
                except (TypeError, ValueError):
                    pass

        try:
            stake = float(values.get("stake") or 0.0)
            odds = float(values.get("odds") or 0.0)
            derived_profit = stake * (odds - 1.0) if result == "WIN" else -stake
            return result, round(derived_profit, 4)
        except (TypeError, ValueError):
            return result, 0.0


def build_production_inputs(
    *,
    database: str | Path,
    module_outputs: list[dict[str, Any]],
) -> V16ProductionInputs:
    total = len(module_outputs)
    successful = sum(1 for item in module_outputs if item.get("ok"))
    runtime_health = successful / total if total else 0.0

    durations = [
        float(item.get("duration_sec", 0.0))
        for item in module_outputs
        if item.get("duration_sec") is not None
    ]
    average_duration = sum(durations) / len(durations) if durations else 0.0
    latency_ms = max(1, round(average_duration * 1000))

    previous_result, previous_profit = _latest_settled_feedback(Path(database))

    return V16ProductionInputs(
        previous_result=previous_result,
        previous_profit=previous_profit,
        runtime_health=round(runtime_health, 4),
        latency_ms=latency_ms,
        execution_ready=successful > 0 and runtime_health >= 0.5,
        source="sqlite:sport_bets+runtime:module_outputs",
    )

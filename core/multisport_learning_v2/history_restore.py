from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from core.db_bootstrap import bootstrap_database
from core.migration_engine import migrate_database
from core.sqlite_helpers import connect, table_columns


ALIASES = {
    "sport_name": "sport",
    "category": "sport",
    "competition": "league",
    "match": "event",
    "pick": "selection",
    "sportsbook": "bookmaker",
    "price": "odds",
    "amount": "stake",
    "bet_result": "result",
    "outcome": "result",
    "profit_loss": "profit",
    "pnl": "profit",
    "probability": "model_probability",
    "predicted_probability": "model_probability",
    "implied_probability": "market_probability",
    "confidence_score": "confidence",
}


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}

    for key, value in row.items():
        if key is None:
            continue
        destination = ALIASES.get(key, key)
        normalized[destination] = value

    result = str(normalized.get("result", "OPEN") or "OPEN").upper()
    if result in {"WON", "WIN", "SUCCESS"}:
        normalized["result"] = "WIN"
    elif result in {"LOST", "LOSS", "FAIL"}:
        normalized["result"] = "LOSS"
    else:
        normalized["result"] = "OPEN"

    return normalized


def restore_history(
    database: str | Path,
    csv_path: str | Path,
) -> dict[str, Any]:
    bootstrap_database(database)
    migrate_database(database)

    source = Path(csv_path)
    if not source.exists():
        return {
            "database": str(Path(database)),
            "csv": str(source),
            "imported": 0,
            "status": "CSV_NOT_FOUND",
        }

    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = [normalize_row(row) for row in csv.DictReader(handle)]

    if not rows:
        return {
            "database": str(Path(database)),
            "csv": str(source),
            "imported": 0,
            "status": "EMPTY",
        }

    with connect(database) as conn:
        available = set(table_columns(conn, "sport_bets"))
        imported = 0

        for row in rows:
            filtered = {
                key: value
                for key, value in row.items()
                if key in available and key != "id"
            }

            if not filtered.get("sport"):
                continue

            columns = list(filtered)
            placeholders = ",".join("?" for _ in columns)
            column_sql = ",".join(f'"{column}"' for column in columns)

            conn.execute(
                f"INSERT INTO sport_bets "
                f"({column_sql}) VALUES ({placeholders})",
                [filtered[column] for column in columns],
            )
            imported += 1

        conn.commit()

    return {
        "database": str(Path(database)),
        "csv": str(source),
        "imported": imported,
        "status": "READY",
    }

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.db_bootstrap import bootstrap_database
from core.sqlite_helpers import connect, table_columns


REQUIRED_COLUMNS = {
    "sport": "TEXT",
    "league": "TEXT",
    "event": "TEXT",
    "selection": "TEXT",
    "bookmaker": "TEXT",
    "odds": "REAL",
    "stake": "REAL",
    "model_probability": "REAL",
    "market_probability": "REAL",
    "confidence": "REAL",
    "result": "TEXT NOT NULL DEFAULT 'OPEN'",
    "profit": "REAL",
    "created_at": "TEXT",
    "settled_at": "TEXT",
}


def migrate_database(database: str | Path) -> dict[str, Any]:
    bootstrap_database(database)
    added: list[str] = []

    with connect(database) as conn:
        existing = set(table_columns(conn, "sport_bets"))

        for column, column_type in REQUIRED_COLUMNS.items():
            if column not in existing:
                conn.execute(
                    f'ALTER TABLE sport_bets '
                    f'ADD COLUMN "{column}" {column_type}'
                )
                added.append(column)

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sport_bets_sport "
            "ON sport_bets(sport)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sport_bets_result "
            "ON sport_bets(result)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sport_bets_sport_result "
            "ON sport_bets(sport, result)"
        )
        conn.commit()

    return {
        "database": str(Path(database)),
        "columns_added": added,
        "status": "READY",
    }

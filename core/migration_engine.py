from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

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

def _connect(database: str | Path) -> sqlite3.Connection:
    path = Path(database)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    }

def migrate_database(database: str | Path) -> dict[str, Any]:
    path = Path(database)
    with _connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sport_bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sport TEXT NOT NULL,
                league TEXT,
                event TEXT,
                selection TEXT,
                bookmaker TEXT,
                odds REAL,
                stake REAL,
                model_probability REAL,
                market_probability REAL,
                confidence REAL,
                result TEXT NOT NULL DEFAULT 'OPEN',
                profit REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                settled_at TEXT
            )
            """
        )
        existing = _columns(conn, "sport_bets")
        added: list[str] = []
        for column, column_type in REQUIRED_COLUMNS.items():
            if column not in existing:
                conn.execute(
                    f'ALTER TABLE sport_bets ADD COLUMN "{column}" {column_type}'
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
        rows = int(conn.execute("SELECT COUNT(*) FROM sport_bets").fetchone()[0])
    return {
        "database": str(path),
        "columns_added": added,
        "rows": rows,
        "status": "READY",
    }

from __future__ import annotations

from pathlib import Path

from core.sqlite_helpers import connect


SPORT_BETS_SCHEMA = """
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
);

CREATE INDEX IF NOT EXISTS idx_sport_bets_sport
ON sport_bets(sport);

CREATE INDEX IF NOT EXISTS idx_sport_bets_result
ON sport_bets(result);

CREATE INDEX IF NOT EXISTS idx_sport_bets_sport_result
ON sport_bets(sport, result);
"""


def bootstrap_database(database: str | Path) -> dict:
    path = Path(database)
    with connect(path) as conn:
        conn.executescript(SPORT_BETS_SCHEMA)
        count = conn.execute(
            "SELECT COUNT(*) FROM sport_bets"
        ).fetchone()[0]
        conn.commit()

    return {
        "database": str(path),
        "table": "sport_bets",
        "rows": int(count),
        "status": "READY",
    }

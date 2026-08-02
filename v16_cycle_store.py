"""
Persistent SQLite history for V16 autonomous cycles.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS v16_cycle_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    cycle_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    stages_completed INTEGER NOT NULL,
    loop_state TEXT,
    loop_score REAL,
    decision TEXT,
    monitor_score REAL,
    errors_count INTEGER NOT NULL,
    previous_result TEXT,
    previous_profit REAL,
    runtime_health REAL,
    latency_ms INTEGER,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_v16_cycle_created_at
ON v16_cycle_history(created_at);
"""


def save_cycle(
    database: str | Path,
    result: dict[str, Any],
    inputs: dict[str, Any],
) -> int:
    path = Path(database)
    path.parent.mkdir(parents=True, exist_ok=True)

    stages = result.get("stages", {})
    loop = stages.get("v16_16_loop", {})
    monitor = stages.get("v16_15_monitor", {})

    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA)
        cursor = conn.execute(
            """
            INSERT INTO v16_cycle_history (
                created_at, cycle_id, status, stages_completed,
                loop_state, loop_score, decision, monitor_score,
                errors_count, previous_result, previous_profit,
                runtime_health, latency_ms, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                int(result.get("cycle_id", 0)),
                str(result.get("status", "UNKNOWN")),
                int(result.get("stages_completed", 0)),
                loop.get("loop_state"),
                loop.get("loop_score"),
                loop.get("decision"),
                monitor.get("monitor_score"),
                len(result.get("errors", [])),
                inputs.get("previous_result"),
                float(inputs.get("previous_profit", 0.0)),
                float(inputs.get("runtime_health", 0.0)),
                int(inputs.get("latency_ms", 0)),
                json.dumps(result, ensure_ascii=False, default=str),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def recent_cycles(database: str | Path, limit: int = 30) -> list[dict[str, Any]]:
    path = Path(database)
    if not path.exists():
        return []

    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT created_at, cycle_id, status, stages_completed,
                   loop_state, loop_score, decision, monitor_score,
                   errors_count, previous_result, previous_profit,
                   runtime_health, latency_ms
            FROM v16_cycle_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        ).fetchall()

    return [dict(row) for row in rows]

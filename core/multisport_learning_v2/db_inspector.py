from __future__ import annotations

from pathlib import Path
from typing import Any

from core.sqlite_helpers import connect, table_columns, table_exists


def inspect_database(database: str | Path) -> dict[str, Any]:
    path = Path(database)

    if not path.exists():
        return {
            "database": str(path),
            "exists": False,
            "tables": {},
            "status": "MISSING",
        }

    with connect(path) as conn:
        tables = [
            str(row[0])
            for row in conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type='table'
                  AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            ).fetchall()
        ]

        details = {}
        for table in tables:
            details[table] = {
                "columns": table_columns(conn, table),
                "rows": int(
                    conn.execute(
                        f'SELECT COUNT(*) FROM "{table}"'
                    ).fetchone()[0]
                ),
            }

    return {
        "database": str(path),
        "exists": True,
        "tables": details,
        "sport_bets_ready": "sport_bets" in details,
        "status": "READY",
    }

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def validate_database(database: str | Path) -> dict:
    path = Path(database)

    if not path.exists():
        raise SystemExit(f"FAILED: database does not exist: {path}")

    with sqlite3.connect(path) as conn:
        table = conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type='table' AND name='sport_bets'
            """
        ).fetchone()

        if table is None:
            raise SystemExit("FAILED: sport_bets table is missing.")

        columns = {
            str(row[1])
            for row in conn.execute(
                "PRAGMA table_info(sport_bets)"
            ).fetchall()
        }

        missing = sorted({"sport", "result"} - columns)
        if missing:
            raise SystemExit(
                "FAILED: sport_bets missing columns: "
                + ", ".join(missing)
            )

        rows = int(
            conn.execute(
                "SELECT COUNT(*) FROM sport_bets"
            ).fetchone()[0]
        )

    result = {
        "database": str(path),
        "table": "sport_bets",
        "rows": rows,
        "required_columns": ["sport", "result"],
        "status": "READY",
    }

    print(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="bets.db")
    args = parser.parse_args()
    validate_database(args.database)


if __name__ == "__main__":
    main()

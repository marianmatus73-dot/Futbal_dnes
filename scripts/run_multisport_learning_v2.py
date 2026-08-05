from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
from pathlib import Path
from pprint import pprint
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from core.migration_engine import migrate_database
from core.multisport_learning_v2.manager import (
    MultisportLearningV2Manager,
)


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


def normalize_result(value: Any) -> str:
    raw = str(value or "OPEN").strip().upper()

    if raw in {"WON", "WIN", "SUCCESS"}:
        return "WIN"

    if raw in {"LOST", "LOSS", "FAIL"}:
        return "LOSS"

    return "OPEN"


def restore_history(
    database: Path,
    csv_path: Path,
) -> dict[str, Any]:
    if not csv_path.exists():
        return {
            "csv": str(csv_path),
            "imported": 0,
            "status": "CSV_NOT_FOUND",
        }

    with sqlite3.connect(database) as conn:
        available = {
            str(row[1])
            for row in conn.execute(
                "PRAGMA table_info(sport_bets)"
            ).fetchall()
        }

        with csv_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            reader = csv.DictReader(handle)
            imported = 0

            for raw_row in reader:
                normalized: dict[str, Any] = {}

                for key, value in raw_row.items():
                    if key is None:
                        continue

                    destination = ALIASES.get(key, key)
                    normalized[destination] = value

                normalized["result"] = normalize_result(
                    normalized.get("result")
                )

                filtered = {
                    key: value
                    for key, value in normalized.items()
                    if key in available and key != "id"
                }

                if not filtered.get("sport"):
                    continue

                columns = list(filtered)
                placeholders = ",".join(
                    "?"
                    for _ in columns
                )
                column_sql = ",".join(
                    f'"{column}"'
                    for column in columns
                )

                conn.execute(
                    f"INSERT INTO sport_bets "
                    f"({column_sql}) "
                    f"VALUES ({placeholders})",
                    [
                        filtered[column]
                        for column in columns
                    ],
                )
                imported += 1

            conn.commit()

    return {
        "csv": str(csv_path),
        "imported": imported,
        "status": "READY",
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--database",
        default=os.getenv("DB_FILE", "bets.db"),
    )

    parser.add_argument(
        "--history-csv",
        default=os.getenv(
            "MULTISPORT_V2_HISTORY_CSV",
            "exports/history_sport_bets.csv",
        ),
    )

    parser.add_argument(
        "--export-dir",
        default=os.getenv(
            "EXPORT_DIR",
            "exports",
        ),
    )

    args = parser.parse_args()

    database = Path(args.database)
    history_csv = Path(args.history_csv)

    print("=== MULTISPORT LEARNING V2.1 PRODUCTION ===")

    pprint(migrate_database(database))
    pprint(restore_history(database, history_csv))

    manager = MultisportLearningV2Manager(database)

    result = manager.run_all(
        export_dir=args.export_dir,
    )

    pprint(result)


if __name__ == "__main__":
    main()

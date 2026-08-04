from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from pprint import pprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import Settings
from core.multisport_learning_v2.manager import MultisportLearningV2Manager


def table_columns(database: Path, table: str) -> set[str]:
    if not database.exists():
        return set()

    with sqlite3.connect(database) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if exists is None:
            return set()

        return {
            str(row[1])
            for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        }


def bootstrap_database(database: Path, table: str) -> None:
    """
    Initialize the normal project database and restore CSV history before V2
    runs on a fresh GitHub Actions runner.
    """
    columns = table_columns(database, table)
    if columns:
        return

    print(
        f"[V2 bootstrap] {table!r} is missing or empty in {database}. "
        "Initializing project tables and restoring history..."
    )

    # Importing main is safe: asyncio.run(run()) is protected by __main__.
    from main import restore_learning_history

    settings = Settings.from_env()
    settings.db_file = str(database)
    restore_learning_history(settings)

    columns = table_columns(database, table)
    if not columns:
        raise RuntimeError(
            f"Bootstrap finished, but table {table!r} still does not exist in "
            f"{database}. Check that core.sport_quant.init_sport_db is present "
            "and that the workflow checked out the exports/history_*.csv files."
        )

    print(
        f"[V2 bootstrap] Ready: {table} has {len(columns)} columns "
        f"in {database}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Multisport Learning Bundle V2."
    )
    parser.add_argument("--database", default="bets.db")
    parser.add_argument("--table", default="sport_bets")
    parser.add_argument("--export-dir", default="exports")
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Do not initialize/restore the database when sport_bets is absent.",
    )
    args = parser.parse_args()

    database = Path(args.database)

    if not args.no_bootstrap:
        bootstrap_database(database, args.table)

    manager = MultisportLearningV2Manager(
        database,
        table=args.table,
    )

    print("=== MULTISPORT LEARNING BUNDLE V2 ===")
    pprint(manager.run_all(export_dir=args.export_dir))


if __name__ == "__main__":
    main()

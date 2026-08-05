from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db_inspector import inspect_database


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="bets.db")
    args = parser.parse_args()

    result = inspect_database(args.database)
    sport_bets = result.get("tables", {}).get("sport_bets")

    if not sport_bets:
        raise SystemExit("FAILED: sport_bets table is missing.")

    required = {"sport", "result"}
    columns = set(sport_bets.get("columns", []))
    missing = sorted(required - columns)

    if missing:
        raise SystemExit(
            "FAILED: sport_bets missing columns: "
            + ", ".join(missing)
        )

    print(
        "READY: sport_bets exists with "
        f"{sport_bets.get('rows', 0)} rows."
    )

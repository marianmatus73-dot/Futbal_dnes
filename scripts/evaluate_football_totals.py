"""Read-only evaluation of archived football Under/Over 2.5 tips."""

from __future__ import annotations

import argparse
import json

from core.football_totals_backtest import evaluate_totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="exports/history_sport_bets.csv", help="SQLite database or exported sport-bet CSV")
    parser.add_argument("--min-samples", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(evaluate_totals(args.db, max(1, args.min_samples)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


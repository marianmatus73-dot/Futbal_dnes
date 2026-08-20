from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import date
from pathlib import Path

from core.config import Settings
from core.sport_context import SportContextDatabase
from core.sportmonks import SportmonksClient, sync_upcoming_context


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely sync verified Sportmonks football context")
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--output", default="exports/sportmonks_sync_summary.json")
    args = parser.parse_args()

    token = os.getenv("SPORTMONKS_API_TOKEN", "").strip()
    if not token:
        raise SystemExit("SPORTMONKS_API_TOKEN is missing")
    settings = Settings.from_env()
    database = SportContextDatabase(settings)
    database.init_db()
    summary = sync_upcoming_context(
        SportmonksClient(token, timeout=float(os.getenv("HTTP_TIMEOUT", "30"))),
        database,
        start_date=args.start_date,
        days=max(1, min(args.days, 7)),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


from __future__ import annotations

import csv
import json
from pathlib import Path
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_csv_snapshots_v15_26(
    source="exports/history_football_market_snapshots_v14.csv"
):
    path = Path(source)

    if not path.exists():
        return []

    rows = []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(dict(row))

    return rows


def run_csv_snapshot_loader_v15_26(
    source="exports/history_football_market_snapshots_v14.csv"
):
    snapshots = load_csv_snapshots_v15_26(source)

    report = {
        "version": "v15.26",
        "created_at": _now(),
        "source": source,
        "snapshots_loaded": len(snapshots),
        "status": "READY" if snapshots else "BUILDING",
    }

    Path("exports").mkdir(exist_ok=True)
    Path("exports/football_csv_snapshot_loader_v15_26.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return report, snapshots

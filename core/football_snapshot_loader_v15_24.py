from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_market_snapshots_v15_24(source="exports/football_market_snapshots_v14.json"):
    path = Path(source)

    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("snapshots", "data", "rows", "items"):
            if isinstance(data.get(key), list):
                return data[key]

    return []


def run_snapshot_loader_v15_24(source="exports/football_market_snapshots_v14.json"):
    snapshots = load_market_snapshots_v15_24(source)

    report = {
        "version": "v15.24",
        "created_at": _now(),
        "source": source,
        "snapshots_loaded": len(snapshots),
        "status": "READY" if snapshots else "BUILDING",
    }

    Path("exports").mkdir(exist_ok=True)
    Path("exports/football_snapshot_loader_v15_24.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return report, snapshots

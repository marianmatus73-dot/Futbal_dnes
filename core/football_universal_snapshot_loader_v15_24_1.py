from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def find_snapshot_sources():
    patterns = [
        "exports/football_market_snapshots_v14.json",
        "exports/*snapshot*.json",
        "data/*snapshot*.json",
        "*snapshot*.json",
    ]

    files = []
    for pattern in patterns:
        files.extend(Path(".").glob(pattern))

    return list(dict.fromkeys(files))


def load_snapshot_file(path: Path):
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


def run_universal_snapshot_loader_v15_24_1():
    sources = find_snapshot_sources()
    snapshots = []

    used_sources = []

    for source in sources:
        rows = load_snapshot_file(source)
        if rows:
            snapshots.extend(rows)
            used_sources.append(str(source))

    report = {
        "version": "v15.24.1",
        "created_at": _now(),
        "sources_checked": len(sources),
        "sources_used": used_sources,
        "snapshots_loaded": len(snapshots),
        "status": "READY" if snapshots else "BUILDING",
    }

    Path("exports").mkdir(exist_ok=True)
    Path("exports/football_universal_snapshot_loader_v15_24_1.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return report, snapshots

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_real_snapshot_parser_v15_22(
    *,
    snapshots: int = 0,
    parsed_snapshots: int = 0,
    fingerprints_created: int = 0,
    join_ready: int = 0,
):
    parsed = round(parsed_snapshots / snapshots * 100, 1) if snapshots else 0.0
    ready = round(join_ready / snapshots * 100, 1) if snapshots else 0.0

    report = {
        "version": "v15.22",
        "created_at": _now(),
        "snapshots": snapshots,
        "parsed_snapshots": parsed_snapshots,
        "fingerprints_created": fingerprints_created,
        "join_ready": join_ready,
        "parsed_percent": parsed,
        "join_ready_percent": ready,
        "status": "READY" if ready >= 90 else "BUILDING",
        "blockers": [] if join_ready else [
            "Real snapshot rows still need parsing"
        ],
    }

    Path("exports").mkdir(exist_ok=True)
    Path("exports/football_real_snapshot_parser_v15_22.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return report

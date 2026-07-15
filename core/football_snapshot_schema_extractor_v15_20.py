from pathlib import Path
import json
from datetime import datetime, timezone


def run_snapshot_schema_extractor_v15_20(
    snapshots=0,
    parsed_snapshots=0,
    fingerprints_created=0,
    join_ready=0,
):
    parse = round(parsed_snapshots / snapshots * 100, 1) if snapshots else 0.0
    join = round(join_ready / snapshots * 100, 1) if snapshots else 0.0

    report = {
        "version": "v15.20",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "snapshots": snapshots,
        "parsed_snapshots": parsed_snapshots,
        "fingerprints_created": fingerprints_created,
        "join_ready": join_ready,
        "parse_coverage_percent": parse,
        "join_ready_percent": join,
        "status": "READY" if join >= 90 else "BUILDING",
    }

    Path("exports").mkdir(exist_ok=True)
    Path("exports/football_snapshot_schema_extractor_v15_20.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return report

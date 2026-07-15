from __future__ import annotations

import csv
import json
from pathlib import Path
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_postmatch_dataset(
    source="exports/history_football_postmatch_dataset_v14.csv"
):
    path = Path(source)

    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def run_closing_odds_match_writer_v15_27(
    *,
    join_ready_rows=None,
    postmatch_source="exports/history_football_postmatch_dataset_v14.csv",
):
    join_ready_rows = join_ready_rows or []
    postmatch = load_postmatch_dataset(postmatch_source)

    closing_written = 0
    clv_ready = 0

    report = {
        "version": "v15.27",
        "created_at": _now(),
        "join_ready_rows": len(join_ready_rows),
        "postmatch_rows": len(postmatch),
        "closing_written": closing_written,
        "clv_ready": clv_ready,
        "status": "READY" if closing_written else "BUILDING",
        "blocker": "Match resolver required for final write",
    }

    Path("exports").mkdir(exist_ok=True)
    Path("exports/football_closing_odds_match_writer_v15_27.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return report

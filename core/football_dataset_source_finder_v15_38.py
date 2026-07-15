from __future__ import annotations

import csv
import json
from pathlib import Path
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_rows(path):
    try:
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                return list(csv.DictReader(f))

        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]

    except Exception:
        pass

    return []


def norm(value):
    return "".join(
        ch for ch in str(value or "").lower()
        if ch.isalnum()
    )


def row_text(row):
    values = [
        row.get("event"),
        row.get("source_hash"),
        row.get("home_team"),
        row.get("away_team"),
        row.get("match_id"),
        row.get("id"),
    ]

    return " ".join(norm(v) for v in values)


def run_dataset_source_finder(
    target="exports/history_football_postmatch_dataset_v14.csv",
):
    target_rows = read_rows(Path(target))

    targets = [row_text(r) for r in target_rows]

    results = []

    for file in Path("exports").glob("*"):
        if file.name == Path(target).name:
            continue

        rows = read_rows(file)

        if not rows:
            continue

        found = 0

        for target_text in targets:
            for row in rows:
                if target_text and target_text[:20] in row_text(row):
                    found += 1
                    break

        results.append({
            "file": str(file),
            "rows": len(rows),
            "matches_found": found,
        })

    results.sort(
        key=lambda x: x["matches_found"],
        reverse=True,
    )

    report = {
        "version": "v15.38",
        "created_at": now(),
        "target_rows": len(target_rows),
        "sources_checked": len(results),
        "results": results[:20],
        "status": "READY",
    }

    Path("exports/football_dataset_source_finder_v15_38.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return report

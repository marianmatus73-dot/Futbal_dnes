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


def run_closing_source_finder_v15_47(resolved_matches):

    hashes = {
        item.get("postmatch", {}).get("source_hash")
        for item in resolved_matches
        if item.get("postmatch", {}).get("source_hash")
    }

    results = []

    for file in Path("exports").glob("*"):
        rows = read_rows(file)

        if not rows:
            continue

        found = 0
        closing_fields = set()

        for row in rows:
            if row.get("source_hash") in hashes:
                found += 1

                for key, value in row.items():
                    if "close" in key.lower() or "odd" in key.lower():
                        if value not in ("", None):
                            closing_fields.add(key)

        if found:
            results.append({
                "file": str(file),
                "matches_found": found,
                "closing_fields": list(closing_fields),
            })

    results.sort(
        key=lambda x: x["matches_found"],
        reverse=True
    )

    return {
        "version": "v15.47",
        "created_at": now(),
        "target_hashes": len(hashes),
        "sources_found": results,
        "status": "READY",
    }

from __future__ import annotations

from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_real_closing_odds_merge_v15_53(opening_rows, closing_snapshots):
    closing_map = {}

    for row in closing_snapshots:
        key = row.get("source_hash")
        if key:
            closing_map[key] = (
                row.get("selected_odds")
                or row.get("odds")
            )

    output = []

    for row in opening_rows:
        item = dict(row)
        item["closing_odds"] = closing_map.get(
            row.get("source_hash")
        )
        output.append(item)

    return {
        "version": "v15.53",
        "created_at": now(),
        "records": len(output),
        "closing_odds_found": sum(
            1 for r in output if r.get("closing_odds")
        ),
        "status": "READY" if output else "BUILDING",
    }, output

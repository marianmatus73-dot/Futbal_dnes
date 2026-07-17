from __future__ import annotations

from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def find_closing_odds(row):
    fields = [
        "closing_odds",
        "close_odds",
        "final_odds",
        "settled_odds",
        "selected_odds",
        "market_odds",
        "odds",
    ]

    for field in fields:
        value = row.get(field)
        if value not in (None, "", "null"):
            return value

    return None


def run_closing_odds_bridge_v15_46(opening_rows, resolved_matches):

    closing_map = {}

    for item in resolved_matches:
        post = item.get("postmatch", {})
        key = post.get("source_hash")

        if key:
            closing_map[key] = find_closing_odds(post)

    output = []

    for row in opening_rows:
        item = dict(row)
        item["closing_odds"] = closing_map.get(
            row.get("source_hash")
        )
        output.append(item)

    return {
        "version": "v15.46",
        "created_at": now(),
        "records": len(output),
        "closing_odds_found": sum(
            1 for r in output if r.get("closing_odds")
        ),
        "status": "READY" if output else "BUILDING",
    }, output

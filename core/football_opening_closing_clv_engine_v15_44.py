from __future__ import annotations

from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def calculate_clv(opening, closing):
    try:
        opening = float(opening)
        closing = float(closing)

        if opening <= 0 or closing <= 0:
            return None

        opening_prob = 1 / opening
        closing_prob = 1 / closing

        return round(
            ((opening_prob - closing_prob) / closing_prob) * 100,
            4,
        )
    except (TypeError, ValueError):
        return None


def run_clv_engine_v15_44(opening_rows):
    output = []

    for row in opening_rows:
        item = dict(row)

        item["clv_percent"] = calculate_clv(
            item.get("opening_odds"),
            item.get("closing_odds"),
        )

        output.append(item)

    return {
        "version": "v15.44",
        "created_at": now(),
        "records": len(output),
        "clv_ready": sum(
            1 for r in output
            if r.get("clv_percent") is not None
        ),
        "status": "READY" if output else "BUILDING",
    }, output

from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime, timezone

OUTPUT = Path("exports/football_closing_clv_v15_42.csv")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def calculate_clv(rows):
    output = []

    for row in rows:
        opening = row.get("opening_odds")
        closing = row.get("closing_odds")

        try:
            opening = float(opening)
            closing = float(closing)

            opening_prob = 1 / opening if opening > 0 else None
            closing_prob = 1 / closing if closing > 0 else None

            clv = None
            if opening_prob and closing_prob:
                clv = round(
                    ((opening_prob - closing_prob) / closing_prob) * 100,
                    4,
                )

        except (TypeError, ValueError):
            clv = None

        new_row = dict(row)
        new_row["clv_percent"] = clv
        output.append(new_row)

    return output


def run_clv_calculator_v15_42(source="exports/football_closing_records_v15_41.csv"):
    path = Path(source)

    if not path.exists():
        return {
            "version": "v15.42",
            "clv_ready": 0,
            "status": "BUILDING",
        }

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    calculated = calculate_clv(rows)

    OUTPUT.parent.mkdir(exist_ok=True)

    if calculated:
        with OUTPUT.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=calculated[0].keys())
            writer.writeheader()
            writer.writerows(calculated)

    return {
        "version": "v15.42",
        "created_at": now(),
        "closing_rows": len(rows),
        "clv_ready": sum(
            1 for row in calculated
            if row.get("clv_percent") is not None
        ),
        "output": str(OUTPUT),
        "status": "READY" if calculated else "BUILDING",
    }

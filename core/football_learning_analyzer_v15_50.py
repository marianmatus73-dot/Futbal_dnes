from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_learning_analyzer_v15_50(
    source="exports/history_football_clv_results_v15_49.csv"
):
    path = Path(source)

    if not path.exists():
        return {
            "version": "v15.50",
            "records": 0,
            "status": "BUILDING",
        }

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    clvs = []

    for row in rows:
        try:
            clvs.append(float(row.get("clv_percent")))
        except (TypeError, ValueError):
            pass

    positive = [x for x in clvs if x > 0]
    negative = [x for x in clvs if x < 0]

    return {
        "version": "v15.50",
        "created_at": now(),
        "records": len(rows),
        "clv_samples": len(clvs),
        "positive_clv": len(positive),
        "negative_clv": len(negative),
        "average_clv": round(sum(clvs) / len(clvs), 4) if clvs else None,
        "recommendations_ready": bool(clvs),
        "status": "READY" if clvs else "BUILDING",
    }

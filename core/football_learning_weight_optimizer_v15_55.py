from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime, timezone


CURRENT_WEIGHTS = {
    "ELO": 0.25,
    "XG": 0.25,
    "FORM": 0.20,
    "MARKET": 0.20,
    "CONTEXT": 0.10,
}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_learning_weight_optimizer_v15_55(
    source="exports/history_football_clv_results_v15_49.csv"
):
    path = Path(source)

    records = 0

    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            records = len(list(csv.DictReader(f)))

    action = "HOLD"

    if records >= 100:
        action = "ANALYZE_ADJUSTMENT"

    return {
        "version": "v15.55",
        "created_at": now(),
        "records_analyzed": records,
        "current_weights": CURRENT_WEIGHTS,
        "recommended_weights": CURRENT_WEIGHTS,
        "action": action,
        "status": "READY",
    }

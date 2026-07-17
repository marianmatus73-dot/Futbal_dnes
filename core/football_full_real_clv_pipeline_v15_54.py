from __future__ import annotations

from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_full_real_clv_pipeline_v15_54(
    opening_report,
    closing_report,
    clv_report,
    learning_report,
    validator_report,
):
    return {
        "version": "v15.54",
        "created_at": now(),
        "opening": opening_report,
        "closing": closing_report,
        "clv": clv_report,
        "learning": learning_report,
        "validator": validator_report,
        "status": "READY",
    }

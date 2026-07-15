from pathlib import Path
import json
from datetime import datetime, timezone


def run_closing_odds_writer_v15_21(
    matches=0,
    snapshots=0,
    closing_written=0,
    clv_ready=0,
):
    coverage = round(closing_written / matches * 100, 1) if matches else 0.0
    clv = round(clv_ready / matches * 100, 1) if matches else 0.0

    report = {
        "version": "v15.21",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "matches": matches,
        "snapshots": snapshots,
        "closing_written": closing_written,
        "clv_ready": clv_ready,
        "closing_coverage_percent": coverage,
        "clv_coverage_percent": clv,
        "status": "READY" if coverage >= 90 else "BUILDING",
    }

    Path("exports").mkdir(exist_ok=True)
    Path("exports/football_closing_odds_writer_v15_21.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report

"""
Generate JSON, CSV and HTML monitoring artifacts for V16 history.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

from v16_cycle_store import recent_cycles


def build_dashboard(
    database: str | Path,
    export_dir: str | Path = "exports",
    limit: int = 50,
) -> dict[str, Any]:
    rows = recent_cycles(database, limit=limit)
    out = Path(export_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {
        "cycles": len(rows),
        "ready_cycles": sum(1 for row in rows if row["status"] == "READY"),
        "error_cycles": sum(1 for row in rows if row["errors_count"] > 0),
        "average_loop_score": round(
            sum(float(row["loop_score"] or 0.0) for row in rows) / len(rows), 4
        ) if rows else 0.0,
        "average_runtime_health": round(
            sum(float(row["runtime_health"] or 0.0) for row in rows) / len(rows), 4
        ) if rows else 0.0,
        "latest": rows[0] if rows else None,
    }

    json_path = out / "v16_dashboard.json"
    json_path.write_text(
        json.dumps({"summary": summary, "cycles": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path = out / "v16_cycle_history.csv"
    fieldnames = list(rows[0].keys()) if rows else [
        "created_at", "cycle_id", "status", "stages_completed", "loop_state",
        "loop_score", "decision", "monitor_score", "errors_count",
        "previous_result", "previous_profit", "runtime_health", "latency_ms",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    table_rows = "\n".join(
        "<tr>" + "".join(
            f"<td>{html.escape(str(row.get(column, '')))}</td>"
            for column in fieldnames
        ) + "</tr>"
        for row in rows
    )
    headers = "".join(f"<th>{html.escape(column)}</th>" for column in fieldnames)
    html_path = out / "v16_dashboard.html"
    html_path.write_text(
        f"""<!doctype html>
<html lang="sk">
<head>
<meta charset="utf-8">
<title>V16 Dashboard</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
.cards {{ display: flex; gap: 12px; flex-wrap: wrap; }}
.card {{ border: 1px solid #ccc; border-radius: 8px; padding: 12px 18px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 13px; }}
th {{ background: #f4f4f4; position: sticky; top: 0; }}
</style>
</head>
<body>
<h1>V16 Autonomous Cycle Dashboard</h1>
<div class="cards">
<div class="card"><strong>Cycles</strong><br>{summary['cycles']}</div>
<div class="card"><strong>READY</strong><br>{summary['ready_cycles']}</div>
<div class="card"><strong>Error cycles</strong><br>{summary['error_cycles']}</div>
<div class="card"><strong>Avg loop score</strong><br>{summary['average_loop_score']}</div>
<div class="card"><strong>Avg runtime health</strong><br>{summary['average_runtime_health']}</div>
</div>
<table>
<thead><tr>{headers}</tr></thead>
<tbody>{table_rows}</tbody>
</table>
</body>
</html>""",
        encoding="utf-8",
    )

    return {
        "summary": summary,
        "json": str(json_path),
        "csv": str(csv_path),
        "html": str(html_path),
        "status": "READY",
    }

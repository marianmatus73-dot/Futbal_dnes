from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any


def export_dashboard(
    result: dict[str, Any],
    export_dir: str | Path,
) -> dict[str, str]:
    out = Path(export_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "multisport_learning_v2_report.json"
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path = out / "multisport_learning_v2_metrics.csv"
    rows = []
    for sport, metrics in result.get("sports", {}).items():
        row = {"sport": sport, **metrics}
        rows.append(row)

    fieldnames = sorted({
        key
        for row in rows
        for key in row.keys()
    }) if rows else ["sport"]

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    headers = "".join(
        f"<th>{html.escape(name)}</th>"
        for name in fieldnames
    )
    body = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row.get(name, '')))}</td>"
            for name in fieldnames
        )
        + "</tr>"
        for row in rows
    )

    html_path = out / "multisport_learning_v2_dashboard.html"
    html_path.write_text(
        f"""<!doctype html>
<html lang="sk">
<head>
<meta charset="utf-8">
<title>Multisport Learning V2</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 13px; }}
th {{ background: #f4f4f4; }}
</style>
</head>
<body>
<h1>Multisport Learning Bundle V2</h1>
<p>Status: {html.escape(str(result.get('status', 'UNKNOWN')))}</p>
<table>
<thead><tr>{headers}</tr></thead>
<tbody>{body}</tbody>
</table>
</body>
</html>""",
        encoding="utf-8",
    )

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "html": str(html_path),
    }

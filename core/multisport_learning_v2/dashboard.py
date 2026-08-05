from __future__ import annotations

import csv
import html
import json
from pathlib import Path


def export_dashboard(
    result: dict,
    export_dir: str | Path,
) -> dict:
    out = Path(export_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "multisport_learning_v2_1_report.json"
    json_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rows = [
        {"sport": sport, **metrics}
        for sport, metrics
        in result.get("sports", {}).items()
    ]

    fieldnames = (
        sorted(
            {
                key
                for row in rows
                for key in row
            }
        )
        if rows
        else ["sport"]
    )

    csv_path = out / "multisport_learning_v2_1_metrics.csv"
    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    headers = "".join(
        f"<th>{html.escape(column)}</th>"
        for column in fieldnames
    )

    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row.get(column, '')))}</td>"
            for column in fieldnames
        )
        + "</tr>"
        for row in rows
    )

    html_path = out / "multisport_learning_v2_1_dashboard.html"
    html_path.write_text(
        f"""<!doctype html>
<html lang="sk">
<head>
<meta charset="utf-8">
<title>Multisport Learning V2.1</title>
</head>
<body>
<h1>Multisport Learning V2.1</h1>
<p>Status: {html.escape(str(result.get("status", "UNKNOWN")))}</p>
<table border="1" cellspacing="0" cellpadding="6">
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

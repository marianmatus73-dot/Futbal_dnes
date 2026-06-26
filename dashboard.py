from __future__ import annotations

import csv
from pathlib import Path

from flask import Flask, render_template_string


app = Flask(__name__)

AUDIT_FILE = Path("exports/pro_tip_audit.csv")


HTML = """
<!doctype html>
<html>
<head>
    <title>Pro Tipper Dashboard</title>
    <style>
        body { font-family: Arial; margin: 40px; background: #111; color: #eee; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #444; padding: 8px; }
        th { background: #222; }
        .good { color: #00ff99; }
        .bad { color: #ff6666; }
    </style>
</head>
<body>
    <h1>Pro Tipper Dashboard</h1>

    <h2>Summary</h2>
    <p>Total tips: {{ total }}</p>
    <p>Average confidence: {{ avg_confidence }}</p>
    <p>Average edge: {{ avg_edge }}%</p>

    <h2>History</h2>

    <table>
        <tr>
            <th>Date</th>
            <th>Sport</th>
            <th>League</th>
            <th>Match</th>
            <th>Pick</th>
            <th>Odds</th>
            <th>Edge</th>
            <th>Confidence</th>
            <th>Stake</th>
        </tr>
        {% for tip in tips %}
        <tr>
            <td>{{ tip.created_at }}</td>
            <td>{{ tip.sport }}</td>
            <td>{{ tip.league }}</td>
            <td>{{ tip.match }}</td>
            <td>{{ tip.pick }}</td>
            <td>{{ tip.odds }}</td>
            <td>{{ tip.edge }}</td>
            <td>{{ tip.confidence }}</td>
            <td>{{ tip.stake_units }}u</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
"""


def load_tips() -> list[dict]:
    if not AUDIT_FILE.exists():
        return []

    with AUDIT_FILE.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@app.route("/")
def index():
    tips = load_tips()

    total = len(tips)

    if total:
        avg_confidence = round(
            sum(float(t.get("confidence", 0) or 0) for t in tips) / total,
            2,
        )
        avg_edge = round(
            sum(float(t.get("edge", 0) or 0) for t in tips) / total * 100,
            2,
        )
    else:
        avg_confidence = 0
        avg_edge = 0

    tips = list(reversed(tips[-100:]))

    return render_template_string(
        HTML,
        tips=tips,
        total=total,
        avg_confidence=avg_confidence,
        avg_edge=avg_edge,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)

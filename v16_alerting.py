"""
Rule-based production alerts for V16 autonomous-cycle health.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_alerts(
    result: dict[str, Any],
    inputs: dict[str, Any],
    export_dir: str | Path = "exports",
) -> dict[str, Any]:
    stages = result.get("stages", {})
    loop = stages.get("v16_16_loop", {})
    monitor = stages.get("v16_15_monitor", {})
    alerts: list[dict[str, str]] = []

    if result.get("status") != "READY":
        alerts.append({"severity": "CRITICAL", "message": "V16 cycle is not READY."})
    if result.get("errors"):
        alerts.append({
            "severity": "CRITICAL",
            "message": f"V16 cycle reported {len(result['errors'])} error(s).",
        })
    if float(loop.get("loop_score", 0.0)) < 0.90:
        alerts.append({"severity": "WARNING", "message": "Loop score fell below 0.90."})
    if float(monitor.get("monitor_score", 0.0)) < 0.90:
        alerts.append({"severity": "WARNING", "message": "Monitor score fell below 0.90."})
    if float(inputs.get("runtime_health", 0.0)) < 0.80:
        alerts.append({"severity": "WARNING", "message": "Runtime health fell below 80%."})
    if int(inputs.get("latency_ms", 0)) > 60000:
        alerts.append({"severity": "WARNING", "message": "Average module latency exceeded 60 seconds."})
    if inputs.get("previous_result") == "LOSS":
        alerts.append({"severity": "INFO", "message": "Latest settled feedback is LOSS."})

    out = Path(export_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "alerts": alerts,
        "count": len(alerts),
        "highest_severity": (
            "CRITICAL" if any(a["severity"] == "CRITICAL" for a in alerts)
            else "WARNING" if any(a["severity"] == "WARNING" for a in alerts)
            else "INFO" if alerts else "NONE"
        ),
        "status": "READY",
    }
    (out / "v16_alerts.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out / "v16_alerts.txt").write_text(
        "\n".join(f"[{a['severity']}] {a['message']}" for a in alerts)
        or "No active V16 alerts.",
        encoding="utf-8",
    )
    return payload

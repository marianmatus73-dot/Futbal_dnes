
"""
V16.23 AUTOMATIC REPORTING & ALERTS

Creates reports and system alerts.
"""


def generate_report(metrics):
    alerts = []

    if metrics.get("roi", 0) < 0:
        alerts.append("NEGATIVE_ROI")

    if metrics.get("wins", 0) == 0:
        alerts.append("NO_WINS")

    return {
        "summary": metrics,
        "alerts": alerts,
        "status": "READY"
    }

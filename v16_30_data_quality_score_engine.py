
"""
V16.30 DATA QUALITY SCORE ENGINE

Scores normalized data quality before model usage.
"""


def calculate_quality(record):
    score = 0

    if record.get("event_id"):
        score += 40

    if record.get("odds"):
        score += 40

    if record.get("normalized"):
        score += 20

    status = "APPROVED" if score >= 80 else "REJECTED"

    return {
        "quality_score": score,
        "status": status
    }

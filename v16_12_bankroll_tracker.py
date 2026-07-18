
"""
V16.12 BANKROLL TRACKER

Tracks decisions and performance.
"""


import json
from pathlib import Path

DB = Path("data/performance_v16.json")


def save_result(record):
    DB.parent.mkdir(exist_ok=True)

    data = []
    if DB.exists():
        data = json.loads(DB.read_text())

    data.append(record)
    DB.write_text(json.dumps(data, indent=2))

    return {
        "saved": 1,
        "status": "READY"
    }


def performance():
    if not DB.exists():
        return {
            "records": 0,
            "status": "READY"
        }

    data = json.loads(DB.read_text())

    return {
        "records": len(data),
        "status": "READY"
    }

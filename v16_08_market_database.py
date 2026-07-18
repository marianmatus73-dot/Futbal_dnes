
"""
V16.08 MARKET DATABASE

Stores market history:
- odds snapshots
- CLV history
- learning data preparation
"""

import json
from pathlib import Path

DB = Path("data/market_history_v16.json")


def save_market(record):
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


def load_history():
    if not DB.exists():
        return []

    return json.loads(DB.read_text())

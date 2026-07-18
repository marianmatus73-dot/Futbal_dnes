
"""
V16.06 REAL DATA COLLECTION

Collects football market snapshots:
- opening odds
- current snapshots
- closing odds
- CLV preparation
"""

import json
from pathlib import Path
from datetime import datetime

STORE = Path("data/football_market_snapshots_v16.json")


def save_snapshot(event):
    STORE.parent.mkdir(exist_ok=True)
    data = []
    if STORE.exists():
        data = json.loads(STORE.read_text())

    event["timestamp"] = datetime.utcnow().isoformat()
    data.append(event)

    STORE.write_text(json.dumps(data, indent=2))
    return {"saved": 1, "status": "READY"}


def collect(events):
    results = []
    for event in events:
        results.append(save_snapshot(event))

    return {
        "version": "V16.06",
        "events_collected": len(events),
        "status": "READY",
        "results": results
    }

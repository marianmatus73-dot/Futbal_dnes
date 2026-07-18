
"""
V16.15 MODEL MEMORY

Stores calibration history and model adjustments.
"""

import json
from pathlib import Path

MEMORY = Path("data/model_memory_v16.json")


def save_memory(record):
    MEMORY.parent.mkdir(exist_ok=True)

    data = []
    if MEMORY.exists():
        data = json.loads(MEMORY.read_text())

    data.append(record)
    MEMORY.write_text(json.dumps(data, indent=2))

    return {
        "saved": True,
        "status": "READY"
    }


def load_memory():
    if not MEMORY.exists():
        return []

    return json.loads(MEMORY.read_text())

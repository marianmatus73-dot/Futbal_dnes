
"""
V16.16 LEAGUE & MARKET PROFILING

Builds profiles by league, market and source.
"""

import json
from pathlib import Path

PROFILE = Path("data/league_profiles_v16.json")


def save_profile(record):
    PROFILE.parent.mkdir(exist_ok=True)

    data = []
    if PROFILE.exists():
        data = json.loads(PROFILE.read_text())

    data.append(record)
    PROFILE.write_text(json.dumps(data, indent=2))

    return {
        "saved": True,
        "status": "READY"
    }


def load_profiles():
    if not PROFILE.exists():
        return []

    return json.loads(PROFILE.read_text())

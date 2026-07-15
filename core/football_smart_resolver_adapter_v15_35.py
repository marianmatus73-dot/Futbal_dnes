from __future__ import annotations

from difflib import SequenceMatcher
import re
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def norm(value):
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def similarity(a, b):
    a = norm(a)
    b = norm(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def resolve_snapshot_matches(snapshot_rows, postmatch_rows):
    resolved = []

    for post in postmatch_rows:
        best = None

        for snap in snapshot_rows:
            direct = (
                similarity(post.get("home_team"), snap.get("home_team"))
                + similarity(post.get("away_team"), snap.get("away_team"))
            ) / 2

            reverse = (
                similarity(post.get("home_team"), snap.get("away_team"))
                + similarity(post.get("away_team"), snap.get("home_team"))
            ) / 2

            score = max(direct, reverse)

            if best is None or score > best["score"]:
                best = {
                    "snapshot": snap,
                    "postmatch": post,
                    "score": round(score, 4),
                }

        if best and best["score"] >= 0.75:
            resolved.append(best)

    return resolved


def run_smart_resolver_adapter_v15_35(snapshot_rows, postmatch_rows):
    resolved = resolve_snapshot_matches(snapshot_rows, postmatch_rows)

    return {
        "version": "v15.35",
        "created_at": _now(),
        "snapshot_rows": len(snapshot_rows),
        "postmatch_rows": len(postmatch_rows),
        "matched_rows": len(resolved),
        "status": "READY" if resolved else "BUILDING",
    }, resolved

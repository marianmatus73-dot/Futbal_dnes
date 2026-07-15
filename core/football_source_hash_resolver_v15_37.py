from __future__ import annotations

from datetime import datetime, timezone
from difflib import SequenceMatcher


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def norm(value):
    return (value or "").strip().lower()


def sim(a, b):
    a = norm(a)
    b = norm(b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    return SequenceMatcher(None, a, b).ratio()


def resolve_by_source_hash(snapshot_rows, postmatch_rows):
    snapshot_hashes = {}

    for row in snapshot_rows:
        key = row.get("source_hash")
        if key:
            snapshot_hashes[key] = row

    resolved = []

    for post in postmatch_rows:
        key = post.get("source_hash")

        if key and key in snapshot_hashes:
            resolved.append({
                "method": "source_hash",
                "snapshot": snapshot_hashes[key],
                "postmatch": post,
                "score": 1.0,
            })

    return resolved


def resolve_fallback(snapshot_rows, postmatch_rows):
    resolved = []

    for post in postmatch_rows:
        best = None

        for snap in snapshot_rows:
            score = (
                sim(post.get("event"), snap.get("event"))
                + sim(post.get("source_hash"), snap.get("source_hash"))
            ) / 2

            if best is None or score > best["score"]:
                best = {
                    "method": "fallback",
                    "snapshot": snap,
                    "postmatch": post,
                    "score": score,
                }

        if best and best["score"] >= 0.75:
            resolved.append(best)

    return resolved


def run_source_hash_resolver_v15_37(snapshot_rows, postmatch_rows):
    resolved = resolve_by_source_hash(snapshot_rows, postmatch_rows)

    if not resolved:
        resolved = resolve_fallback(snapshot_rows, postmatch_rows)

    return {
        "version": "v15.37",
        "created_at": _now(),
        "snapshot_rows": len(snapshot_rows),
        "postmatch_rows": len(postmatch_rows),
        "matched_rows": len(resolved),
        "methods": list({r["method"] for r in resolved}),
        "status": "READY" if resolved else "BUILDING",
    }, resolved

from __future__ import annotations

import json
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def norm(value):
    return "".join(
        ch for ch in (value or "").lower()
        if ch.isalnum()
    )


def sim(a, b):
    a = norm(a)
    b = norm(b)
    if not a or not b:
        return 0.0
    return round(SequenceMatcher(None, a, b).ratio(), 4)


def debug_resolver(snapshot_rows, postmatch_rows):
    report_rows = []

    for post in postmatch_rows:
        best = None

        for snap in snapshot_rows:
            direct = (
                sim(post.get("home_team"), snap.get("home_team")) +
                sim(post.get("away_team"), snap.get("away_team"))
            ) / 2

            reverse = (
                sim(post.get("home_team"), snap.get("away_team")) +
                sim(post.get("away_team"), snap.get("home_team"))
            ) / 2

            score = max(direct, reverse)

            item = {
                "post_home": post.get("home_team"),
                "post_away": post.get("away_team"),
                "snap_home": snap.get("home_team"),
                "snap_away": snap.get("away_team"),
                "score": round(score, 4),
                "direct": round(direct, 4),
                "reverse": round(reverse, 4),
            }

            if best is None or score > best["score"]:
                best = item

        if best:
            report_rows.append(best)

    output = {
        "version": "v15.36",
        "created_at": _now(),
        "postmatch_checked": len(postmatch_rows),
        "snapshot_checked": len(snapshot_rows),
        "best_candidates": report_rows,
    }

    Path("exports").mkdir(exist_ok=True)
    Path("exports/football_resolver_debug_v15_36.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return output

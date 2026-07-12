from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_match_id_resolver_v15_16(
    *,
    postmatch_matches: int = 0,
    market_snapshots: int = 0,
    matches_resolved: int = 0,
    closing_recovered: int = 0,
) -> dict:
    resolve_rate = (
        round((matches_resolved / postmatch_matches) * 100, 1)
        if postmatch_matches
        else 0.0
    )

    closing_rate = (
        round((closing_recovered / postmatch_matches) * 100, 1)
        if postmatch_matches
        else 0.0
    )

    blockers = []

    if matches_resolved < postmatch_matches:
        blockers.append("Match IDs unresolved")

    if closing_recovered < postmatch_matches:
        blockers.append("Closing odds not recovered")

    if market_snapshots == 0:
        blockers.append("No market snapshots")

    return {
        "version": "v15.16",
        "created_at": _now(),
        "postmatch_matches": postmatch_matches,
        "market_snapshots": market_snapshots,
        "matches_resolved": matches_resolved,
        "closing_recovered": closing_recovered,
        "match_resolution_percent": resolve_rate,
        "closing_coverage_percent": closing_rate,
        "status": "READY" if closing_rate >= 90 else "BUILDING",
        "blockers": blockers,
    }


def export_match_id_resolver_v15_16(report: dict, export_dir: str = "exports"):
    directory = Path(export_dir)
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "football_match_id_resolver_v15_16.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_match_id_resolver_v15_16(**kwargs) -> dict:
    report = build_match_id_resolver_v15_16(**kwargs)
    export_match_id_resolver_v15_16(report)
    return report

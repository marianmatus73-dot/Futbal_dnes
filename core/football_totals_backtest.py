"""Evaluate only genuinely pre-match, settled football totals observations."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


def _time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def evaluate_totals(db_file: str | Path, min_samples: int = 30) -> dict:
    """Use recorded pre-match tips, never reconstructed post-match features."""
    path = Path(db_file)
    if not path.is_file():
        return {"status": "NO_DATABASE", "sample": 0, "required_sample": min_samples}

    with closing(sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)) as conn:
        table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sport_bets'"
        ).fetchone()
        if not table:
            return {"status": "NO_BET_HISTORY", "sample": 0, "required_sample": min_samples}
        columns = {row[1] for row in conn.execute("PRAGMA table_info(sport_bets)")}
        required = {"sport", "market", "selection", "odds", "prob_final", "stake", "start_time", "created_at", "result"}
        if not required.issubset(columns):
            return {"status": "MISSING_COLUMNS", "missing": sorted(required - columns), "sample": 0}
        rows = conn.execute(
            """SELECT selection, odds, prob_final, stake, start_time, created_at, result
               FROM sport_bets WHERE sport='football' AND market='totals_2.5'"""
        ).fetchall()

    excluded = {"unsettled": 0, "not_pre_match": 0, "invalid_values": 0}
    valid = []
    for selection, odds, probability, stake, start, created, result in rows:
        if str(result or "").upper() not in {"WON", "LOST"}:
            excluded["unsettled"] += 1
            continue
        start_at, created_at = _time(start), _time(created)
        if start_at is None or created_at is None or created_at >= start_at:
            excluded["not_pre_match"] += 1
            continue
        try:
            odds, probability, stake = float(odds), float(probability), float(stake)
        except (TypeError, ValueError):
            excluded["invalid_values"] += 1
            continue
        if odds <= 1 or not 0 < probability < 1 or stake <= 0 or not str(selection).lower().startswith(("over", "under")):
            excluded["invalid_values"] += 1
            continue
        valid.append((odds, probability, stake, str(result).upper() == "WON"))

    sample = len(valid)
    if sample < min_samples:
        return {
            "status": "INSUFFICIENT_SAMPLE", "sample": sample,
            "required_sample": min_samples, "observations_found": len(rows),
            "excluded": excluded,
            "note": "No performance conclusion or automatic activation is justified.",
        }
    turnover = sum(stake for _, _, stake, _ in valid)
    profit = sum(stake * (odds - 1) if won else -stake for odds, _, stake, won in valid)
    return {
        "status": "EVALUATED", "sample": sample, "required_sample": min_samples,
        "observations_found": len(rows), "excluded": excluded,
        "wins": sum(won for _, _, _, won in valid),
        "turnover": round(turnover, 2), "net_profit": round(profit, 2),
        "yield_pct": round(100 * profit / turnover, 2),
        "brier_score": round(sum((probability - int(won)) ** 2 for _, probability, _, won in valid) / sample, 4),
        "note": "Retrospective evaluation of recorded tips, not a full model walk-forward backtest.",
    }


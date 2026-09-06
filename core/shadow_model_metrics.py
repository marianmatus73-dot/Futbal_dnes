from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import Settings


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def build_shadow_model_metrics(
    settings: Settings,
    *,
    sport: str,
    minimum_events: int = 150,
) -> dict[str, Any]:
    """Measure shadow data once per event/market/selection.

    The earliest stored observation is the canonical, leakage-safe sample.
    Repeated hourly snapshots remain available for line-movement research but
    never inflate readiness or calibration counts.
    """
    database = Path(settings.db_file or "bets.db")
    empty = {
        "mode": "SHADOW",
        "settled_events": 0,
        "open_events": 0,
        "canonical_samples": 0,
        "raw_rows": 0,
        "duplicate_snapshots_excluded": 0,
        "probability_samples": 0,
        "market_brier_score": None,
        "calibration_error": None,
        "average_odds": None,
        "minimum_events": minimum_events,
        "readiness_pct": 0.0,
        "maturity": "EMPTY",
        "publishing_unlocked": False,
    }
    if not database.exists():
        return empty

    with sqlite3.connect(database) as conn:
        conn.row_factory = sqlite3.Row
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='sport_learning_observations'"
        ).fetchone()
        if not exists:
            return empty
        rows = conn.execute(
            """
            SELECT observed_at, external_event_id, market, selection, odds,
                   market_probability, result
            FROM sport_learning_observations
            WHERE LOWER(sport)=LOWER(?)
              AND TRIM(COALESCE(external_event_id, '')) <> ''
            ORDER BY datetime(observed_at), id
            """,
            (sport,),
        ).fetchall()

    canonical: dict[tuple[str, str, str], sqlite3.Row] = {}
    event_states: dict[str, set[str]] = {}
    for row in rows:
        event_id = str(row["external_event_id"])
        key = (event_id, str(row["market"]), str(row["selection"]))
        canonical.setdefault(key, row)
        event_states.setdefault(event_id, set()).add(str(row["result"] or "OPEN").upper())

    settled_event_ids = {
        event_id
        for event_id, states in event_states.items()
        if states and states.issubset({"WON", "LOST", "VOID"})
    }
    open_events = len(event_states) - len(settled_event_ids)
    probability_rows: list[tuple[float, int]] = []
    odds_values: list[float] = []
    for (event_id, _market, _selection), row in canonical.items():
        odds = _number(row["odds"])
        if odds is not None and odds > 1.0:
            odds_values.append(odds)
        if event_id not in settled_event_ids:
            continue
        probability = _number(row["market_probability"])
        result = str(row["result"] or "").upper()
        if probability is not None and 0.0 < probability < 1.0 and result in {"WON", "LOST"}:
            probability_rows.append((probability, 1 if result == "WON" else 0))

    brier = None
    calibration = None
    if probability_rows:
        brier = sum((probability - target) ** 2 for probability, target in probability_rows) / len(probability_rows)
        bins: dict[int, list[tuple[float, int]]] = {}
        for probability, target in probability_rows:
            bins.setdefault(min(9, int(probability * 10)), []).append((probability, target))
        calibration = sum(
            len(values) / len(probability_rows)
            * abs(
                sum(probability for probability, _ in values) / len(values)
                - sum(target for _, target in values) / len(values)
            )
            for values in bins.values()
        )

    settled_events = len(settled_event_ids)
    maturity = (
        "READY_FOR_MODEL"
        if settled_events >= minimum_events
        else "DEVELOPING"
        if settled_events >= 50
        else "EARLY"
        if settled_events > 0
        else "COLLECTING"
        if rows
        else "EMPTY"
    )
    return {
        **empty,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "settled_events": settled_events,
        "open_events": open_events,
        "canonical_samples": len(canonical),
        "raw_rows": len(rows),
        "duplicate_snapshots_excluded": max(0, len(rows) - len(canonical)),
        "probability_samples": len(probability_rows),
        "market_brier_score": round(brier, 6) if brier is not None else None,
        "calibration_error": round(calibration, 6) if calibration is not None else None,
        "average_odds": round(sum(odds_values) / len(odds_values), 4) if odds_values else None,
        "readiness_pct": round(min(settled_events / max(minimum_events, 1), 1.0) * 100.0, 2),
        "maturity": maturity,
    }


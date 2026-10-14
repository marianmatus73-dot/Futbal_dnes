from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from core.config import Settings


def _target(result: str) -> int | None:
    value = str(result or "").upper()
    if value in {"WON", "WIN", "V"}:
        return 1
    if value in {"LOST", "LOSS", "P"}:
        return 0
    return None


def _tier(odds: Any) -> str:
    try:
        value = float(odds)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if value < 1.20:
        return "BELOW_MINIMUM_<1.20"
    if value <= 1.60:
        return "SAFE_1.20_1.60"
    if value <= 2.50:
        return "VALUE_1.61_2.50"
    return "EXPERIMENTAL_2.51_PLUS"


def _evaluate(rows: list[tuple[float, int]], min_samples: int) -> dict:
    if len(rows) < min_samples:
        return {"samples": len(rows), "status": "BUILDING"}
    split = max(1, min(len(rows) - 1, int(len(rows) * .70)))
    train, test = rows[:split], rows[split:]
    bins: dict[int, list[int]] = {}
    for probability, target in train:
        bins.setdefault(min(9, int(probability * 10)), []).append(target)
    calibrated_bins = {
        key: (sum(values) + 2.5) / (len(values) + 5.0)
        for key, values in bins.items()
    }
    raw_brier = sum((p - y) ** 2 for p, y in test) / len(test)
    calibrated_brier = sum(
        (calibrated_bins.get(min(9, int(p * 10)), .50) - y) ** 2
        for p, y in test
    ) / len(test)
    return {
        "samples": len(rows), "train_samples": len(train),
        "test_samples": len(test), "split": "chronological_70_30",
        "raw_brier": round(raw_brier, 6),
        "calibrated_brier": round(calibrated_brier, 6),
        "calibration_decision": (
            "ACCEPT" if calibrated_brier < raw_brier else "REJECT"
        ),
        "status": "READY",
    }


def walkforward_report(
    settings: Settings,
    min_samples: int = 30,
    export_dir: str | Path = "exports",
) -> dict:
    db = Path(settings.db_file or "bets.db")
    report: dict[str, dict] = {}
    if not db.exists():
        return report
    with sqlite3.connect(db) as conn:
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sport_bets'"
        ).fetchone() is None:
            return report
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(sport_bets)")}
        order_parts = [name for name in ("start_time", "created_at", "id") if name in columns]
        order_sql = ", ".join(order_parts) if order_parts else "rowid"
        market_sql = "market" if "market" in columns else "'unknown'"
        odds_sql = "odds" if "odds" in columns else "NULL"
        tracking_filter = (
            "AND COALESCE(NULLIF(TRIM(tracking_mode), ''), 'LIVE')='LIVE'"
            if "tracking_mode" in columns else ""
        )
        raw = conn.execute(
            f"""
            SELECT sport, {market_sql}, {odds_sql}, prob_final, result
            FROM sport_bets
            WHERE sport IS NOT NULL AND prob_final BETWEEN 0.01 AND 0.99
              {tracking_filter}
            ORDER BY {order_sql}
            """
        ).fetchall()
        segments: dict[str, list[tuple[float, int]]] = {}
        for sport, market, odds, probability, result in raw:
            target = _target(result)
            if target is None:
                continue
            item = (float(probability), target)
            labels = (
                str(sport),
                f"{sport}/market={market or 'unknown'}",
                f"{sport}/tier={_tier(odds)}",
            )
            for label in labels:
                segments.setdefault(label, []).append(item)
        report = {
            label: _evaluate(rows, min_samples)
            for label, rows in sorted(segments.items())
        }
    output = Path(export_dir) / "sport_walkforward_calibration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def walkforward_summary(report: dict) -> str:
    ready = [row for row in report.values() if row.get("status") == "READY"]
    accepted = [row for row in ready if row.get("calibration_decision") == "ACCEPT"]
    rejected = [row for row in ready if row.get("calibration_decision") == "REJECT"]
    building = len(report) - len(ready)
    return (
        "\n=== TIME-SPLIT CALIBRATION ===\n"
        f"Segments: {len(report)} | ready={len(ready)} | building={building} | "
        f"calibration accepted={len(accepted)} | rejected={len(rejected)}\n"
        "Calibration is evaluated chronologically and is never accepted when "
        "it worsens Brier score.\n"
    )


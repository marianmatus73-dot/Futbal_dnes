from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from core.config import Settings


def _target(result: str) -> int | None:
    value = str(result or "").upper()
    if value in {"WON", "WIN", "V"}:
        return 1
    if value in {"LOST", "LOSS", "P"}:
        return 0
    return None


def walkforward_report(settings: Settings, min_samples: int = 30) -> dict:
    db = Path(settings.db_file or "bets.db")
    report: dict[str, dict] = {}
    if not db.exists():
        return report
    with sqlite3.connect(db) as conn:
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sport_bets'"
        ).fetchone() is None:
            return report
        sports = [row[0] for row in conn.execute(
            "SELECT DISTINCT sport FROM sport_bets WHERE sport IS NOT NULL"
        ).fetchall()]
        for sport in sports:
            raw = conn.execute(
                """
                SELECT prob_final, result FROM sport_bets
                WHERE sport=? AND prob_final BETWEEN 0.01 AND 0.99
                ORDER BY COALESCE(start_time, created_at), id
                """,
                (sport,),
            ).fetchall()
            rows = [(float(p), _target(r)) for p, r in raw]
            rows = [(p, y) for p, y in rows if y is not None]
            if len(rows) < min_samples:
                report[sport] = {"samples": len(rows), "status": "BUILDING"}
                continue

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
            report[sport] = {
                "samples": len(rows), "train_samples": len(train),
                "test_samples": len(test), "split": "chronological_70_30",
                "raw_brier": round(raw_brier, 6),
                "calibrated_brier": round(calibrated_brier, 6),
                "status": "READY",
            }
    output = Path("exports/sport_walkforward_calibration.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.config import Settings


SETTLED = {"WON", "WIN", "LOST", "LOSS", "VOID", "PUSH", "V", "P"}


def _time(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def build_settlement_integrity(
    settings: Settings,
    export_dir: str | Path = "exports",
    overdue_hours: int = 8,
) -> dict[str, Any]:
    """Audit whether every historical tip can safely enter model learning."""
    database = Path(settings.db_file or "bets.db")
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overdue_hours": overdue_hours,
        "sports": [],
    }
    if not database.exists():
        return payload
    with sqlite3.connect(database) as conn:
        conn.row_factory = sqlite3.Row
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sport_bets'"
        ).fetchone() is None:
            return payload
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(sport_bets)")}
        wanted = [
            name for name in (
                "sport", "result", "start_time", "profit", "final_score",
                "closing_odds", "tracking_mode",
            ) if name in columns
        ]
        rows = conn.execute(f"SELECT {', '.join(wanted)} FROM sport_bets").fetchall()

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, overdue_hours))
    sports = sorted({str(row["sport"] or "UNKNOWN") for row in rows})
    for sport in sports:
        sport_rows = [
            row for row in rows
            if str(row["sport"] or "UNKNOWN") == sport
            and (
                "tracking_mode" not in columns
                or str(row["tracking_mode"] or "LIVE") == "LIVE"
            )
        ]
        settled = [
            row for row in sport_rows
            if str(row["result"] or "").strip().upper() in SETTLED
        ]
        open_rows = [
            row for row in sport_rows
            if str(row["result"] or "OPEN").strip().upper() not in SETTLED
        ]
        overdue = sum(
            timestamp is not None and timestamp < cutoff
            for row in open_rows
            if (timestamp := _time(row["start_time"])) is not None
        )
        missing_profit = sum(
            row["profit"] in (None, "") for row in settled
        ) if "profit" in columns else len(settled)
        missing_score = sum(
            not str(row["final_score"] or "").strip() for row in settled
        ) if "final_score" in columns else len(settled)
        closing_samples = sum(
            row["closing_odds"] not in (None, "") for row in settled
        ) if "closing_odds" in columns else 0
        closing_coverage = closing_samples / len(settled) if settled else 0.0
        if missing_profit or missing_score:
            status = "BLOCKED"
        elif overdue:
            status = "WARNING"
        elif settled and closing_coverage < .80:
            status = "WARNING"
        else:
            status = "READY" if settled else "COLLECTING"
        payload["sports"].append({
            "sport": sport,
            "status": status,
            "settled": len(settled),
            "open": len(open_rows),
            "overdue_open": overdue,
            "settled_missing_profit": missing_profit,
            "settled_missing_final_score": missing_score,
            "closing_samples": closing_samples,
            "closing_coverage": round(closing_coverage, 4),
        })

    target = Path(export_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "settlement_integrity.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def settlement_integrity_report(payload: dict[str, Any]) -> str:
    lines = ["\n=== SETTLEMENT INTEGRITY ==="]
    for row in payload.get("sports", []):
        lines.append(
            f"- {row['sport']}: {row['status']} | settled={row['settled']} | "
            f"open={row['open']} | overdue={row['overdue_open']} | "
            f"missing score={row['settled_missing_final_score']} | "
            f"missing profit={row['settled_missing_profit']} | "
            f"closing={row['closing_coverage'] * 100:.1f}%"
        )
    return "\n".join(lines) + "\n"

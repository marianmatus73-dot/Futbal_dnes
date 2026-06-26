from __future__ import annotations

import sqlite3
from pathlib import Path

from core.config import Settings


def audit_block_summary(
    settings: Settings,
    sport: str | None = None,
    limit: int = 10,
) -> str:
    db_file = Path(settings.db_file or "bets.db")

    if not db_file.exists():
        return "Audit summary: database not found."

    where = "WHERE decision = 'BLOCK'"
    params = []

    if sport:
        where += " AND sport = ?"
        params.append(sport)

    try:
        with sqlite3.connect(db_file) as conn:
            rows = conn.execute(
                f"""
                SELECT sport, reason, COUNT(*) as cnt
                FROM sport_decision_audit
                {where}
                GROUP BY sport, reason
                ORDER BY cnt DESC
                LIMIT ?
                """,
                (*params, limit),
            ).fetchall()

    except Exception as e:
        return f"Audit summary failed: {e}"

    if not rows:
        return "Audit summary: no blocked decisions found."

    text = "\n\n=== BLOCKED DECISIONS SUMMARY ===\n"

    for sport_name, reason, count in rows:
        text += f"- {sport_name}: {reason} = {count}\n"

    return text

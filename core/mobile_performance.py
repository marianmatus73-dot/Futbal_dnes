from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from core.config import Settings


def export_mobile_performance(settings: Settings, *, export_dir: Path) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    destination = export_dir / "mobile_performance.json"
    starting_bank = float(settings.bank)
    points: list[dict] = []
    bank = starting_bank
    peak = starting_bank

    if Path(settings.db_file).exists():
        with sqlite3.connect(settings.db_file) as conn:
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(sport_bets)")}
            if {"result", "settled_at", "sport"}.issubset(columns):
                profit = "COALESCE(profit, 0)" if "profit" in columns else "0"
                clv = "clv_pct" if "clv_pct" in columns else "NULL"
                rows = conn.execute(
                    f"""SELECT sport, result, {profit}, {clv}, settled_at
                    FROM sport_bets
                    WHERE UPPER(result) IN ('WON','LOST')
                      AND settled_at IS NOT NULL
                    ORDER BY datetime(settled_at), id"""
                ).fetchall()
                for sport, result, row_profit, row_clv, settled_at in rows[-365:]:
                    bank += float(row_profit or 0)
                    peak = max(peak, bank)
                    drawdown = ((bank - peak) / peak * 100) if peak else 0
                    points.append({
                        "settled_at": settled_at,
                        "sport": str(sport or ""),
                        "result": str(result),
                        "profit": round(float(row_profit or 0), 4),
                        "bankroll": round(bank, 2),
                        "drawdown_pct": round(drawdown, 2),
                        "clv_pct": round(float(row_clv), 3) if row_clv not in (None, "") else None,
                    })

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "starting_bankroll": starting_bank,
        "current_bankroll": round(bank, 2),
        "points": points,
    }
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=export_dir, suffix=".tmp", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination

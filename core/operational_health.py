from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.config import Settings
from core.sport_policy import sport_policy


SPORTS = ("football", "tennis", "basketball", "hockey", "baseball", "handball", "nfl")
MINIMUM_SETTLED = {
    "football": 100,
    "baseball": 100,
    "tennis": 75,
    "basketball": 75,
    "hockey": 75,
    "handball": 50,
    "nfl": 50,
}
SETTLED_RESULTS = ("WON", "WIN", "LOST", "LOSS", "V", "P", "VOID", "PUSH")


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is None:
        return set()
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _scalar(conn: sqlite3.Connection, query: str, params: tuple = ()) -> Any:
    row = conn.execute(query, params).fetchone()
    return row[0] if row else None


def _sport_health(
    conn: sqlite3.Connection,
    sport: str,
    professional_table: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    bet_columns = _columns(conn, "sport_bets")
    snapshot_columns = _columns(conn, "sport_odds_snapshots")
    settled = open_bets = missing_identity = snapshots_24h = 0
    latest_snapshot = None

    if {"sport", "result"}.issubset(bet_columns):
        settled = int(_scalar(
            conn,
            "SELECT COUNT(*) FROM sport_bets WHERE LOWER(sport)=? "
            "AND UPPER(TRIM(COALESCE(result,''))) IN (?,?,?,?,?,?,?,?)",
            (sport, *SETTLED_RESULTS),
        ) or 0)
        open_bets = int(_scalar(
            conn,
            "SELECT COUNT(*) FROM sport_bets WHERE LOWER(sport)=? "
            "AND UPPER(TRIM(COALESCE(result,'OPEN'))) NOT IN (?,?,?,?,?,?,?,?)",
            (sport, *SETTLED_RESULTS),
        ) or 0)
        if "external_event_id" in bet_columns:
            missing_identity = int(_scalar(
                conn,
                "SELECT COUNT(*) FROM sport_bets WHERE LOWER(sport)=? "
                "AND UPPER(TRIM(COALESCE(result,'OPEN'))) NOT IN (?,?,?,?,?,?,?,?) "
                "AND TRIM(COALESCE(external_event_id,''))=''",
                (sport, *SETTLED_RESULTS),
            ) or 0)

    if {"sport", "captured_at"}.issubset(snapshot_columns):
        cutoff = (now - timedelta(hours=24)).isoformat()
        snapshots_24h = int(_scalar(
            conn,
            "SELECT COUNT(*) FROM sport_odds_snapshots WHERE LOWER(sport)=? "
            "AND captured_at>=?",
            (sport, cutoff),
        ) or 0)
        latest_snapshot = _scalar(
            conn,
            "SELECT MAX(captured_at) FROM sport_odds_snapshots WHERE LOWER(sport)=?",
            (sport,),
        )

    metrics = (
        professional_table.get("sports", {}).get(sport, {}).get("all_time", {})
    )
    settled = max(settled, int(metrics.get("settled", 0) or 0))
    yield_pct = float(metrics.get("yield_pct", 0.0) or 0.0)
    minimum = MINIMUM_SETTLED[sport]
    publishing_mode = "SHADOW" if sport == "handball" else "LIVE"

    if missing_identity:
        status = "ATTENTION"
        message = f"{missing_identity} otvorených záznamov nemá stabilné ID"
    elif settled < minimum:
        status = "COLLECTING"
        message = f"Kalibrácia {settled}/{minimum} uzavretých tipov"
    elif settled >= minimum and yield_pct <= -5.0:
        status = "CAUTION"
        message = "Dostatok dát, ale výkon vyžaduje opatrnosť"
    elif snapshots_24h or open_bets:
        status = "READY"
        message = "Dáta prichádzajú a model má dostatočnú vzorku"
    else:
        status = "IDLE"
        message = "Bez dnešných udalostí; nejde automaticky o chybu"

    policy = sport_policy(sport)
    return {
        "sport": sport,
        "status": status,
        "message": message,
        "publishing_mode": publishing_mode,
        "snapshots_24h": snapshots_24h,
        "latest_snapshot_at": latest_snapshot,
        "open_bets": open_bets,
        "settled_bets": settled,
        "minimum_settled": minimum,
        "missing_event_id": missing_identity,
        "yield_pct": round(yield_pct, 2),
        "rules": {
            "min_edge_pct": round(policy.min_edge * 100, 1),
            "min_confidence": policy.min_confidence,
            "odds": [policy.min_odds, policy.max_odds],
            "max_tips": policy.max_tips,
        },
    }


def build_operational_health(
    settings: Settings,
    risk_summary: Any = None,
    professional_table: dict[str, Any] | None = None,
    export_dir: str | Path = "exports",
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    table = professional_table or {}
    db_path = Path(settings.db_file or "bets.db")
    if db_path.exists():
        with closing(sqlite3.connect(db_path)) as conn:
            sports = [_sport_health(conn, sport, table, now) for sport in SPORTS]
    else:
        with closing(sqlite3.connect(":memory:")) as conn:
            sports = [_sport_health(conn, sport, table, now) for sport in SPORTS]

    funnel = {
        "candidates": int(getattr(risk_summary, "candidates", 0) or 0),
        "accepted": int(getattr(risk_summary, "accepted", 0) or 0),
        "rejected": int(getattr(risk_summary, "rejected", 0) or 0),
        "rejection_reasons": dict(getattr(risk_summary, "rejected_reasons", {}) or {}),
    }
    payload = {
        "schema_version": 1,
        "generated_at": now.isoformat(),
        "funnel": funnel,
        "football_without_paid_xg": {
            "paid_xg_required": False,
            "h2h": "LIVE",
            "double_chance": "LIVE",
            "totals": "WATCH_ONLY_WITHOUT_XG",
            "note": "1X2 a dvojtip fungujú bez plateného xG. Góly nad/pod sa bez xG iba sledujú a vyhodnocujú.",
        },
        "sports": sports,
    }
    output = Path(export_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "operational_health.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def operational_health_report(payload: dict[str, Any]) -> str:
    funnel = payload.get("funnel", {})
    lines = [
        "\n=== OPERATIONAL HEALTH ===",
        f"Funnel: candidates={funnel.get('candidates', 0)} | "
        f"accepted={funnel.get('accepted', 0)} | rejected={funnel.get('rejected', 0)}",
    ]
    for item in payload.get("sports", []):
        lines.append(
            f"- {item['sport']}: {item['status']} | settled={item['settled_bets']}/"
            f"{item['minimum_settled']} | snapshots24h={item['snapshots_24h']} | "
            f"mode={item['publishing_mode']}"
        )
    return "\n".join(lines) + "\n"

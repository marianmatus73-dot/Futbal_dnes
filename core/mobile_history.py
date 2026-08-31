from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from core.config import Settings


def _result(value: object) -> str:
    normalized = str(value or "").strip().upper()
    return {
        "V": "WON",
        "P": "LOST",
        "WIN": "WON",
        "LOSS": "LOST",
    }.get(normalized, normalized if normalized in {"WON", "LOST", "VOID", "UNRESOLVED"} else "OPEN")


def export_mobile_tip_history(
    settings: Settings,
    *,
    export_dir: Path,
    limit_per_sport: int = 5,
) -> Path:
    """Export recent real analyses for the mobile sport screens."""
    export_dir.mkdir(parents=True, exist_ok=True)
    destination = export_dir / "mobile_tip_history.json"
    sports: dict[str, list[dict]] = {}
    db_file = Path(settings.db_file)

    if db_file.exists():
        with sqlite3.connect(db_file) as conn:
            conn.row_factory = sqlite3.Row
            table_exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sport_bets'"
            ).fetchone()
            if table_exists:
                columns = {
                    str(row[1]) for row in conn.execute("PRAGMA table_info(sport_bets)")
                }

                def field(name: str, fallback: str = "NULL") -> str:
                    return name if name in columns else f"{fallback} AS {name}"

                rows = conn.execute(
                    f"""
                    SELECT id, sport, league, event, selection, odds,
                           {field('market', "'h2h'")},
                           {field('result', "'OPEN'")},
                           {field('bookmaker')}, {field('prob_final')},
                           {field('edge')}, {field('stake')},
                           {field('closing_odds')}, {field('clv_pct')},
                           {field('settlement_source')},
                           {field('start_time')}, {field('created_at')},
                           {field('settled_at')}, {field('final_score')},
                           {field('home_goals')}, {field('away_goals')}
                    FROM sport_bets
                    WHERE TRIM(COALESCE(sport, '')) <> ''
                      AND TRIM(COALESCE(event, '')) <> ''
                    ORDER BY COALESCE(start_time, created_at, '') DESC, id DESC
                    """
                ).fetchall()

                # sport_bets stores every evaluated selection before the
                # professional filter.  A two-way market can therefore contain
                # both teams from the same match.  The mobile screen must show
                # one model choice per event + market, not two apparent tips
                # against each other.
                grouped: dict[str, dict[tuple[str, str], sqlite3.Row]] = {}
                for row in rows:
                    sport = str(row["sport"]).strip().lower()
                    key = (
                        str(row["event"]).strip().casefold(),
                        str(row["market"] or "h2h").strip().casefold(),
                    )
                    current = grouped.setdefault(sport, {}).get(key)
                    rank = (
                        float(row["edge"] or 0),
                        float(row["prob_final"] or 0),
                        float(row["stake"] or 0),
                        int(row["id"] or 0),
                    )
                    current_rank = (
                        float(current["edge"] or 0),
                        float(current["prob_final"] or 0),
                        float(current["stake"] or 0),
                        int(current["id"] or 0),
                    ) if current is not None else None
                    if current_rank is None or rank > current_rank:
                        grouped[sport][key] = row

                for sport, event_rows in grouped.items():
                    selected_rows = sorted(
                        event_rows.values(),
                        key=lambda row: (
                            str(row["start_time"] or row["created_at"] or ""),
                            int(row["id"] or 0),
                        ),
                        reverse=True,
                    )[:limit_per_sport]
                    items = sports.setdefault(sport, [])
                    for row in selected_rows:
                        final_score = str(row["final_score"] or "").strip()
                        if not final_score and row["home_goals"] is not None and row["away_goals"] is not None:
                            final_score = f'{row["home_goals"]}-{row["away_goals"]}'
                        items.append(
                            {
                                "sport": sport,
                                "league": str(row["league"] or ""),
                                "match": str(row["event"]),
                                "pick": str(row["selection"] or ""),
                                "market": str(row["market"] or "h2h"),
                                "odds": float(row["odds"] or 0),
                                "bookmaker": str(row["bookmaker"] or ""),
                                "model_probability": float(row["prob_final"] or 0),
                                "edge": float(row["edge"] or 0),
                                "stake": float(row["stake"] or 0),
                                "closing_odds": float(row["closing_odds"]) if row["closing_odds"] not in (None, "") else None,
                                "clv_pct": float(row["clv_pct"]) if row["clv_pct"] not in (None, "") else None,
                                "settlement_source": str(row["settlement_source"] or ""),
                                "result": _result(row["result"]),
                                "final_score": final_score or None,
                                "start_time": row["start_time"],
                                "created_at": row["created_at"],
                                "settled_at": row["settled_at"],
                            }
                        )

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "limit_per_sport": limit_per_sport,
        "sports": sports,
    }
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=export_dir, suffix=".tmp", delete=False
    )
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


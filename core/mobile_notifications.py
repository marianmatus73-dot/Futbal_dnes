from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


def _key(item: dict[str, Any]) -> str:
    return "|".join(str(item.get(name) or "").strip().casefold() for name in ("sport", "event", "match", "market", "selection", "pick"))


def build_notifications(previous: dict[str, Any] | None, current: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only actionable changes. This is an in-app feed, not noisy push spam."""
    old_rows = (previous or {}).get("selected", [])
    old = {_key(row): row for row in old_rows}
    new_rows = current.get("selected", [])
    new = {_key(row): row for row in new_rows}
    created = datetime.now(timezone.utc).isoformat()
    events: list[dict[str, Any]] = []
    for key, row in new.items():
        before = old.get(key)
        stage = str(row.get("release_stage") or "").upper()
        if stage == "FINAL" and (before is None or str(before.get("release_stage") or "").upper() != "FINAL"):
            events.append({"type": "FINAL", "title": "Tip je finálne potvrdený", "message": str(row.get("match") or row.get("event") or ""), "created_at": created, "sport": row.get("sport")})
        if bool(row.get("lineup_verified")) and not bool((before or {}).get("lineup_verified")):
            events.append({"type": "LINEUP", "title": "Zostavy sú potvrdené", "message": str(row.get("match") or row.get("event") or ""), "created_at": created, "sport": row.get("sport")})
        opening = row.get("opening_odds")
        odds = row.get("odds")
        if opening and odds and float(opening) > 1 and abs(float(odds) / float(opening) - 1) >= .05:
            events.append({"type": "ODDS_MOVE", "title": "Výrazný pohyb kurzu", "message": f"{row.get('match') or row.get('event')}: {float(opening):.2f} → {float(odds):.2f}", "created_at": created, "sport": row.get("sport")})
    for key, row in old.items():
        if key not in new and str(row.get("release_stage") or "").upper() == "FINAL":
            events.append({"type": "CANCELLED", "title": "Finálny tip už nie je aktívny", "message": str(row.get("match") or row.get("event") or ""), "created_at": created, "sport": row.get("sport")})
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    seen = {(item["type"], item["message"]) for item in events}
    for item in (previous or {}).get("notifications", []):
        try:
            created_at = datetime.fromisoformat(str(item.get("created_at") or "").replace("Z", "+00:00"))
        except ValueError:
            continue
        signature = (item.get("type"), item.get("message"))
        if created_at >= cutoff and signature not in seen:
            events.append(item)
            seen.add(signature)
    return events[:50]

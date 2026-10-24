from __future__ import annotations

import json
import hashlib
import os
import re
import sqlite3
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from core.mobile_notifications import build_notifications
from core.rejection_explanations import enrich_rejection
from core.config import Settings


def _tip_payload(tip, decision: str) -> dict:
    payload = asdict(tip)
    payload.update(
        {
            "event": tip.match,
            "selection": tip.pick,
            "market_probability": tip.implied_probability,
            "stake_u": tip.stake_units,
            "decision": decision,
        }
    )
    return payload


def _rejected_payload(tip) -> dict:
    payload = _tip_payload(tip, "REJECT")
    payload.setdefault("rejection_reason", "not selected for the published top list")
    return enrich_rejection(payload)


def build_low_odds_watch(
    accepted: list,
    rejected: list[dict],
    *,
    limit: int = 0,
    audit_candidates: list[dict] | None = None,
) -> list[dict]:
    """Build a separate daily football experiment for odds 1.20-1.60.

    The production filter remains untouched.  When it provides fewer than the
    requested number of low-price rows, the current scan audit supplies the
    most probable unique matches as observation-only selections. A non-positive
    limit keeps every eligible unique selection; production can still set a
    positive safety cap through ``LOW_ODDS_WATCH_LIMIT``.
    """
    rows: list[dict] = []
    for tip in accepted:
        if str(getattr(tip, "sport", "")).lower() != "football":
            continue
        odds = float(getattr(tip, "odds", 0) or 0)
        if 1.20 <= odds <= 1.60:
            item = _tip_payload(tip, "ACCEPT")
            item["watch_status"] = "PASSED_PRO_FILTER"
            rows.append(item)

    for candidate in rejected:
        if str(candidate.get("sport", "")).lower() != "football":
            continue
        odds = float(candidate.get("odds", 0) or 0)
        if not 1.20 <= odds <= 1.60:
            continue
        item = dict(candidate)
        item.update(
            {
                "match": item.get("match") or item.get("event") or "",
                "pick": item.get("pick") or item.get("selection") or "",
                "model_probability": item.get("model_probability", item.get("prob_final", 0)),
                "market_probability": item.get("market_probability", item.get("prob_market", 0)),
                "confidence": item.get("confidence", item.get("effective_confidence", item.get("score", 0))),
                "stake_amount": item.get("stake_amount", item.get("stake", 0)),
                "risk": item.get("risk", "medium"),
                "decision": "REJECT",
                "watch_status": "WATCH_ONLY",
            }
        )
        reason = str(item.get("rejection_reason") or "did not pass the production filter")
        item["rejected_reasons"] = item.get("rejected_reasons") or [reason]
        rows.append(item)

    for candidate in audit_candidates or []:
        odds = float(candidate.get("odds", 0) or 0)
        if not 1.20 <= odds <= 1.60:
            continue
        reason = str(candidate.get("reason") or "upstream football scan")
        final_match = re.search(r"(?:^|;)\s*(?:final|fallback)=([0-9.]+)", reason)
        model_probability = (
            float(candidate.get("prob_model"))
            if candidate.get("prob_model") not in (None, "")
            else float(final_match.group(1))
            if final_match
            else float(candidate.get("prob_market", 0) or 0)
        )
        market_probability = float(candidate.get("prob_market", 0) or 0)
        market = str(candidate.get("market") or "h2h").strip().casefold()
        if market not in {"h2h", "double_chance", "totals_2.5"}:
            continue
        if not 0 < model_probability < 1 or not 0 < market_probability < 1:
            continue
        rows.append({
            "sport": "football",
            "league": str(candidate.get("league") or "football"),
            "event": str(candidate.get("event") or ""),
            "match": str(candidate.get("event") or ""),
            "selection": str(candidate.get("selection") or ""),
            "pick": str(candidate.get("selection") or ""),
            "market": market,
            "odds": odds,
            "model_probability": model_probability,
            "market_probability": market_probability,
            "implied_probability": market_probability,
            "edge": model_probability - market_probability,
            "confidence": round(model_probability * 100),
            "bookmaker": str(candidate.get("bookmaker") or "market"),
            "bookmaker_weight": 1.0,
            "bookmaker_samples": 0,
            "bookmaker_label": "EXPERIMENT",
            "risk": "low_odds_experiment",
            "stake_units": 0.10,
            "stake_u": 0.10,
            "stake_amount": 0,
            "created_at": str(candidate.get("created_at") or ""),
            "external_event_id": str(candidate.get("external_event_id") or ""),
            "start_time": str(candidate.get("start_time") or ""),
            "decision": "WATCH",
            "watch_status": "DAILY_LOW_ODDS_EXPERIMENT",
            "not_official_tip": True,
            "reason": reason,
            "rejected_reasons": ["Denný experiment; neprešiel ostrým value filtrom"],
        })

    best: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (
            str(row.get("event") or row.get("match") or "").strip().casefold(),
            str(row.get("market") or "h2h").strip().casefold(),
        )
        rank = (
            row.get("decision") == "ACCEPT",
            float(row.get("confidence") or 0),
            float(row.get("model_probability") or 0) * float(row.get("odds") or 0) - 1.0,
        )
        current = best.get(key)
        current_rank = (
            current.get("decision") == "ACCEPT",
            float(current.get("confidence") or 0),
            float(current.get("model_probability") or 0) * float(current.get("odds") or 0) - 1.0,
        ) if current else None
        if current_rank is None or rank > current_rank:
            best[key] = row
    ranked = sorted(
        best.values(),
        key=lambda row: (
            row.get("decision") == "ACCEPT",
            float(row.get("model_probability") or 0),
            float(row.get("model_probability") or 0) * float(row.get("odds") or 0) - 1.0,
        ),
        reverse=True,
    )

    # Give each supported football market a fair place before filling the
    # remaining slots by score. Otherwise abundant h2h rows hide totals and
    # double-chance observations even when those markets were collected.
    selected: list[dict] = []
    selected_ids: set[int] = set()
    selection_limit = max(1, limit) if limit > 0 else None
    for market in ("h2h", "double_chance", "totals_2.5"):
        candidate = next(
            (row for row in ranked if str(row.get("market")) == market),
            None,
        )
        if candidate is not None and (
            selection_limit is None or len(selected) < selection_limit
        ):
            selected.append(candidate)
            selected_ids.add(id(candidate))
    for candidate in ranked:
        if selection_limit is not None and len(selected) >= selection_limit:
            break
        if id(candidate) not in selected_ids:
            selected.append(candidate)
            selected_ids.add(id(candidate))
    return selected


def persist_low_odds_watch(settings: Settings, rows: list[dict]) -> int:
    """Persist observation-only football picks so they can be settled.

    These rows use a dedicated tracking mode and never count as LIVE tips.
    One unit is used only for comparable experiment yield; no real stake is
    allocated by the professional risk engine.
    """
    database = Path(settings.db_file or os.getenv("DB_FILE", "bets.db"))
    if not database.exists() or not rows:
        return 0
    with sqlite3.connect(database) as conn:
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(sport_bets)")}
        for name, definition in {
            "tracking_mode": "TEXT DEFAULT 'LIVE'",
            "external_event_id": "TEXT",
            "engine_version": "TEXT",
        }.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE sport_bets ADD COLUMN {name} {definition}")
        before = conn.total_changes
        for row in rows:
            event = str(row.get("event") or row.get("match") or "").strip()
            selection = str(row.get("selection") or row.get("pick") or "").strip()
            market = str(row.get("market") or "h2h").strip().casefold()
            external_id = str(row.get("external_event_id") or "").strip()
            start_time = str(row.get("start_time") or "").strip()
            odds = float(row.get("odds") or 0)
            if not event or not selection or not start_time or odds <= 1.01:
                continue
            identity = external_id or f"{event}|{start_time}"
            source_hash = hashlib.sha256(
                f"LOW_ODDS_WATCH|{identity}|{market}|{selection}".encode("utf-8")
            ).hexdigest()[:24]
            probability = float(row.get("model_probability") or 0)
            market_probability = float(
                row.get("market_probability") or row.get("implied_probability") or 0
            )
            home, separator, away = event.partition(" vs ")
            conn.execute(
                """
                INSERT OR IGNORE INTO sport_bets (
                    sport, league, event, home_team, away_team, market,
                    selection, odds, prob_model, prob_market, prob_final,
                    edge, stake, bookmaker, start_time, score, source_hash,
                    result, external_event_id, engine_version, tracking_mode
                ) VALUES (
                    'football', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1.0, ?, ?, ?,
                    ?, 'OPEN', ?, 'football-2.0-low-odds-watch', 'LOW_ODDS_WATCH'
                )
                """,
                (
                    str(row.get("league") or "football"), event, home,
                    away if separator else "", market, selection, odds,
                    probability, market_probability, probability,
                    probability * odds - 1.0,
                    str(row.get("bookmaker") or "market"), start_time,
                    float(row.get("confidence") or probability * 100),
                    source_hash, external_id,
                ),
            )
        return conn.total_changes - before


def save_latest_tip_card(
    selected: list,
    rejected: list,
    *,
    export_dir: Path,
    top_limit: int,
    min_edge: float = 0.04,
    min_confidence: int = 65,
    low_odds_watch: list[dict] | None = None,
) -> Path:
    """Atomically replace the daily card, including an empty-card run."""
    export_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().astimezone().isoformat()
    payload = {
        "schema_version": 3,
        "generated_at": generated_at,
        "publishable": bool(selected),
        "policy": {
            "min_edge": min_edge,
            "odds_min": 1.0,
            "odds_max": 999.0,
            "min_confidence": min_confidence,
            "top_limit": top_limit,
        },
        "selected": [_tip_payload(tip, "ACCEPT") for tip in selected],
        "rejected_sample": [_rejected_payload(tip) for tip in rejected],
        "low_odds_watch": list(low_odds_watch or []),
    }
    destination = export_dir / "latest_tip_card.json"
    previous = None
    if destination.exists():
        try:
            previous = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = None
    payload["notifications"] = build_notifications(previous, payload)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=export_dir,
        prefix="tip-card-",
        suffix=".tmp",
        delete=False,
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


def save_latest_rejected_candidates(
    risk_rejected: list[dict],
    selection_rejected: list,
    *,
    export_dir: Path,
) -> Path:
    """Atomically export every rejected candidate from the current run."""
    export_dir.mkdir(parents=True, exist_ok=True)
    candidates = [enrich_rejection(item) for item in risk_rejected]
    for tip in selection_rejected:
        item = _tip_payload(tip, "REJECT")
        item.update(
            {
                "rejection_stage": "VALUE_OR_TOP_SELECTION",
                "rejection_reason": "not selected for the published top list",
            }
        )
        candidates.append(enrich_rejection(item))

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "total": len(candidates),
        "candidates": candidates,
    }
    destination = export_dir / "latest_rejected_candidates.json"
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=export_dir,
        prefix="rejected-candidates-",
        suffix=".tmp",
        delete=False,
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


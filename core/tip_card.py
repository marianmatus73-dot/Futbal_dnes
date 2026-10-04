from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from core.mobile_notifications import build_notifications
from core.rejection_explanations import enrich_rejection


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
    limit: int = 5,
) -> list[dict]:
    """Build a separate football watchlist for decimal odds 1.20-1.60."""
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
    return sorted(
        best.values(),
        key=lambda row: (
            row.get("decision") == "ACCEPT",
            float(row.get("confidence") or 0),
            float(row.get("model_probability") or 0) * float(row.get("odds") or 0) - 1.0,
        ),
        reverse=True,
    )[: max(1, limit)]


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


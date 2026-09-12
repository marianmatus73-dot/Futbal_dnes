from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path


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


def save_latest_tip_card(
    selected: list,
    rejected: list,
    *,
    export_dir: Path,
    top_limit: int,
    min_edge: float = 0.04,
    min_confidence: int = 65,
) -> Path:
    """Atomically replace the daily card, including an empty-card run."""
    export_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().astimezone().isoformat()
    payload = {
        "schema_version": 2,
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
        "rejected_sample": [_tip_payload(tip, "REJECT") for tip in rejected],
    }
    destination = export_dir / "latest_tip_card.json"
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
    candidates = list(risk_rejected)
    for tip in selection_rejected:
        item = _tip_payload(tip, "REJECT")
        item.update(
            {
                "rejection_stage": "VALUE_OR_TOP_SELECTION",
                "rejection_reason": "not selected for the published top list",
            }
        )
        candidates.append(item)

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


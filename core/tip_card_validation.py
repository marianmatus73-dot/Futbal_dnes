from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_CANDIDATE_FIELDS = (
    "sport", "event", "selection", "odds", "model_probability",
    "market_probability", "edge", "confidence", "decision",
)


class TipCardValidationError(ValueError):
    pass


def _number(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise TipCardValidationError(f"{field} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TipCardValidationError(f"{field} must be numeric") from exc


def validate_tip_card(path: Path, *, max_age_minutes: int = 30) -> dict:
    try:
        card = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TipCardValidationError(f"Cannot read valid tip card: {path}") from exc

    if card.get("schema_version") != 2:
        raise TipCardValidationError("Unsupported tip card schema_version")

    try:
        generated = datetime.fromisoformat(str(card["generated_at"]).replace("Z", "+00:00"))
    except (KeyError, ValueError) as exc:
        raise TipCardValidationError("generated_at is missing or invalid") from exc
    if generated.tzinfo is None:
        raise TipCardValidationError("generated_at must include a timezone")
    age_seconds = (datetime.now(timezone.utc) - generated.astimezone(timezone.utc)).total_seconds()
    if age_seconds < -300 or age_seconds > max_age_minutes * 60:
        raise TipCardValidationError("Tip card is not fresh")

    selected = card.get("selected")
    rejected = card.get("rejected_sample")
    if not isinstance(selected, list) or not isinstance(rejected, list):
        raise TipCardValidationError("selected and rejected_sample must be lists")
    if bool(card.get("publishable")) != bool(selected):
        raise TipCardValidationError("publishable does not match selected tips")

    policy = card.get("policy") or {}
    top_limit = int(_number(policy.get("top_limit"), "policy.top_limit"))
    if top_limit < 1 or len(selected) > top_limit:
        raise TipCardValidationError("selected tips exceed policy.top_limit")

    rows = [("selected", row) for row in selected] + [
        ("rejected_sample", row) for row in rejected
    ]
    for index, (group, row) in enumerate(rows):
        if not isinstance(row, dict):
            raise TipCardValidationError(f"{group}[{index}] must be an object")
        missing = [name for name in REQUIRED_CANDIDATE_FIELDS if row.get(name) in (None, "")]
        if missing:
            raise TipCardValidationError(f"{group}[{index}] missing: {','.join(missing)}")
        odds = _number(row["odds"], "odds")
        model = _number(row["model_probability"], "model_probability")
        market = _number(row["market_probability"], "market_probability")
        edge = _number(row["edge"], "edge")
        confidence = _number(row["confidence"], "confidence")
        if odds <= 1 or not 0 < model < 1 or not 0 < market < 1:
            raise TipCardValidationError(f"{group}[{index}] has an invalid probability or odds")
        if not 0 <= confidence <= 100:
            raise TipCardValidationError(f"{group}[{index}] has invalid confidence")
        if abs(edge - (model - market)) > 0.002:
            raise TipCardValidationError(f"{group}[{index}] has inconsistent edge")
        expected = "ACCEPT" if group == "selected" else "REJECT"
        if row["decision"] != expected:
            raise TipCardValidationError(f"{group}[{index}] has invalid decision")

    return {
        "status": "READY",
        "generated_at": card["generated_at"],
        "selected": len(selected),
        "rejected": len(rejected),
        "complete": len(rows),
    }

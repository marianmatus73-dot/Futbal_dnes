from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def to_float_or_none(value) -> float | None:
    if value is None or value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bet_to_tip_dict(bet: Any, fallback_sport: str = "") -> dict | None:
    if bet is None:
        return None

    if isinstance(bet, dict):
        data = dict(bet)
    elif is_dataclass(bet):
        data = asdict(bet)
    else:
        data = {
            key: getattr(bet, key)
            for key in dir(bet)
            if not key.startswith("_")
            and not callable(getattr(bet, key, None))
        }

    odds = to_float_or_none(
        data.get("odds")
        or data.get("price")
        or data.get("best_odds")
    )

    if odds is None or odds <= 1:
        return None

    prob_final = to_float_or_none(
        data.get("prob_final")
        or data.get("model_probability")
        or data.get("probability")
        or data.get("win_probability")
    )

    prob_market = to_float_or_none(
        data.get("prob_market")
        or data.get("market_probability")
        or data.get("implied_probability")
    )

    prob_model = to_float_or_none(
        data.get("prob_model")
        or data.get("model_probability")
        or prob_final
    )

    raw_edge = to_float_or_none(data.get("edge"))

    if raw_edge is None and prob_final is not None:
        raw_edge = (prob_final * odds) - 1.0

    event = (
        data.get("event")
        or data.get("match")
        or data.get("fixture")
        or data.get("game")
        or "Unknown"
    )

    selection = (
        data.get("selection")
        or data.get("pick")
        or data.get("bet")
        or data.get("market")
        or "Unknown"
    )

    reason_parts = []

    if data.get("reason"):
        reason_parts.append(str(data.get("reason")))

    if raw_edge is not None:
        reason_parts.append(f"Raw edge {raw_edge:.1%}")

    if data.get("score") not in (None, ""):
        reason_parts.append(f"Score {data.get('score')}")

    return {
        "sport": data.get("sport") or fallback_sport,
        "league": data.get("league") or data.get("competition") or "Unknown",
        "match": event,
        "pick": selection,
        "odds": odds,

        "model_probability": prob_final or prob_model or prob_market,
        "market_probability": prob_market,

        "elo_probability": prob_model or prob_final or prob_market,
        "xg_probability": prob_final or prob_model or prob_market,
        "form_probability": prob_final or prob_model or prob_market,

        "raw_edge": raw_edge,
        "injury_penalty": data.get("injury_penalty", 0.0),
        "news_penalty": data.get("news_penalty", 0.0),

        "bookmaker": data.get("bookmaker", ""),
        "reason": " | ".join(reason_parts),
    }

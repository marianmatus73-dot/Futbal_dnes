from __future__ import annotations


def to_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def clamp(value: float, low: float = 0.01, high: float = 0.99) -> float:
    return max(low, min(high, value))


def implied_probability(odds: float) -> float:
    if odds <= 1:
        return 0.0
    return 1 / odds


def enrich_football_tip(tip: dict) -> dict:
    odds = to_float(tip.get("odds"), 0.0)

    base_prob = (
        to_float(tip.get("model_probability"))
        or to_float(tip.get("probability"))
        or to_float(tip.get("win_probability"))
    )

    market_prob = to_float(tip.get("market_probability"))

    if market_prob is None and odds > 1:
        market_prob = implied_probability(odds)

    if base_prob is None:
        base_prob = market_prob or 0.5

    elo_probability = to_float(tip.get("elo_probability"))
    if elo_probability is None:
        elo_probability = base_prob

    xg_probability = to_float(tip.get("xg_probability"))
    if xg_probability is None:
        home_xg = to_float(tip.get("home_xg"))
        away_xg = to_float(tip.get("away_xg"))

        if home_xg is not None and away_xg is not None:
            total = home_xg + away_xg
            xg_probability = home_xg / total if total > 0 else base_prob
        else:
            xg_probability = base_prob

    form_probability = to_float(tip.get("form_probability"))
    if form_probability is None:
        form_score = to_float(tip.get("form_score"))

        if form_score is not None:
            form_probability = clamp(0.5 + form_score * 0.15)
        else:
            form_probability = base_prob

    injury_penalty = to_float(tip.get("injury_penalty"), 0.0)
    news_penalty = to_float(tip.get("news_penalty"), 0.0)

    enriched = dict(tip)

    enriched["elo_probability"] = clamp(elo_probability)
    enriched["xg_probability"] = clamp(xg_probability)
    enriched["form_probability"] = clamp(form_probability)
    enriched["market_probability"] = clamp(market_prob or implied_probability(odds))
    enriched["injury_penalty"] = max(0.0, injury_penalty)
    enriched["news_penalty"] = max(0.0, news_penalty)

    reason = enriched.get("reason", "")

    extra_reason = (
        "Football consensus signals: "
        f"ELO {enriched['elo_probability']:.1%}, "
        f"xG {enriched['xg_probability']:.1%}, "
        f"Form {enriched['form_probability']:.1%}, "
        f"Market {enriched['market_probability']:.1%}"
    )

    enriched["reason"] = f"{reason} | {extra_reason}" if reason else extra_reason

    return enriched

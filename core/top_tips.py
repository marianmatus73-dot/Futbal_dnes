from __future__ import annotations


def select_top_tips(tips: list, limit: int = 5, min_confidence: int = 65) -> list:
    filtered = [
        tip for tip in tips
        if getattr(tip, "confidence", 0) >= min_confidence
        and getattr(tip, "edge", 0) > 0
    ]

    return sorted(
        filtered,
        key=lambda t: (
            getattr(t, "confidence", 0),
            getattr(t, "edge", 0),
            getattr(t, "stake_units", 0),
        ),
        reverse=True,
    )[:limit]


def select_telegram_tips(tips: list, min_confidence: int = 80) -> list:
    return [
        tip for tip in tips
        if getattr(tip, "confidence", 0) >= min_confidence
    ]

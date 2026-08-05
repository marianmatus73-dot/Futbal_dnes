from __future__ import annotations


def adaptive_weights(sports: dict) -> dict[str, float]:
    raw = {}

    for sport, metrics in sports.items():
        settled = float(metrics.get("settled_bets") or 0)
        quality = float(metrics.get("data_quality") or 0)
        yield_value = float(metrics.get("yield") or 0)

        score = (
            min(settled / 200, 1.0) * 0.45
            + quality * 0.35
            + (max(min(yield_value, 0.25), -0.25) + 0.25) * 0.20
        )
        raw[sport] = max(score, 0.01)

    total = sum(raw.values())
    return {
        sport: round(score / total, 4)
        for sport, score in raw.items()
    } if total else {}

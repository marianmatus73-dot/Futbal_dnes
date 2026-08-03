from __future__ import annotations

from typing import Any


def adaptive_weights(
    sports: dict[str, dict[str, Any]]
) -> dict[str, float]:
    raw: dict[str, float] = {}

    for sport, metrics in sports.items():
        settled = float(metrics.get("settled_bets") or 0)
        quality = float(metrics.get("data_quality") or 0)
        yield_value = metrics.get("yield")
        yield_component = max(min(float(yield_value or 0), 0.25), -0.25) + 0.25
        sample_component = min(settled / 200.0, 1.0)

        raw[sport] = max(
            0.01,
            sample_component * 0.45
            + quality * 0.35
            + yield_component * 0.20,
        )

    total = sum(raw.values())
    if total <= 0:
        return {
            sport: round(1.0 / len(raw), 4)
            for sport in raw
        } if raw else {}

    return {
        sport: round(score / total, 4)
        for sport, score in raw.items()
    }


def recommendations(
    sports: dict[str, dict[str, Any]]
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []

    for sport, metrics in sports.items():
        if metrics.get("settled_bets", 0) == 0:
            output.append({
                "sport": sport,
                "priority": "HIGH",
                "recommendation": "Connect settlement history for this sport.",
            })
        elif metrics.get("probability_samples", 0) == 0:
            output.append({
                "sport": sport,
                "priority": "HIGH",
                "recommendation": "Persist model probabilities for Brier/log-loss evaluation.",
            })
        elif metrics.get("maturity") != "READY":
            output.append({
                "sport": sport,
                "priority": "MEDIUM",
                "recommendation": "Collect more settled samples before full model tuning.",
            })

        yield_value = metrics.get("yield")
        if yield_value is not None and float(yield_value) < 0:
            output.append({
                "sport": sport,
                "priority": "HIGH",
                "recommendation": "Reduce model weight and review selection thresholds.",
            })

    return output

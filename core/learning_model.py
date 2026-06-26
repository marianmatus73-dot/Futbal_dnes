from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean


RESULTS_FILE = Path("exports/pro_tip_results.csv")
MODEL_FILE = Path("exports/learning_weights.csv")


DEFAULT_WEIGHTS = {
    "elo": 0.25,
    "xg": 0.25,
    "form": 0.20,
    "market": 0.20,
    "context": 0.10,
}


def load_learning_weights() -> dict[str, float]:
    if not MODEL_FILE.exists():
        return DEFAULT_WEIGHTS.copy()

    with MODEL_FILE.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return DEFAULT_WEIGHTS.copy()

    return {
        row["signal"]: float(row["weight"])
        for row in rows
        if row.get("signal") and row.get("weight")
    }


def save_learning_weights(weights: dict[str, float]) -> None:
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    with MODEL_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["signal", "weight"])
        writer.writeheader()

        for signal, weight in weights.items():
            writer.writerow(
                {
                    "signal": signal,
                    "weight": round(weight, 4),
                }
            )


def retrain_from_results() -> dict[str, float]:
    if not RESULTS_FILE.exists():
        return load_learning_weights()

    with RESULTS_FILE.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    graded = [
        row for row in rows
        if row.get("result") in {"WON", "LOST"}
    ]

    if len(graded) < 20:
        return load_learning_weights()

    weights = DEFAULT_WEIGHTS.copy()

    # jednoduchý safe learning základ
    roi = []

    for row in graded:
        odds = float(row.get("odds", 0) or 0)
        stake = float(row.get("stake", 1) or 1)
        result = row.get("result")

        profit = odds * stake - stake if result == "WON" else -stake
        roi.append(profit / stake)

    avg_roi = mean(roi)

    if avg_roi > 0.03:
        weights["elo"] += 0.02
        weights["xg"] += 0.02
        weights["market"] -= 0.02
    elif avg_roi < -0.03:
        weights["market"] += 0.03
        weights["elo"] -= 0.01
        weights["xg"] -= 0.01

    total = sum(weights.values())
    weights = {key: value / total for key, value in weights.items()}

    save_learning_weights(weights)
    return weights

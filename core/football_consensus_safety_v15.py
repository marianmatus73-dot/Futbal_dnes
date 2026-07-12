from __future__ import annotations

import math
from typing import Iterable


def calculate_consensus_safety(
    probabilities: Iterable[float | None],
    *,
    penalty_factor: float = 2.5,
) -> float:
    """
    Return model-agreement safety score from 0.0 to 1.0.

    1.0 means near-perfect agreement.
    Lower values mean stronger disagreement.

    Population standard deviation is used deliberately because the supplied
    probabilities represent the full active ensemble, not a sample.
    """
    valid: list[float] = []

    for value in probabilities:
        if value is None:
            continue

        try:
            probability = float(value)
        except (TypeError, ValueError):
            continue

        if (
            math.isnan(probability)
            or math.isinf(probability)
            or probability <= 0.0
            or probability >= 1.0
        ):
            continue

        valid.append(probability)

    if len(valid) < 2:
        return 1.0

    mean = sum(valid) / len(valid)
    variance = sum(
        (probability - mean) ** 2
        for probability in valid
    ) / len(valid)
    std_dev = math.sqrt(max(0.0, variance))

    safety = 1.0 - std_dev * max(
        0.0,
        float(penalty_factor),
    )

    return round(
        max(0.0, min(1.0, safety)),
        3,
    )

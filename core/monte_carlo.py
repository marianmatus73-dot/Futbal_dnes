from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class MonteCarloResult:
    simulations: int
    probability_input: float
    simulated_win_probability: float
    expected_profit_per_unit: float
    expected_roi_pct: float
    standard_deviation: float
    profit_ci_low: float
    profit_ci_high: float
    probability_of_profit: float
    probability_of_loss: float
    risk_score: int


def clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    return max(low, min(high, value))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 10_000) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    percentile = clamp(percentile, 0.0, 1.0)

    position = (len(ordered) - 1) * percentile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return ordered[lower_index]

    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    fraction = position - lower_index

    return lower_value + (upper_value - lower_value) * fraction


def _risk_score(
    probability_of_loss: float,
    expected_roi_pct: float,
    standard_deviation: float,
) -> int:
    """
    Risk score:
    1   = veľmi nízke riziko
    100 = veľmi vysoké riziko
    """

    loss_component = clamp(probability_of_loss) * 70.0
    volatility_component = clamp(standard_deviation / 2.0) * 25.0

    roi_bonus = clamp(
        max(expected_roi_pct, 0.0) / 20.0,
        0.0,
        1.0,
    ) * 15.0

    score = loss_component + volatility_component - roi_bonus

    return int(round(max(1.0, min(100.0, score))))


def simulate_single_bet(
    probability: float,
    odds: float,
    simulations: int | None = None,
    seed: int | None = None,
) -> MonteCarloResult:
    """
    Simuluje rovnaký tip veľakrát so stake 1 jednotka.

    Pri výhre:
        profit = odds - 1

    Pri prehre:
        profit = -1
    """

    probability = clamp(_safe_float(probability, 0.0), 0.001, 0.999)
    odds = _safe_float(odds, 0.0)

    if odds <= 1.0:
        raise ValueError("Odds must be greater than 1.0.")

    if simulations is None:
        simulations = _safe_int(
            os.getenv("MONTE_CARLO_SIMULATIONS", "10000"),
            10_000,
        )

    simulations = max(1_000, min(int(simulations), 100_000))

    rng = random.Random(seed)

    profits: list[float] = []
    wins = 0

    win_profit = odds - 1.0

    for _ in range(simulations):
        if rng.random() < probability:
            wins += 1
            profits.append(win_profit)
        else:
            profits.append(-1.0)

    simulated_win_probability = wins / simulations
    expected_profit = sum(profits) / simulations
    expected_roi_pct = expected_profit * 100.0

    variance = (
        sum(
            (profit - expected_profit) ** 2
            for profit in profits
        )
        / simulations
    )

    standard_deviation = math.sqrt(max(variance, 0.0))

    profit_ci_low = _percentile(profits, 0.025)
    profit_ci_high = _percentile(profits, 0.975)

    profitable_results = sum(1 for profit in profits if profit > 0)
    losing_results = sum(1 for profit in profits if profit < 0)

    probability_of_profit = profitable_results / simulations
    probability_of_loss = losing_results / simulations

    risk_score = _risk_score(
        probability_of_loss=probability_of_loss,
        expected_roi_pct=expected_roi_pct,
        standard_deviation=standard_deviation,
    )

    return MonteCarloResult(
        simulations=simulations,
        probability_input=round(probability, 6),
        simulated_win_probability=round(
            simulated_win_probability,
            6,
        ),
        expected_profit_per_unit=round(expected_profit, 6),
        expected_roi_pct=round(expected_roi_pct, 4),
        standard_deviation=round(standard_deviation, 6),
        profit_ci_low=round(profit_ci_low, 4),
        profit_ci_high=round(profit_ci_high, 4),
        probability_of_profit=round(probability_of_profit, 6),
        probability_of_loss=round(probability_of_loss, 6),
        risk_score=risk_score,
    )


def monte_carlo_score(result: MonteCarloResult) -> float:
    """
    Doplnkové skóre 0–100.

    Nepoužíva sa ako nová pravdepodobnosť zápasu.
    Používa sa iba na zoradenie a risk kontrolu.
    """

    roi_component = clamp(
        result.expected_roi_pct / 20.0,
        -1.0,
        1.0,
    )

    win_component = (
        result.simulated_win_probability - 0.5
    ) * 2.0

    risk_penalty = result.risk_score / 100.0

    score = (
        50.0
        + roi_component * 30.0
        + win_component * 20.0
        - risk_penalty * 15.0
    )

    return round(max(1.0, min(100.0, score)), 2)


def format_monte_carlo_reason(result: MonteCarloResult) -> str:
    return (
        f"Monte Carlo {result.simulations} sims: "
        f"win={result.simulated_win_probability:.1%}, "
        f"ROI={result.expected_roi_pct:.2f}%, "
        f"loss risk={result.probability_of_loss:.1%}, "
        f"risk score={result.risk_score}/100"
    )

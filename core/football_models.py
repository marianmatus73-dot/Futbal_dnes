from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial
from typing import Dict, Tuple


@dataclass
class MatchProbabilities:
    home: float
    draw: float
    away: float

    over_15: float
    under_15: float
    over_25: float
    under_25: float
    over_35: float
    under_35: float

    btts_yes: float
    btts_no: float

    expected_home_goals: float
    expected_away_goals: float

    correct_scores: Dict[str, float]


def clamp(
    value: float,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    return max(low, min(high, value))


def poisson_probability(
    goals: int,
    expected_goals: float,
) -> float:
    if goals < 0:
        return 0.0

    expected_goals = max(0.01, expected_goals)

    return (
        exp(-expected_goals)
        * (expected_goals ** goals)
        / factorial(goals)
    )


def dixon_coles_adjustment(
    home_goals: int,
    away_goals: int,
    expected_home_goals: float,
    expected_away_goals: float,
    rho: float = -0.08,
) -> float:
    """
    Dixon-Coles korekcia nízkych skóre.

    rho býva zvyčajne mierne záporné číslo.
    Bezpečný rozsah je približne -0.20 až 0.20.
    """

    rho = max(-0.25, min(0.25, rho))

    if home_goals == 0 and away_goals == 0:
        return 1.0 - (
            expected_home_goals
            * expected_away_goals
            * rho
        )

    if home_goals == 0 and away_goals == 1:
        return 1.0 + expected_home_goals * rho

    if home_goals == 1 and away_goals == 0:
        return 1.0 + expected_away_goals * rho

    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho

    return 1.0


def correct_score_matrix(
    expected_home_goals: float,
    expected_away_goals: float,
    max_goals: int = 8,
    rho: float = -0.08,
) -> Dict[Tuple[int, int], float]:
    expected_home_goals = max(0.05, expected_home_goals)
    expected_away_goals = max(0.05, expected_away_goals)
    max_goals = max(3, max_goals)

    matrix: Dict[Tuple[int, int], float] = {}
    total_probability = 0.0

    for home_goals in range(max_goals + 1):
        home_probability = poisson_probability(
            home_goals,
            expected_home_goals,
        )

        for away_goals in range(max_goals + 1):
            away_probability = poisson_probability(
                away_goals,
                expected_away_goals,
            )

            probability = (
                home_probability
                * away_probability
                * dixon_coles_adjustment(
                    home_goals,
                    away_goals,
                    expected_home_goals,
                    expected_away_goals,
                    rho,
                )
            )

            probability = max(0.0, probability)
            matrix[(home_goals, away_goals)] = probability
            total_probability += probability

    if total_probability <= 0:
        return matrix

    return {
        score: probability / total_probability
        for score, probability in matrix.items()
    }


def _total_goals_probability(
    matrix: Dict[Tuple[int, int], float],
    threshold: float,
    over: bool,
) -> float:
    probability = 0.0

    for (home_goals, away_goals), value in matrix.items():
        total_goals = home_goals + away_goals

        if over and total_goals > threshold:
            probability += value

        if not over and total_goals < threshold:
            probability += value

    return clamp(probability)


def calculate_match_probabilities(
    expected_home_goals: float,
    expected_away_goals: float,
    max_goals: int = 8,
    rho: float = -0.08,
    top_scores: int = 10,
) -> MatchProbabilities:
    matrix = correct_score_matrix(
        expected_home_goals=expected_home_goals,
        expected_away_goals=expected_away_goals,
        max_goals=max_goals,
        rho=rho,
    )

    home = 0.0
    draw = 0.0
    away = 0.0
    btts_yes = 0.0

    for (home_goals, away_goals), probability in matrix.items():
        if home_goals > away_goals:
            home += probability
        elif home_goals == away_goals:
            draw += probability
        else:
            away += probability

        if home_goals > 0 and away_goals > 0:
            btts_yes += probability

    ordered_scores = sorted(
        matrix.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:max(1, top_scores)]

    correct_scores = {
        f"{home_goals}-{away_goals}": round(
            probability,
            6,
        )
        for (home_goals, away_goals), probability
        in ordered_scores
    }

    over_15 = _total_goals_probability(
        matrix,
        threshold=1.5,
        over=True,
    )
    over_25 = _total_goals_probability(
        matrix,
        threshold=2.5,
        over=True,
    )
    over_35 = _total_goals_probability(
        matrix,
        threshold=3.5,
        over=True,
    )

    return MatchProbabilities(
        home=clamp(home),
        draw=clamp(draw),
        away=clamp(away),

        over_15=over_15,
        under_15=clamp(1.0 - over_15),

        over_25=over_25,
        under_25=clamp(1.0 - over_25),

        over_35=over_35,
        under_35=clamp(1.0 - over_35),

        btts_yes=clamp(btts_yes),
        btts_no=clamp(1.0 - btts_yes),

        expected_home_goals=max(
            0.05,
            expected_home_goals,
        ),
        expected_away_goals=max(
            0.05,
            expected_away_goals,
        ),

        correct_scores=correct_scores,
    )


def blend_1x2_with_market(
    model: MatchProbabilities,
    market_home: float,
    market_draw: float,
    market_away: float,
    market_weight: float = 0.35,
) -> MatchProbabilities:
    market_weight = clamp(market_weight)
    model_weight = 1.0 - market_weight

    total_market = (
        market_home
        + market_draw
        + market_away
    )

    if total_market > 0:
        market_home /= total_market
        market_draw /= total_market
        market_away /= total_market

    blended_home = (
        model.home * model_weight
        + market_home * market_weight
    )
    blended_draw = (
        model.draw * model_weight
        + market_draw * market_weight
    )
    blended_away = (
        model.away * model_weight
        + market_away * market_weight
    )

    total = (
        blended_home
        + blended_draw
        + blended_away
    )

    if total > 0:
        blended_home /= total
        blended_draw /= total
        blended_away /= total

    return MatchProbabilities(
        home=clamp(blended_home),
        draw=clamp(blended_draw),
        away=clamp(blended_away),

        over_15=model.over_15,
        under_15=model.under_15,
        over_25=model.over_25,
        under_25=model.under_25,
        over_35=model.over_35,
        under_35=model.under_35,

        btts_yes=model.btts_yes,
        btts_no=model.btts_no,

        expected_home_goals=model.expected_home_goals,
        expected_away_goals=model.expected_away_goals,

        correct_scores=model.correct_scores,
    )


def probability_for_selection(
    probabilities: MatchProbabilities,
    selection: str,
    home_team: str,
    away_team: str,
) -> float | None:
    normalized = selection.strip().lower()

    if normalized == home_team.strip().lower():
        return probabilities.home

    if normalized == away_team.strip().lower():
        return probabilities.away

    if normalized in {
        "draw",
        "x",
        "remíza",
        "remiza",
    }:
        return probabilities.draw

    return None

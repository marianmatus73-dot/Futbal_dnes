from __future__ import annotations

from dataclasses import dataclass
import random

from core.football_models import calculate_match_probabilities


@dataclass
class SimulationResult:
    home_probability: float
    draw_probability: float
    away_probability: float

    over25_probability: float
    under25_probability: float

    btts_probability: float

    expected_home_goals: float
    expected_away_goals: float

    average_home_goals: float
    average_away_goals: float

    simulations: int


def simulate_match(
    expected_home_goals: float,
    expected_away_goals: float,
    simulations: int = 10000,
) -> SimulationResult:

    model = calculate_match_probabilities(
        expected_home_goals,
        expected_away_goals,
    )

    home = 0
    draw = 0
    away = 0

    over25 = 0
    btts = 0

    total_home_goals = 0
    total_away_goals = 0

    score_items = list(model.correct_scores.items())
    probabilities = [p for _, p in score_items]

    total_probability = sum(probabilities)

    if total_probability <= 0:
        total_probability = 1.0

    probabilities = [
        p / total_probability
        for p in probabilities
    ]

    labels = [score for score, _ in score_items]

    for _ in range(simulations):

        score = random.choices(
            labels,
            weights=probabilities,
            k=1,
        )[0]

        home_goals, away_goals = map(
            int,
            score.split("-"),
        )

        total_home_goals += home_goals
        total_away_goals += away_goals

        if home_goals > away_goals:
            home += 1

        elif home_goals < away_goals:
            away += 1

        else:
            draw += 1

        if home_goals + away_goals >= 3:
            over25 += 1

        if home_goals > 0 and away_goals > 0:
            btts += 1

    return SimulationResult(
        home_probability=home / simulations,
        draw_probability=draw / simulations,
        away_probability=away / simulations,

        over25_probability=over25 / simulations,
        under25_probability=1 - over25 / simulations,

        btts_probability=btts / simulations,

        expected_home_goals=expected_home_goals,
        expected_away_goals=expected_away_goals,

        average_home_goals=total_home_goals / simulations,
        average_away_goals=total_away_goals / simulations,

        simulations=simulations,
    )

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple


MIN_XG = 0.05
MAX_XG = 6.00
DEFAULT_MAX_GOALS = 10


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def poisson_pmf(goals: int, expected_goals: float) -> float:
    if goals < 0:
        return 0.0

    expected_goals = clamp(float(expected_goals), MIN_XG, MAX_XG)

    return (
        math.exp(-expected_goals)
        * expected_goals ** goals
        / math.factorial(goals)
    )


@dataclass
class PoissonMarkets:
    home_win: float
    draw: float
    away_win: float

    over_05: float
    under_05: float
    over_15: float
    under_15: float
    over_25: float
    under_25: float
    over_35: float
    under_35: float
    over_45: float
    under_45: float

    btts_yes: float
    btts_no: float

    home_over_05: float
    home_over_15: float
    home_over_25: float

    away_over_05: float
    away_over_15: float
    away_over_25: float

    expected_home_goals: float
    expected_away_goals: float

    score_matrix: Dict[Tuple[int, int], float]
    top_correct_scores: Dict[str, float]


class FootballPoissonModel:
    def __init__(
        self,
        *,
        max_goals: int = DEFAULT_MAX_GOALS,
    ) -> None:
        self.max_goals = max(5, int(max_goals))

    def score_matrix(
        self,
        expected_home_goals: float,
        expected_away_goals: float,
    ) -> Dict[Tuple[int, int], float]:
        home_xg = clamp(
            float(expected_home_goals),
            MIN_XG,
            MAX_XG,
        )
        away_xg = clamp(
            float(expected_away_goals),
            MIN_XG,
            MAX_XG,
        )

        matrix: Dict[Tuple[int, int], float] = {}
        total = 0.0

        for home_goals in range(self.max_goals + 1):
            home_prob = poisson_pmf(home_goals, home_xg)

            for away_goals in range(self.max_goals + 1):
                probability = (
                    home_prob
                    * poisson_pmf(away_goals, away_xg)
                )

                matrix[(home_goals, away_goals)] = probability
                total += probability

        if total <= 0:
            return matrix

        return {
            score: probability / total
            for score, probability in matrix.items()
        }

    @staticmethod
    def _probability(
        matrix: Dict[Tuple[int, int], float],
        predicate,
    ) -> float:
        return clamp(
            sum(
                probability
                for score, probability in matrix.items()
                if predicate(*score)
            ),
            0.0,
            1.0,
        )

    def calculate(
        self,
        expected_home_goals: float,
        expected_away_goals: float,
        *,
        top_scores: int = 10,
    ) -> PoissonMarkets:
        matrix = self.score_matrix(
            expected_home_goals,
            expected_away_goals,
        )

        home_win = self._probability(
            matrix,
            lambda h, a: h > a,
        )
        draw = self._probability(
            matrix,
            lambda h, a: h == a,
        )
        away_win = self._probability(
            matrix,
            lambda h, a: h < a,
        )

        def over_line(line: float) -> float:
            return self._probability(
                matrix,
                lambda h, a: h + a > line,
            )

        over_05 = over_line(0.5)
        over_15 = over_line(1.5)
        over_25 = over_line(2.5)
        over_35 = over_line(3.5)
        over_45 = over_line(4.5)

        btts_yes = self._probability(
            matrix,
            lambda h, a: h > 0 and a > 0,
        )

        home_over_05 = self._probability(
            matrix,
            lambda h, a: h > 0.5,
        )
        home_over_15 = self._probability(
            matrix,
            lambda h, a: h > 1.5,
        )
        home_over_25 = self._probability(
            matrix,
            lambda h, a: h > 2.5,
        )

        away_over_05 = self._probability(
            matrix,
            lambda h, a: a > 0.5,
        )
        away_over_15 = self._probability(
            matrix,
            lambda h, a: a > 1.5,
        )
        away_over_25 = self._probability(
            matrix,
            lambda h, a: a > 2.5,
        )

        ordered = sorted(
            matrix.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:max(1, int(top_scores))]

        top_correct_scores = {
            f"{home_goals}-{away_goals}": round(
                probability,
                6,
            )
            for (home_goals, away_goals), probability
            in ordered
        }

        total_1x2 = home_win + draw + away_win

        if total_1x2 > 0:
            home_win /= total_1x2
            draw /= total_1x2
            away_win /= total_1x2

        return PoissonMarkets(
            home_win=home_win,
            draw=draw,
            away_win=away_win,

            over_05=over_05,
            under_05=1.0 - over_05,
            over_15=over_15,
            under_15=1.0 - over_15,
            over_25=over_25,
            under_25=1.0 - over_25,
            over_35=over_35,
            under_35=1.0 - over_35,
            over_45=over_45,
            under_45=1.0 - over_45,

            btts_yes=btts_yes,
            btts_no=1.0 - btts_yes,

            home_over_05=home_over_05,
            home_over_15=home_over_15,
            home_over_25=home_over_25,

            away_over_05=away_over_05,
            away_over_15=away_over_15,
            away_over_25=away_over_25,

            expected_home_goals=clamp(
                float(expected_home_goals),
                MIN_XG,
                MAX_XG,
            ),
            expected_away_goals=clamp(
                float(expected_away_goals),
                MIN_XG,
                MAX_XG,
            ),

            score_matrix=matrix,
            top_correct_scores=top_correct_scores,
        )


def calculate_poisson_markets(
    expected_home_goals: float,
    expected_away_goals: float,
    *,
    max_goals: int = DEFAULT_MAX_GOALS,
    top_scores: int = 10,
) -> PoissonMarkets:
    model = FootballPoissonModel(max_goals=max_goals)

    return model.calculate(
        expected_home_goals,
        expected_away_goals,
        top_scores=top_scores,
    )


def probability_for_market(
    result: PoissonMarkets,
    *,
    market: str,
    selection: str,
    home_team: str = "",
    away_team: str = "",
) -> float | None:
    normalized_market = str(market or "").strip().lower()
    normalized_selection = str(selection or "").strip().lower()

    if normalized_market in {"h2h", "1x2"}:
        if normalized_selection == str(home_team).strip().lower():
            return result.home_win

        if normalized_selection == str(away_team).strip().lower():
            return result.away_win

        if normalized_selection in {"draw", "x", "remíza", "remiza"}:
            return result.draw

    mapping = {
        ("totals", "over 0.5"): result.over_05,
        ("totals", "under 0.5"): result.under_05,
        ("totals", "over 1.5"): result.over_15,
        ("totals", "under 1.5"): result.under_15,
        ("totals", "over 2.5"): result.over_25,
        ("totals", "under 2.5"): result.under_25,
        ("totals", "over 3.5"): result.over_35,
        ("totals", "under 3.5"): result.under_35,
        ("totals", "over 4.5"): result.over_45,
        ("totals", "under 4.5"): result.under_45,
        ("btts", "yes"): result.btts_yes,
        ("btts", "no"): result.btts_no,
    }

    return mapping.get((normalized_market, normalized_selection))

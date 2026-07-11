from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

from core.football_poisson import (
    MAX_XG,
    MIN_XG,
    FootballPoissonModel,
    PoissonMarkets,
    clamp,
)


DEFAULT_RHO = -0.08
MIN_RHO = -0.25
MAX_RHO = 0.25


def dixon_coles_tau(
    home_goals: int,
    away_goals: int,
    expected_home_goals: float,
    expected_away_goals: float,
    rho: float = DEFAULT_RHO,
) -> float:
    """
    Dixon-Coles low-score correction.

    The correction is applied only to:
    0-0, 0-1, 1-0 and 1-1.
    """
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
    rho = clamp(float(rho), MIN_RHO, MAX_RHO)

    if home_goals == 0 and away_goals == 0:
        value = 1.0 - (home_xg * away_xg * rho)
    elif home_goals == 0 and away_goals == 1:
        value = 1.0 + (home_xg * rho)
    elif home_goals == 1 and away_goals == 0:
        value = 1.0 + (away_xg * rho)
    elif home_goals == 1 and away_goals == 1:
        value = 1.0 - rho
    else:
        value = 1.0

    return max(0.01, value)


def estimate_rho(
    *,
    observed_draw_rate: float | None = None,
    observed_low_score_rate: float | None = None,
    base_rho: float = DEFAULT_RHO,
) -> float:
    """
    Lightweight adaptive rho estimate.

    Negative rho generally increases 0-0 and 1-1 probability and reduces
    1-0 / 0-1 probability. The estimate is intentionally conservative.
    """
    rho = float(base_rho)

    if observed_draw_rate is not None:
        draw_rate = clamp(float(observed_draw_rate), 0.05, 0.60)
        rho -= (draw_rate - 0.26) * 0.35

    if observed_low_score_rate is not None:
        low_score_rate = clamp(
            float(observed_low_score_rate),
            0.05,
            0.90,
        )
        rho -= (low_score_rate - 0.38) * 0.20

    return clamp(rho, MIN_RHO, MAX_RHO)


@dataclass
class DixonColesMarkets:
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

    rho: float
    score_matrix: Dict[Tuple[int, int], float]
    top_correct_scores: Dict[str, float]

    poisson_home_win: float
    poisson_draw: float
    poisson_away_win: float

    draw_adjustment: float
    reason: str


class FootballDixonColesModel:
    def __init__(
        self,
        *,
        max_goals: int = 10,
        rho: float = DEFAULT_RHO,
    ) -> None:
        self.max_goals = max(5, int(max_goals))
        self.rho = clamp(float(rho), MIN_RHO, MAX_RHO)
        self.poisson = FootballPoissonModel(max_goals=self.max_goals)

    def corrected_matrix(
        self,
        expected_home_goals: float,
        expected_away_goals: float,
        *,
        rho: float | None = None,
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
        active_rho = self.rho if rho is None else clamp(
            float(rho),
            MIN_RHO,
            MAX_RHO,
        )

        base_matrix = self.poisson.score_matrix(home_xg, away_xg)

        corrected: Dict[Tuple[int, int], float] = {}
        total = 0.0

        for (home_goals, away_goals), probability in base_matrix.items():
            tau = dixon_coles_tau(
                home_goals,
                away_goals,
                home_xg,
                away_xg,
                active_rho,
            )

            adjusted_probability = max(
                0.0,
                probability * tau,
            )

            corrected[(home_goals, away_goals)] = adjusted_probability
            total += adjusted_probability

        if total <= 0:
            return base_matrix

        return {
            score: probability / total
            for score, probability in corrected.items()
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
        rho: float | None = None,
        top_scores: int = 10,
    ) -> DixonColesMarkets:
        active_rho = self.rho if rho is None else clamp(
            float(rho),
            MIN_RHO,
            MAX_RHO,
        )

        poisson_result: PoissonMarkets = self.poisson.calculate(
            expected_home_goals,
            expected_away_goals,
            top_scores=top_scores,
        )

        matrix = self.corrected_matrix(
            expected_home_goals,
            expected_away_goals,
            rho=active_rho,
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
            lambda h, a: h >= 1,
        )
        home_over_15 = self._probability(
            matrix,
            lambda h, a: h >= 2,
        )
        home_over_25 = self._probability(
            matrix,
            lambda h, a: h >= 3,
        )

        away_over_05 = self._probability(
            matrix,
            lambda h, a: a >= 1,
        )
        away_over_15 = self._probability(
            matrix,
            lambda h, a: a >= 2,
        )
        away_over_25 = self._probability(
            matrix,
            lambda h, a: a >= 3,
        )

        total_1x2 = home_win + draw + away_win

        if total_1x2 > 0:
            home_win /= total_1x2
            draw /= total_1x2
            away_win /= total_1x2

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
            for (home_goals, away_goals), probability in ordered
        }

        draw_adjustment = draw - poisson_result.draw

        reason = (
            f"Dixon-Coles: rho={active_rho:.3f}; "
            f"home={home_win:.3f}; draw={draw:.3f}; "
            f"away={away_win:.3f}; "
            f"draw_adjustment={draw_adjustment:+.4f}; "
            f"xG={expected_home_goals:.2f}-{expected_away_goals:.2f}"
        )

        return DixonColesMarkets(
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

            rho=active_rho,
            score_matrix=matrix,
            top_correct_scores=top_correct_scores,

            poisson_home_win=poisson_result.home_win,
            poisson_draw=poisson_result.draw,
            poisson_away_win=poisson_result.away_win,

            draw_adjustment=draw_adjustment,
            reason=reason,
        )


def calculate_dixon_coles_markets(
    expected_home_goals: float,
    expected_away_goals: float,
    *,
    rho: float = DEFAULT_RHO,
    max_goals: int = 10,
    top_scores: int = 10,
) -> DixonColesMarkets:
    model = FootballDixonColesModel(
        max_goals=max_goals,
        rho=rho,
    )

    return model.calculate(
        expected_home_goals,
        expected_away_goals,
        top_scores=top_scores,
    )


def probability_for_market(
    result: DixonColesMarkets,
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

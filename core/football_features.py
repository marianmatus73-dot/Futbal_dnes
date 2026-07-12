from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

from core.config import Settings
from core.football_dixon_coles import (
    DixonColesMarkets,
    calculate_dixon_coles_markets,
)
from core.football_elo import EloPrediction, predict_football_elo
from core.football_team_elo_v14 import predict_team_elo_v14
from core.football_market import FootballMarketSnapshot
from core.football_team_form import (
    FormPrediction,
    predict_football_form,
)
from core.football_xg import MatchXGEstimate, estimate_match_xg
from core.football_team_xg_v14 import predict_team_xg_v14
from core.football_ensemble_v14 import (
    build_football_ensemble_v14,
)
from core.football_consensus_safety_v15 import (
    calculate_consensus_safety,
)
from core.football_context_v15 import (
    build_football_context_v15,
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_selection(
    selection: str,
    home_team: str,
    away_team: str,
) -> str:
    normalized = str(selection or "").strip().lower()

    if normalized == str(home_team).strip().lower():
        return "HOME"

    if normalized == str(away_team).strip().lower():
        return "AWAY"

    if normalized in {"draw", "x", "remíza", "remiza"}:
        return "DRAW"

    return normalized.upper()


@dataclass
class FootballFeatures:
    sport: str
    league: str
    sport_key: str

    event: str
    home_team: str
    away_team: str
    commence_time: str

    selection: str
    selection_type: str
    bookmaker: str
    odds: float

    market_home_probability: float
    market_draw_probability: float
    market_away_probability: float
    market_selection_probability: float
    market_overround: float
    bookmaker_count: int

    xg_home: float
    xg_away: float
    xg_total: float
    xg_difference: float
    xg_home_reliability: float
    xg_away_reliability: float

    elo_home_probability: float
    elo_draw_probability: float
    elo_away_probability: float
    elo_selection_probability: float
    elo_difference: float
    elo_home_reliability: float
    elo_away_reliability: float

    form_home_probability: float
    form_draw_probability: float
    form_away_probability: float
    form_selection_probability: float
    form_difference: float
    form_home_reliability: float
    form_away_reliability: float

    dixon_home_probability: float
    dixon_draw_probability: float
    dixon_away_probability: float
    dixon_selection_probability: float
    dixon_rho: float
    dixon_draw_adjustment: float

    over_25_probability: float
    under_25_probability: float
    btts_yes_probability: float
    btts_no_probability: float

    model_consensus_probability: float
    model_dispersion: float
    consensus_safety: float
    competition_importance: float
    is_international: float
    is_knockout: float
    is_qualification: float
    season_stage: float
    context_reliability: float
    market_model_gap: float
    raw_edge: float

    home_advantage_elo: float
    home_advantage_xg: float
    league_weight: float
    bookmaker_weight: float
    sport_weight: float

    confidence_input: float
    reliability_input: float

    probability_source: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def numeric_vector(self) -> list[float]:
        """
        Stable feature order for a future football-only meta model.
        Do not change casually after training starts.
        """
        return [
            self.odds,
            self.market_home_probability,
            self.market_draw_probability,
            self.market_away_probability,
            self.market_selection_probability,
            self.market_overround,
            float(self.bookmaker_count),

            self.xg_home,
            self.xg_away,
            self.xg_total,
            self.xg_difference,
            self.xg_home_reliability,
            self.xg_away_reliability,

            self.elo_home_probability,
            self.elo_draw_probability,
            self.elo_away_probability,
            self.elo_selection_probability,
            self.elo_difference,
            self.elo_home_reliability,
            self.elo_away_reliability,

            self.form_home_probability,
            self.form_draw_probability,
            self.form_away_probability,
            self.form_selection_probability,
            self.form_difference,
            self.form_home_reliability,
            self.form_away_reliability,

            self.dixon_home_probability,
            self.dixon_draw_probability,
            self.dixon_away_probability,
            self.dixon_selection_probability,
            self.dixon_rho,
            self.dixon_draw_adjustment,

            self.over_25_probability,
            self.under_25_probability,
            self.btts_yes_probability,
            self.btts_no_probability,

            self.model_consensus_probability,
            self.model_dispersion,
            self.market_model_gap,
            self.raw_edge,

            self.home_advantage_elo,
            self.home_advantage_xg,
            self.league_weight,
            self.bookmaker_weight,
            self.sport_weight,

            self.confidence_input,
            self.reliability_input,
        ]


@dataclass
class FootballModelBundle:
    xg: MatchXGEstimate
    elo: EloPrediction
    form: FormPrediction
    dixon_coles: DixonColesMarkets


def _selection_probability(
    *,
    selection_type: str,
    home_probability: float,
    draw_probability: float,
    away_probability: float,
) -> float:
    if selection_type == "HOME":
        return home_probability

    if selection_type == "DRAW":
        return draw_probability

    if selection_type == "AWAY":
        return away_probability

    return 0.0


def _dispersion(values: list[float]) -> float:
    clean = [value for value in values if value > 0]

    if not clean:
        return 0.0

    average = sum(clean) / len(clean)
    variance = sum((value - average) ** 2 for value in clean) / len(clean)

    return variance ** 0.5


def build_model_bundle(
    settings: Settings,
    *,
    market: FootballMarketSnapshot,
    league_average_xg: float = 1.35,
    home_advantage_xg: float = 1.08,
    home_advantage_elo: float = 65.0,
    dixon_rho: float = -0.08,
) -> FootballModelBundle:
    legacy_xg = estimate_match_xg(
        settings,
        home_team=market.home_team,
        away_team=market.away_team,
        league=market.league,
        league_average_xg=league_average_xg,
        home_advantage=home_advantage_xg,
    )

    v14_xg = predict_team_xg_v14(
        settings,
        league=market.league,
        home_team=market.home_team,
        away_team=market.away_team,
        league_average_xg=league_average_xg,
        home_advantage=home_advantage_xg,
    )

    v14_weight = clamp(
        v14_xg.combined_reliability
        * (1.0 - v14_xg.uncertainty_penalty * 0.50),
        0.0,
        0.85,
    )

    blended_home_xg = (
        v14_xg.home_xg * v14_weight
        + legacy_xg.home_expected_goals * (1.0 - v14_weight)
    )
    blended_away_xg = (
        v14_xg.away_xg * v14_weight
        + legacy_xg.away_expected_goals * (1.0 - v14_weight)
    )

    xg = MatchXGEstimate(
        home_team=market.home_team,
        away_team=market.away_team,
        league=market.league,
        home_expected_goals=clamp(blended_home_xg, 0.15, 4.50),
        away_expected_goals=clamp(blended_away_xg, 0.15, 4.50),
        home_attack_component=(
            legacy_xg.home_attack_component * (1.0 - v14_weight)
            + v14_xg.home_attack_strength * v14_weight
        ),
        home_defense_component=(
            legacy_xg.home_defense_component * (1.0 - v14_weight)
            + v14_xg.home_defense_strength * v14_weight
        ),
        away_attack_component=(
            legacy_xg.away_attack_component * (1.0 - v14_weight)
            + v14_xg.away_attack_strength * v14_weight
        ),
        away_defense_component=(
            legacy_xg.away_defense_component * (1.0 - v14_weight)
            + v14_xg.away_defense_strength * v14_weight
        ),
        home_reliability=clamp(
            max(
                legacy_xg.home_reliability,
                v14_xg.home_reliability * v14_weight,
            ),
            0.0,
            1.0,
        ),
        away_reliability=clamp(
            max(
                legacy_xg.away_reliability,
                v14_xg.away_reliability * v14_weight,
            ),
            0.0,
            1.0,
        ),
        reason=(
            f"v14 xG blend: weight={v14_weight:.3f}; "
            f"legacy={legacy_xg.home_expected_goals:.3f}-"
            f"{legacy_xg.away_expected_goals:.3f}; "
            f"team_v14={v14_xg.home_xg:.3f}-{v14_xg.away_xg:.3f}; "
            f"uncertainty={v14_xg.uncertainty_penalty:.3f}; "
            f"{v14_xg.reason}"
        ),
    )

    legacy_elo = predict_football_elo(
        settings,
        home_team=market.home_team,
        away_team=market.away_team,
        league=market.league,
        home_advantage_elo=home_advantage_elo,
    )

    v14_elo = predict_team_elo_v14(
        settings,
        league=market.league,
        home_team=market.home_team,
        away_team=market.away_team,
        home_advantage_elo=home_advantage_elo,
    )

    v14_elo_weight = clamp(
        v14_elo.combined_reliability
        * (1.0 - v14_elo.uncertainty_penalty * 0.50),
        0.0,
        0.85,
    )

    blended_home_probability = (
        v14_elo.home_probability * v14_elo_weight
        + legacy_elo.home_probability * (1.0 - v14_elo_weight)
    )
    blended_draw_probability = (
        v14_elo.draw_probability * v14_elo_weight
        + legacy_elo.draw_probability * (1.0 - v14_elo_weight)
    )
    blended_away_probability = (
        v14_elo.away_probability * v14_elo_weight
        + legacy_elo.away_probability * (1.0 - v14_elo_weight)
    )

    elo_total = (
        blended_home_probability
        + blended_draw_probability
        + blended_away_probability
    )

    if elo_total <= 0:
        blended_home_probability = legacy_elo.home_probability
        blended_draw_probability = legacy_elo.draw_probability
        blended_away_probability = legacy_elo.away_probability
        elo_total = 1.0

    elo = EloPrediction(
        home_team=market.home_team,
        away_team=market.away_team,
        league=market.league,
        home_probability=blended_home_probability / elo_total,
        draw_probability=blended_draw_probability / elo_total,
        away_probability=blended_away_probability / elo_total,
        home_effective_elo=(
            v14_elo.home_rating * v14_elo_weight
            + legacy_elo.home_effective_elo * (1.0 - v14_elo_weight)
        ),
        away_effective_elo=(
            v14_elo.away_rating * v14_elo_weight
            + legacy_elo.away_effective_elo * (1.0 - v14_elo_weight)
        ),
        elo_difference=(
            v14_elo.rating_difference * v14_elo_weight
            + legacy_elo.elo_difference * (1.0 - v14_elo_weight)
        ),
        home_reliability=clamp(
            max(
                legacy_elo.home_reliability,
                v14_elo.home_reliability * v14_elo_weight,
            ),
            0.0,
            1.0,
        ),
        away_reliability=clamp(
            max(
                legacy_elo.away_reliability,
                v14_elo.away_reliability * v14_elo_weight,
            ),
            0.0,
            1.0,
        ),
        reason=(
            f"v14 ELO blend: weight={v14_elo_weight:.3f}; "
            f"legacy_diff={legacy_elo.elo_difference:.1f}; "
            f"team_v14_diff={v14_elo.rating_difference:.1f}; "
            f"uncertainty={v14_elo.uncertainty_penalty:.3f}; "
            f"{v14_elo.reason}"
        ),
    )

    form = predict_football_form(
        settings,
        home_team=market.home_team,
        away_team=market.away_team,
        league=market.league,
    )

    dixon_coles = calculate_dixon_coles_markets(
        xg.home_expected_goals,
        xg.away_expected_goals,
        rho=dixon_rho,
    )

    return FootballModelBundle(
        xg=xg,
        elo=elo,
        form=form,
        dixon_coles=dixon_coles,
    )


def build_football_features(
    settings: Settings,
    *,
    market: FootballMarketSnapshot,
    selection: str,
    odds: float,
    bookmaker: str,
    league_weight: float = 1.0,
    bookmaker_weight: float = 1.0,
    sport_weight: float = 1.0,
    league_average_xg: float = 1.35,
    home_advantage_xg: float = 1.08,
    home_advantage_elo: float = 65.0,
    dixon_rho: float = -0.08,
) -> FootballFeatures:
    selection_type = normalize_selection(
        selection,
        market.home_team,
        market.away_team,
    )

    bundle = build_model_bundle(
        settings,
        market=market,
        league_average_xg=league_average_xg,
        home_advantage_xg=home_advantage_xg,
        home_advantage_elo=home_advantage_elo,
        dixon_rho=dixon_rho,
    )

    market_selection_probability = _selection_probability(
        selection_type=selection_type,
        home_probability=market.consensus_home,
        draw_probability=market.consensus_draw,
        away_probability=market.consensus_away,
    )

    elo_selection_probability = _selection_probability(
        selection_type=selection_type,
        home_probability=bundle.elo.home_probability,
        draw_probability=bundle.elo.draw_probability,
        away_probability=bundle.elo.away_probability,
    )

    form_selection_probability = _selection_probability(
        selection_type=selection_type,
        home_probability=bundle.form.home_form_probability,
        draw_probability=bundle.form.draw_form_probability,
        away_probability=bundle.form.away_form_probability,
    )

    dixon_selection_probability = _selection_probability(
        selection_type=selection_type,
        home_probability=bundle.dixon_coles.home_win,
        draw_probability=bundle.dixon_coles.draw,
        away_probability=bundle.dixon_coles.away_win,
    )

    model_probabilities = [
        elo_selection_probability,
        form_selection_probability,
        dixon_selection_probability,
    ]

    xg_reliability = clamp(
        (
            bundle.xg.home_reliability
            + bundle.xg.away_reliability
        )
        / 2.0,
        0.0,
        1.0,
    )
    elo_reliability = clamp(
        (
            bundle.elo.home_reliability
            + bundle.elo.away_reliability
        )
        / 2.0,
        0.0,
        1.0,
    )
    form_reliability = clamp(
        (
            bundle.form.home_reliability
            + bundle.form.away_reliability
        )
        / 2.0,
        0.0,
        1.0,
    )

    reliability_input = clamp(
        (
            xg_reliability
            + elo_reliability
            + form_reliability
        )
        / 3.0,
        0.0,
        1.0,
    )

    ensemble = build_football_ensemble_v14(
        market_probability=market_selection_probability,
        elo_probability=elo_selection_probability,
        form_probability=form_selection_probability,
        dixon_probability=dixon_selection_probability,
        elo_reliability=elo_reliability,
        form_reliability=form_reliability,
        xg_reliability=xg_reliability,
        market_overround=market.overround,
        bookmaker_count=market.bookmaker_count,
        league_calibration_reliability=0.0,
    )

    context = build_football_context_v15(
        league=market.league,
        sport_key=market.sport_key,
        event=market.event,
        commence_time=market.commence_time,
    )

    model_consensus_probability = ensemble.probability
    model_dispersion = ensemble.dispersion
    consensus_safety = calculate_consensus_safety(
        [
            elo_selection_probability,
            form_selection_probability,
            dixon_selection_probability,
            market_selection_probability,
        ]
    )
    market_model_gap = (
        model_consensus_probability
        - market_selection_probability
    )
    raw_edge = model_consensus_probability * float(odds) - 1.0

    confidence_input = clamp(
        1.0
        - model_dispersion * 3.0
        + consensus_safety * 0.12
        + context.context_reliability * 0.04
        + reliability_input * 0.35
        - market.overround * 0.50,
        0.0,
        1.0,
    )

    reason = (
        f"football features: selection={selection_type}; "
        f"market={market_selection_probability:.4f}; "
        f"elo={elo_selection_probability:.4f}; "
        f"form={form_selection_probability:.4f}; "
        f"dixon={dixon_selection_probability:.4f}; "
        f"consensus={model_consensus_probability:.4f}; "
        f"dispersion={model_dispersion:.4f}; "
        f"consensus_safety={consensus_safety:.3f}; "
        f"{context.reason}; "
        f"reliability={reliability_input:.3f}; "
        f"raw_edge={raw_edge:.4f}; "
        f"{ensemble.reason}; "
        f"xg_reason={bundle.xg.reason}; "
        f"elo_reason={bundle.elo.reason}"
    )

    return FootballFeatures(
        sport="football",
        league=market.league,
        sport_key=market.sport_key,

        event=market.event,
        home_team=market.home_team,
        away_team=market.away_team,
        commence_time=market.commence_time,

        selection=selection,
        selection_type=selection_type,
        bookmaker=bookmaker,
        odds=float(odds),

        market_home_probability=market.consensus_home,
        market_draw_probability=market.consensus_draw,
        market_away_probability=market.consensus_away,
        market_selection_probability=market_selection_probability,
        market_overround=market.overround,
        bookmaker_count=market.bookmaker_count,

        xg_home=bundle.xg.home_expected_goals,
        xg_away=bundle.xg.away_expected_goals,
        xg_total=(
            bundle.xg.home_expected_goals
            + bundle.xg.away_expected_goals
        ),
        xg_difference=(
            bundle.xg.home_expected_goals
            - bundle.xg.away_expected_goals
        ),
        xg_home_reliability=bundle.xg.home_reliability,
        xg_away_reliability=bundle.xg.away_reliability,

        elo_home_probability=bundle.elo.home_probability,
        elo_draw_probability=bundle.elo.draw_probability,
        elo_away_probability=bundle.elo.away_probability,
        elo_selection_probability=elo_selection_probability,
        elo_difference=bundle.elo.elo_difference,
        elo_home_reliability=bundle.elo.home_reliability,
        elo_away_reliability=bundle.elo.away_reliability,

        form_home_probability=bundle.form.home_form_probability,
        form_draw_probability=bundle.form.draw_form_probability,
        form_away_probability=bundle.form.away_form_probability,
        form_selection_probability=form_selection_probability,
        form_difference=(
            bundle.form.home_form_score
            - bundle.form.away_form_score
        ),
        form_home_reliability=bundle.form.home_reliability,
        form_away_reliability=bundle.form.away_reliability,

        dixon_home_probability=bundle.dixon_coles.home_win,
        dixon_draw_probability=bundle.dixon_coles.draw,
        dixon_away_probability=bundle.dixon_coles.away_win,
        dixon_selection_probability=dixon_selection_probability,
        dixon_rho=bundle.dixon_coles.rho,
        dixon_draw_adjustment=bundle.dixon_coles.draw_adjustment,

        over_25_probability=bundle.dixon_coles.over_25,
        under_25_probability=bundle.dixon_coles.under_25,
        btts_yes_probability=bundle.dixon_coles.btts_yes,
        btts_no_probability=bundle.dixon_coles.btts_no,

        model_consensus_probability=model_consensus_probability,
        model_dispersion=model_dispersion,
        consensus_safety=consensus_safety,
        competition_importance=context.competition_importance,
        is_international=context.is_international,
        is_knockout=context.is_knockout,
        is_qualification=context.is_qualification,
        season_stage=context.season_stage,
        context_reliability=context.context_reliability,
        market_model_gap=market_model_gap,
        raw_edge=raw_edge,

        home_advantage_elo=home_advantage_elo,
        home_advantage_xg=home_advantage_xg,
        league_weight=league_weight,
        bookmaker_weight=bookmaker_weight,
        sport_weight=sport_weight,

        confidence_input=confidence_input,
        reliability_input=reliability_input,

        probability_source="FOOTBALL_V13_CONSENSUS",
        reason=reason,
    )

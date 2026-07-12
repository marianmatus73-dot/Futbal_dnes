from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import Settings
from core.football_features import FootballFeatures
from core.football_meta import FootballMetaPrediction


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _signal(
    label: str,
    value: Any,
    *,
    impact: str,
) -> dict[str, Any]:
    return {
        "label": label,
        "value": value,
        "impact": impact,
    }


@dataclass
class FootballExplanationV15:
    source_hash: str
    decision: str
    verdict: str
    confidence: float
    risk: str
    positive_signals: list[dict[str, Any]]
    negative_signals: list[dict[str, Any]]
    neutral_signals: list[dict[str, Any]]
    summary: str


class FootballExplainabilityV15:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db_file = Path(settings.db_file or "bets.db")
        self.init_db()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def init_db(self) -> None:
        self.db_file.parent.mkdir(parents=True, exist_ok=True)

        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS football_explainability_v15 (
                    source_hash TEXT PRIMARY KEY,
                    sport_key TEXT NOT NULL DEFAULT '',
                    league TEXT NOT NULL DEFAULT '',
                    event TEXT NOT NULL DEFAULT '',
                    selection TEXT NOT NULL DEFAULT '',
                    bookmaker TEXT NOT NULL DEFAULT '',
                    commence_time TEXT NOT NULL DEFAULT '',

                    decision TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    final_probability REAL,
                    market_probability REAL,
                    raw_edge REAL,
                    adjusted_edge REAL,
                    confidence REAL,
                    risk TEXT,

                    positive_signals TEXT NOT NULL,
                    negative_signals TEXT NOT NULL,
                    neutral_signals TEXT NOT NULL,
                    summary TEXT NOT NULL,

                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS ix_football_explainability_v15
                ON football_explainability_v15 (
                    decision,
                    league,
                    commence_time
                )
                """
            )

            conn.commit()

    @staticmethod
    def build(
        *,
        source_hash: str,
        features: FootballFeatures,
        meta_prediction: FootballMetaPrediction,
        decision: str,
        adjusted_edge: float,
        confidence: float,
        risk: str,
        rejection_reason: str = "",
    ) -> FootballExplanationV15:
        positive: list[dict[str, Any]] = []
        negative: list[dict[str, Any]] = []
        neutral: list[dict[str, Any]] = []

        if features.raw_edge >= 0.08:
            positive.append(
                _signal(
                    "Raw edge",
                    round(features.raw_edge, 4),
                    impact="positive",
                )
            )
        elif features.raw_edge <= 0:
            negative.append(
                _signal(
                    "Raw edge",
                    round(features.raw_edge, 4),
                    impact="negative",
                )
            )
        else:
            neutral.append(
                _signal(
                    "Raw edge",
                    round(features.raw_edge, 4),
                    impact="neutral",
                )
            )

        if features.consensus_safety >= 0.85:
            positive.append(
                _signal(
                    "Consensus safety",
                    round(features.consensus_safety, 3),
                    impact="positive",
                )
            )
        elif features.consensus_safety < 0.60:
            negative.append(
                _signal(
                    "Consensus safety",
                    round(features.consensus_safety, 3),
                    impact="negative",
                )
            )
        else:
            neutral.append(
                _signal(
                    "Consensus safety",
                    round(features.consensus_safety, 3),
                    impact="neutral",
                )
            )

        if features.market_model_gap >= 0.04:
            positive.append(
                _signal(
                    "Model-market gap",
                    round(features.market_model_gap, 4),
                    impact="positive",
                )
            )
        elif features.market_model_gap <= 0:
            negative.append(
                _signal(
                    "Model-market gap",
                    round(features.market_model_gap, 4),
                    impact="negative",
                )
            )
        else:
            neutral.append(
                _signal(
                    "Model-market gap",
                    round(features.market_model_gap, 4),
                    impact="neutral",
                )
            )

        if abs(features.elo_difference) >= 50:
            positive.append(
                _signal(
                    "ELO separation",
                    round(features.elo_difference, 1),
                    impact="positive",
                )
            )
        else:
            neutral.append(
                _signal(
                    "ELO separation",
                    round(features.elo_difference, 1),
                    impact="neutral",
                )
            )

        if abs(features.form_difference) >= 0.08:
            positive.append(
                _signal(
                    "Form separation",
                    round(features.form_difference, 4),
                    impact="positive",
                )
            )
        else:
            neutral.append(
                _signal(
                    "Form separation",
                    round(features.form_difference, 4),
                    impact="neutral",
                )
            )

        if features.xg_home_reliability <= 0.05 and features.xg_away_reliability <= 0.05:
            negative.append(
                _signal(
                    "Historical xG",
                    "unavailable",
                    impact="negative",
                )
            )
        else:
            positive.append(
                _signal(
                    "xG reliability",
                    round(
                        (
                            features.xg_home_reliability
                            + features.xg_away_reliability
                        )
                        / 2.0,
                        3,
                    ),
                    impact="positive",
                )
            )

        if features.competition_importance >= 0.85:
            positive.append(
                _signal(
                    "Competition importance",
                    round(features.competition_importance, 2),
                    impact="positive",
                )
            )
        else:
            neutral.append(
                _signal(
                    "Competition importance",
                    round(features.competition_importance, 2),
                    impact="neutral",
                )
            )

        if features.is_knockout:
            positive.append(
                _signal(
                    "Knockout match",
                    True,
                    impact="positive",
                )
            )

        if features.is_qualification:
            neutral.append(
                _signal(
                    "Qualification match",
                    True,
                    impact="neutral",
                )
            )

        if features.model_dispersion >= 0.12:
            negative.append(
                _signal(
                    "Model dispersion",
                    round(features.model_dispersion, 4),
                    impact="negative",
                )
            )
        else:
            positive.append(
                _signal(
                    "Model dispersion",
                    round(features.model_dispersion, 4),
                    impact="positive",
                )
            )

        if meta_prediction.source != "META_MODEL":
            neutral.append(
                _signal(
                    "Probability source",
                    meta_prediction.source,
                    impact="neutral",
                )
            )

        if rejection_reason:
            negative.append(
                _signal(
                    "Decision rule",
                    rejection_reason,
                    impact="negative",
                )
            )

        verdict = (
            "PASS"
            if decision.upper() == "PASS"
            else "REJECT"
        )

        summary = (
            f"{verdict}: positive={len(positive)}, "
            f"negative={len(negative)}, "
            f"neutral={len(neutral)}, "
            f"confidence={confidence:.3f}, "
            f"risk={risk}"
        )

        return FootballExplanationV15(
            source_hash=source_hash,
            decision=decision.upper(),
            verdict=verdict,
            confidence=confidence,
            risk=risk,
            positive_signals=positive,
            negative_signals=negative,
            neutral_signals=neutral,
            summary=summary,
        )

    def save(
        self,
        *,
        explanation: FootballExplanationV15,
        features: FootballFeatures,
        meta_prediction: FootballMetaPrediction,
        adjusted_edge: float,
    ) -> None:
        timestamp = now_utc()

        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO football_explainability_v15 (
                    source_hash,
                    sport_key,
                    league,
                    event,
                    selection,
                    bookmaker,
                    commence_time,
                    decision,
                    verdict,
                    final_probability,
                    market_probability,
                    raw_edge,
                    adjusted_edge,
                    confidence,
                    risk,
                    positive_signals,
                    negative_signals,
                    neutral_signals,
                    summary,
                    created_at,
                    updated_at
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(source_hash) DO UPDATE SET
                    decision=excluded.decision,
                    verdict=excluded.verdict,
                    final_probability=excluded.final_probability,
                    market_probability=excluded.market_probability,
                    raw_edge=excluded.raw_edge,
                    adjusted_edge=excluded.adjusted_edge,
                    confidence=excluded.confidence,
                    risk=excluded.risk,
                    positive_signals=excluded.positive_signals,
                    negative_signals=excluded.negative_signals,
                    neutral_signals=excluded.neutral_signals,
                    summary=excluded.summary,
                    updated_at=excluded.updated_at
                """,
                (
                    explanation.source_hash,
                    features.sport_key,
                    features.league,
                    features.event,
                    features.selection,
                    features.bookmaker,
                    features.commence_time,
                    explanation.decision,
                    explanation.verdict,
                    meta_prediction.probability,
                    features.market_selection_probability,
                    features.raw_edge,
                    adjusted_edge,
                    explanation.confidence,
                    explanation.risk,
                    json.dumps(
                        explanation.positive_signals,
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        explanation.negative_signals,
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        explanation.neutral_signals,
                        ensure_ascii=False,
                    ),
                    explanation.summary,
                    timestamp,
                    timestamp,
                ),
            )
            conn.commit()


def explain_and_save_football_decision_v15(
    settings: Settings,
    *,
    source_hash: str,
    features: FootballFeatures,
    meta_prediction: FootballMetaPrediction,
    decision: str,
    adjusted_edge: float,
    confidence: float,
    risk: str,
    rejection_reason: str = "",
) -> FootballExplanationV15:
    engine = FootballExplainabilityV15(settings)
    explanation = engine.build(
        source_hash=source_hash,
        features=features,
        meta_prediction=meta_prediction,
        decision=decision,
        adjusted_edge=adjusted_edge,
        confidence=confidence,
        risk=risk,
        rejection_reason=rejection_reason,
    )
    engine.save(
        explanation=explanation,
        features=features,
        meta_prediction=meta_prediction,
        adjusted_edge=adjusted_edge,
    )
    return explanation

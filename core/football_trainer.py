from __future__ import annotations

import json
import math
import pickle
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.football_meta import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    FootballMetaMetadata,
    save_football_meta_metadata,
)


MIN_TRAINING_SAMPLES = 80
DEFAULT_TEST_FRACTION = 0.20
RANDOM_STATE = 42


FEATURE_ORDER = [
    "odds",
    "market_home_probability",
    "market_draw_probability",
    "market_away_probability",
    "market_selection_probability",
    "market_overround",
    "bookmaker_count",
    "xg_home",
    "xg_away",
    "xg_total",
    "xg_difference",
    "xg_home_reliability",
    "xg_away_reliability",
    "elo_home_probability",
    "elo_draw_probability",
    "elo_away_probability",
    "elo_selection_probability",
    "elo_difference",
    "elo_home_reliability",
    "elo_away_reliability",
    "form_home_probability",
    "form_draw_probability",
    "form_away_probability",
    "form_selection_probability",
    "form_difference",
    "form_home_reliability",
    "form_away_reliability",
    "dixon_home_probability",
    "dixon_draw_probability",
    "dixon_away_probability",
    "dixon_selection_probability",
    "dixon_rho",
    "dixon_draw_adjustment",
    "over_25_probability",
    "under_25_probability",
    "btts_yes_probability",
    "btts_no_probability",
    "model_consensus_probability",
    "model_dispersion",
    "consensus_safety",
    "market_model_gap",
    "raw_edge",
    "home_advantage_elo",
    "home_advantage_xg",
    "league_weight",
    "bookmaker_weight",
    "sport_weight",
    "confidence_input",
    "reliability_input",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if math.isnan(result) or math.isinf(result):
        return default

    return result


def normalize_result(value: Any) -> str:
    result = str(value or "").strip().upper()

    if result == "V":
        return "WON"

    if result == "P":
        return "LOST"

    return result


def _table_exists(
    conn: sqlite3.Connection,
    table_name: str,
) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type='table' AND name=?
        """,
        (table_name,),
    ).fetchone()

    return row is not None


def _column_names(
    conn: sqlite3.Connection,
    table_name: str,
) -> set[str]:
    return {
        str(row[1])
        for row in conn.execute(
            f"PRAGMA table_info({table_name})"
        ).fetchall()
    }


def _feature_value(
    row: sqlite3.Row,
    column: str,
) -> float:
    try:
        return safe_float(row[column], 0.0)
    except (KeyError, IndexError):
        return 0.0


def load_training_data(
    db_file: str | Path,
) -> tuple[list[list[float]], list[int]]:
    """
    Loads football-only settled rows.

    Preferred source:
        football_feature_history

    Fallback source:
        sport_bets, using the columns that exist there and filling the
        remaining v13 features with zeros.
    """
    db_path = Path(db_file)

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        if _table_exists(conn, "football_feature_history"):
            columns = _column_names(
                conn,
                "football_feature_history",
            )

            required = {"result", *FEATURE_ORDER}
            missing = required - columns

            if missing:
                raise RuntimeError(
                    "football_feature_history is missing columns: "
                    + ", ".join(sorted(missing))
                )

            rows = conn.execute(
                """
                SELECT *
                FROM football_feature_history
                WHERE UPPER(result) IN ('WON','LOST','V','P')
                ORDER BY id
                """
            ).fetchall()

            features = [
                [
                    _feature_value(row, column)
                    for column in FEATURE_ORDER
                ]
                for row in rows
            ]
            labels = [
                1 if normalize_result(row["result"]) == "WON" else 0
                for row in rows
            ]

            return features, labels

        if not _table_exists(conn, "sport_bets"):
            return [], []

        columns = _column_names(conn, "sport_bets")

        rows = conn.execute(
            """
            SELECT *
            FROM sport_bets
            WHERE sport='football'
              AND UPPER(result) IN ('WON','LOST','V','P')
            ORDER BY id
            """
        ).fetchall()

        features: list[list[float]] = []
        labels: list[int] = []

        for row in rows:
            odds = (
                _feature_value(row, "odds")
                if "odds" in columns
                else 0.0
            )
            prob_market = (
                _feature_value(row, "prob_market")
                if "prob_market" in columns
                else 0.0
            )
            prob_final = (
                _feature_value(row, "prob_final")
                if "prob_final" in columns
                else prob_market
            )
            edge = (
                _feature_value(row, "edge")
                if "edge" in columns
                else prob_final * odds - 1.0
            )
            score = (
                _feature_value(row, "score")
                if "score" in columns
                else edge * 100.0
            )

            row_features = {
                name: 0.0
                for name in FEATURE_ORDER
            }

            row_features.update(
                {
                    "odds": odds,
                    "market_selection_probability": prob_market,
                    "model_consensus_probability": prob_final,
                    "market_model_gap": prob_final - prob_market,
                    "raw_edge": edge,
                    "confidence_input": max(
                        0.0,
                        min(1.0, score / 100.0),
                    ),
                }
            )

            features.append(
                [
                    safe_float(row_features[name], 0.0)
                    for name in FEATURE_ORDER
                ]
            )
            labels.append(
                1 if normalize_result(row["result"]) == "WON" else 0
            )

        return features, labels


def ensure_feature_history_table(
    db_file: str | Path,
) -> None:
    db_path = Path(db_file)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    feature_columns = ",\n".join(
        f"{name} REAL NOT NULL DEFAULT 0.0"
        for name in FEATURE_ORDER
    )

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS football_feature_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                created_at TEXT NOT NULL,
                settled_at TEXT,
                sport_key TEXT NOT NULL DEFAULT '',
                league TEXT NOT NULL DEFAULT '',
                event TEXT NOT NULL DEFAULT '',
                selection TEXT NOT NULL DEFAULT '',
                bookmaker TEXT NOT NULL DEFAULT '',
                commence_time TEXT NOT NULL DEFAULT '',

                {feature_columns},

                result TEXT NOT NULL DEFAULT 'OPEN',
                source_hash TEXT UNIQUE
            )
            """
        )

        existing_columns = _column_names(
            conn,
            "football_feature_history",
        )

        for feature_name in FEATURE_ORDER:
            if feature_name not in existing_columns:
                conn.execute(
                    f"""
                    ALTER TABLE football_feature_history
                    ADD COLUMN {feature_name}
                    REAL NOT NULL DEFAULT 0.0
                    """
                )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS ix_football_feature_history_result
            ON football_feature_history (result, league, created_at)
            """
        )

        conn.commit()


def _classification_metrics(
    y_true: list[int],
    probabilities: list[float],
) -> dict[str, float]:
    if not y_true:
        return {
            "accuracy": 0.0,
            "brier_score": 1.0,
            "log_loss": 99.0,
        }

    predictions = [
        1 if probability >= 0.50 else 0
        for probability in probabilities
    ]

    accuracy = sum(
        int(predicted == actual)
        for predicted, actual in zip(predictions, y_true)
    ) / len(y_true)

    brier_score = sum(
        (probability - actual) ** 2
        for probability, actual in zip(probabilities, y_true)
    ) / len(y_true)

    epsilon = 1e-12
    log_loss = -sum(
        actual * math.log(max(epsilon, probability))
        + (1 - actual) * math.log(
            max(epsilon, 1.0 - probability)
        )
        for probability, actual in zip(probabilities, y_true)
    ) / len(y_true)

    return {
        "accuracy": accuracy,
        "brier_score": brier_score,
        "log_loss": log_loss,
    }


def train(
    db_file: str | Path,
    *,
    model_path: str = DEFAULT_MODEL_PATH,
    metadata_path: str = DEFAULT_METADATA_PATH,
    min_samples: int = MIN_TRAINING_SAMPLES,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> bool:
    """
    Trains a football-only probability model.

    Uses LogisticRegression with StandardScaler when scikit-learn is
    installed. If sklearn is unavailable, training is skipped safely.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print(
            "Football meta training skipped: "
            "scikit-learn is not installed."
        )
        return False

    ensure_feature_history_table(db_file)
    features, labels = load_training_data(db_file)

    sample_count = len(labels)

    if sample_count < max(20, int(min_samples)):
        print(
            "Football meta training skipped: "
            f"{sample_count} settled football samples; "
            f"minimum is {min_samples}."
        )
        return False

    positive_count = sum(labels)
    negative_count = sample_count - positive_count

    if positive_count < 10 or negative_count < 10:
        print(
            "Football meta training skipped: "
            f"insufficient class balance "
            f"(wins={positive_count}, losses={negative_count})."
        )
        return False

    test_fraction = max(0.10, min(0.40, float(test_fraction)))

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_fraction,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    C=0.70,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)

    probabilities = [
        safe_float(row[1], 0.5)
        for row in model.predict_proba(x_test)
    ]

    metrics = _classification_metrics(
        y_test,
        probabilities,
    )

    # Conservative validation score:
    # accuracy and Brier quality combined into a 0-1 value.
    brier_quality = max(
        0.0,
        min(1.0, 1.0 - metrics["brier_score"] / 0.25),
    )
    validation_score = max(
        0.0,
        min(
            1.0,
            metrics["accuracy"] * 0.55
            + brier_quality * 0.45,
        ),
    )

    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)

    with model_file.open("wb") as handle:
        pickle.dump(model, handle)

    metadata = FootballMetaMetadata(
        model_type="sklearn_logistic_regression_pipeline",
        trained_at=now_utc(),
        samples=sample_count,
        feature_count=len(FEATURE_ORDER),
        positive_rate=positive_count / sample_count,
        validation_score=validation_score,
        version="v13",
        feature_order=list(FEATURE_ORDER),
    )

    save_football_meta_metadata(
        metadata,
        path=metadata_path,
    )

    metrics_path = Path(
        str(metadata_path).replace(".json", "_metrics.json")
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "trained_at": metadata.trained_at,
                "samples": sample_count,
                "train_samples": len(y_train),
                "test_samples": len(y_test),
                "wins": positive_count,
                "losses": negative_count,
                "positive_rate": metadata.positive_rate,
                "accuracy": metrics["accuracy"],
                "brier_score": metrics["brier_score"],
                "log_loss": metrics["log_loss"],
                "validation_score": validation_score,
                "feature_count": len(FEATURE_ORDER),
                "model_path": str(model_file),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Football meta model trained successfully: "
        f"samples={sample_count}, "
        f"accuracy={metrics['accuracy']:.3f}, "
        f"brier={metrics['brier_score']:.3f}, "
        f"validation={validation_score:.3f}"
    )

    return True


if __name__ == "__main__":
    from core.config import Settings

    settings = Settings.from_env()
    train(settings.db_file)

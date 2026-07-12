from __future__ import annotations

import json
import math
import pickle
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import Settings
from core.football_meta import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
)
from core.football_trainer import (
    FEATURE_ORDER,
    ensure_feature_history_table,
    load_training_data,
)


DEFAULT_REPORT_PATH = "exports/football_meta_v14_report.json"
DEFAULT_IMPORTANCE_PATH = "exports/football_meta_v14_feature_importance.json"
DEFAULT_MILESTONES = (20, 50, 80, 120, 200, 350, 500)


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


def classification_metrics(
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


@dataclass
class FootballMetaV14Result:
    trained: bool
    samples: int
    wins: int
    losses: int
    milestone: int
    model_type: str
    validation_score: float
    skipped_reason: str
    report_path: str
    importance_path: str


class FootballMetaAIV14:
    def __init__(
        self,
        settings: Settings,
        *,
        model_path: str = DEFAULT_MODEL_PATH,
        metadata_path: str = DEFAULT_METADATA_PATH,
        report_path: str = DEFAULT_REPORT_PATH,
        importance_path: str = DEFAULT_IMPORTANCE_PATH,
        milestones: tuple[int, ...] = DEFAULT_MILESTONES,
    ) -> None:
        self.settings = settings
        self.db_file = Path(settings.db_file or "bets.db")
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.report_path = Path(report_path)
        self.importance_path = Path(importance_path)
        self.milestones = tuple(sorted(set(int(x) for x in milestones if int(x) >= 20)))

        ensure_feature_history_table(self.db_file)

    def _trained_samples(self) -> int:
        if not self.metadata_path.exists():
            return 0

        try:
            payload = json.loads(
                self.metadata_path.read_text(encoding="utf-8")
            )
            return int(payload.get("samples", 0) or 0)
        except Exception:
            return 0

    def _next_milestone(self, samples: int) -> int:
        for milestone in self.milestones:
            if samples <= milestone:
                return milestone

        step = 250
        return ((samples // step) + 1) * step

    def _should_train(
        self,
        samples: int,
        trained_samples: int,
    ) -> tuple[bool, int, str]:
        milestone = self._next_milestone(samples)

        if samples < self.milestones[0]:
            return (
                False,
                self.milestones[0],
                f"not enough settled samples: {samples}/{self.milestones[0]}",
            )

        if trained_samples == 0:
            return True, milestone, ""

        crossed = [
            value
            for value in self.milestones
            if trained_samples < value <= samples
        ]

        if crossed:
            return True, max(crossed), ""

        if samples >= self.milestones[-1]:
            if samples - trained_samples >= 250:
                return True, milestone, ""

        return (
            False,
            milestone,
            "no new training milestone reached",
        )

    def _build_candidates(self) -> list[tuple[str, Any]]:
        try:
            from sklearn.ensemble import (
                HistGradientBoostingClassifier,
                RandomForestClassifier,
            )
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise RuntimeError(
                "scikit-learn is not installed"
            ) from exc

        candidates: list[tuple[str, Any]] = [
            (
                "logistic_regression",
                Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        (
                            "classifier",
                            LogisticRegression(
                                max_iter=2500,
                                class_weight="balanced",
                                random_state=42,
                                C=0.70,
                            ),
                        ),
                    ]
                ),
            ),
        ]

        candidates.append(
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=350,
                    max_depth=7,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            )
        )

        candidates.append(
            (
                "hist_gradient_boosting",
                HistGradientBoostingClassifier(
                    learning_rate=0.04,
                    max_iter=250,
                    max_leaf_nodes=15,
                    min_samples_leaf=10,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            )
        )

        return candidates

    def _feature_importance(
        self,
        model_name: str,
        model: Any,
    ) -> list[dict[str, float | str]]:
        values: list[float] | None = None

        if model_name == "logistic_regression":
            classifier = model.named_steps["classifier"]
            values = [
                abs(float(value))
                for value in classifier.coef_[0]
            ]
        elif hasattr(model, "feature_importances_"):
            values = [
                float(value)
                for value in model.feature_importances_
            ]

        if values is None:
            return []

        total = sum(values)

        if total <= 0:
            total = 1.0

        rows = [
            {
                "feature": feature,
                "importance": value / total,
            }
            for feature, value in zip(FEATURE_ORDER, values)
        ]

        rows.sort(
            key=lambda row: float(row["importance"]),
            reverse=True,
        )

        return rows

    def train(self) -> FootballMetaV14Result:
        try:
            from sklearn.model_selection import train_test_split
        except ImportError:
            return FootballMetaV14Result(
                trained=False,
                samples=0,
                wins=0,
                losses=0,
                milestone=self.milestones[0],
                model_type="",
                validation_score=0.0,
                skipped_reason="scikit-learn is not installed",
                report_path=str(self.report_path),
                importance_path=str(self.importance_path),
            )

        features, labels = load_training_data(self.db_file)

        samples = len(labels)
        wins = sum(labels)
        losses = samples - wins
        trained_samples = self._trained_samples()

        should_train, milestone, skipped_reason = self._should_train(
            samples,
            trained_samples,
        )

        if not should_train:
            return FootballMetaV14Result(
                trained=False,
                samples=samples,
                wins=wins,
                losses=losses,
                milestone=milestone,
                model_type="",
                validation_score=0.0,
                skipped_reason=skipped_reason,
                report_path=str(self.report_path),
                importance_path=str(self.importance_path),
            )

        if wins < 5 or losses < 5:
            return FootballMetaV14Result(
                trained=False,
                samples=samples,
                wins=wins,
                losses=losses,
                milestone=milestone,
                model_type="",
                validation_score=0.0,
                skipped_reason=(
                    f"insufficient class balance: wins={wins}, losses={losses}"
                ),
                report_path=str(self.report_path),
                importance_path=str(self.importance_path),
            )

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=0.25 if samples < 80 else 0.20,
            random_state=42,
            stratify=labels,
        )

        candidate_reports: list[dict[str, Any]] = []
        best_name = ""
        best_model: Any = None
        best_score = -1.0
        best_metrics: dict[str, float] = {}

        for model_name, model in self._build_candidates():
            model.fit(x_train, y_train)

            probabilities = [
                safe_float(row[1], 0.5)
                for row in model.predict_proba(x_test)
            ]

            metrics = classification_metrics(
                y_test,
                probabilities,
            )

            brier_quality = max(
                0.0,
                min(
                    1.0,
                    1.0 - metrics["brier_score"] / 0.25,
                ),
            )
            log_quality = max(
                0.0,
                min(
                    1.0,
                    1.0 - metrics["log_loss"] / 0.693147,
                ),
            )

            validation_score = max(
                0.0,
                min(
                    1.0,
                    metrics["accuracy"] * 0.45
                    + brier_quality * 0.40
                    + log_quality * 0.15,
                ),
            )

            candidate_reports.append(
                {
                    "model_type": model_name,
                    **metrics,
                    "validation_score": validation_score,
                }
            )

            if validation_score > best_score:
                best_score = validation_score
                best_name = model_name
                best_model = model
                best_metrics = metrics

        if best_model is None:
            return FootballMetaV14Result(
                trained=False,
                samples=samples,
                wins=wins,
                losses=losses,
                milestone=milestone,
                model_type="",
                validation_score=0.0,
                skipped_reason="no model candidate trained successfully",
                report_path=str(self.report_path),
                importance_path=str(self.importance_path),
            )

        # Refit winner on all settled samples.
        best_model.fit(features, labels)

        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        with self.model_path.open("wb") as handle:
            pickle.dump(best_model, handle)

        metadata = {
            "model_type": f"football_meta_v14_{best_name}",
            "trained_at": now_utc(),
            "samples": samples,
            "feature_count": len(FEATURE_ORDER),
            "positive_rate": wins / samples,
            "validation_score": best_score,
            "version": "v14",
            "feature_order": list(FEATURE_ORDER),
            "milestone": milestone,
        }

        self.metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.metadata_path.write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        importance = self._feature_importance(
            best_name,
            best_model,
        )

        self.importance_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.importance_path.write_text(
            json.dumps(
                {
                    "trained_at": metadata["trained_at"],
                    "model_type": best_name,
                    "samples": samples,
                    "features": importance,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        report = {
            "trained_at": metadata["trained_at"],
            "samples": samples,
            "wins": wins,
            "losses": losses,
            "positive_rate": metadata["positive_rate"],
            "milestone": milestone,
            "selected_model": best_name,
            "selected_validation_score": best_score,
            "selected_metrics": best_metrics,
            "candidate_models": candidate_reports,
            "next_milestone": self._next_milestone(samples + 1),
            "model_path": str(self.model_path),
            "metadata_path": str(self.metadata_path),
            "importance_path": str(self.importance_path),
        }

        self.report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.report_path.write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return FootballMetaV14Result(
            trained=True,
            samples=samples,
            wins=wins,
            losses=losses,
            milestone=milestone,
            model_type=best_name,
            validation_score=best_score,
            skipped_reason="",
            report_path=str(self.report_path),
            importance_path=str(self.importance_path),
        )


def run_football_meta_ai_v14(
    settings: Settings,
) -> FootballMetaV14Result:
    return FootballMetaAIV14(settings).train()


if __name__ == "__main__":
    settings = Settings.from_env()
    result = run_football_meta_ai_v14(settings)

    print(
        "Football Meta AI v14: "
        f"trained={result.trained}, "
        f"samples={result.samples}, "
        f"wins={result.wins}, "
        f"losses={result.losses}, "
        f"milestone={result.milestone}, "
        f"model={result.model_type or 'none'}, "
        f"validation={result.validation_score:.3f}"
    )

    if result.skipped_reason:
        print(result.skipped_reason)

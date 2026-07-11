from __future__ import annotations

import json
import math
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.football_features import FootballFeatures


DEFAULT_MODEL_PATH = "models/football_meta_model.pkl"
DEFAULT_METADATA_PATH = "models/football_meta_model.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if math.isnan(result) or math.isinf(result):
        return default

    return result


@dataclass
class FootballMetaPrediction:
    probability: float
    fallback_probability: float
    source: str
    model_loaded: bool
    model_path: str
    confidence: float
    reliability: float
    reason: str


@dataclass
class FootballMetaMetadata:
    model_type: str = "unknown"
    trained_at: str = ""
    samples: int = 0
    feature_count: int = 0
    positive_rate: float = 0.0
    validation_score: float = 0.0
    version: str = "v13"
    feature_order: list[str] | None = None


class FootballMetaModel:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        metadata_path: str = DEFAULT_METADATA_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.model: Any | None = None
        self.metadata = FootballMetaMetadata()
        self.load_error = ""

        self._load_metadata()
        self._load_model()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def _load_metadata(self) -> None:
        if not self.metadata_path.exists():
            return

        try:
            raw = json.loads(
                self.metadata_path.read_text(encoding="utf-8")
            )

            if isinstance(raw, dict):
                self.metadata = FootballMetaMetadata(
                    model_type=str(raw.get("model_type", "unknown")),
                    trained_at=str(raw.get("trained_at", "")),
                    samples=int(raw.get("samples", 0) or 0),
                    feature_count=int(raw.get("feature_count", 0) or 0),
                    positive_rate=safe_float(
                        raw.get("positive_rate"),
                        0.0,
                    ),
                    validation_score=safe_float(
                        raw.get("validation_score"),
                        0.0,
                    ),
                    version=str(raw.get("version", "v13")),
                    feature_order=raw.get("feature_order"),
                )
        except Exception as exc:
            self.load_error = (
                f"metadata load failed: {type(exc).__name__}: {exc}"
            )

    def _load_model(self) -> None:
        if not self.model_path.exists():
            self.load_error = (
                self.load_error
                or f"model not found: {self.model_path}"
            )
            return

        try:
            with self.model_path.open("rb") as handle:
                self.model = pickle.load(handle)
        except Exception as exc:
            self.model = None
            self.load_error = (
                f"model load failed: {type(exc).__name__}: {exc}"
            )

    @staticmethod
    def fallback_probability(features: FootballFeatures) -> float:
        """
        Conservative fallback built from the v13 consensus.

        It shrinks the internal model towards the market when:
        - reliability is low,
        - dispersion is high,
        - market overround is high.
        """
        base = clamp(
            safe_float(
                features.model_consensus_probability,
                features.market_selection_probability,
            ),
            0.01,
            0.99,
        )

        market = clamp(
            safe_float(features.market_selection_probability, base),
            0.01,
            0.99,
        )

        reliability = clamp(
            safe_float(features.reliability_input, 0.0),
            0.0,
            1.0,
        )
        confidence = clamp(
            safe_float(features.confidence_input, 0.0),
            0.0,
            1.0,
        )
        dispersion = clamp(
            safe_float(features.model_dispersion, 0.0),
            0.0,
            0.50,
        )
        overround = clamp(
            safe_float(features.market_overround, 0.0),
            0.0,
            0.50,
        )

        trust = (
            reliability * 0.45
            + confidence * 0.35
            + (1.0 - dispersion * 2.0) * 0.15
            + (1.0 - overround) * 0.05
        )
        trust = clamp(trust, 0.20, 0.90)

        probability = base * trust + market * (1.0 - trust)

        # Avoid extreme football probabilities without strong evidence.
        return clamp(probability, 0.03, 0.92)

    def _predict_with_model(
        self,
        features: FootballFeatures,
    ) -> float:
        if self.model is None:
            raise RuntimeError("Football meta model is not loaded")

        vector = features.numeric_vector()
        matrix = [vector]

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(matrix)

            if len(probabilities) == 0:
                raise RuntimeError("predict_proba returned no rows")

            row = probabilities[0]

            if len(row) >= 2:
                return safe_float(row[1], 0.0)

            if len(row) == 1:
                return safe_float(row[0], 0.0)

            raise RuntimeError("predict_proba returned an empty row")

        if hasattr(self.model, "predict"):
            prediction = self.model.predict(matrix)

            if len(prediction) == 0:
                raise RuntimeError("predict returned no rows")

            return safe_float(prediction[0], 0.0)

        if callable(self.model):
            return safe_float(self.model(vector), 0.0)

        raise TypeError(
            "Loaded football meta model has no predict interface"
        )

    def predict(
        self,
        features: FootballFeatures,
    ) -> FootballMetaPrediction:
        fallback = self.fallback_probability(features)

        confidence = clamp(
            safe_float(features.confidence_input, 0.0),
            0.0,
            1.0,
        )
        reliability = clamp(
            safe_float(features.reliability_input, 0.0),
            0.0,
            1.0,
        )

        if not self.is_loaded:
            reason = (
                "football meta fallback; "
                f"{self.load_error or 'model unavailable'}; "
                f"consensus={features.model_consensus_probability:.4f}; "
                f"market={features.market_selection_probability:.4f}; "
                f"fallback={fallback:.4f}"
            )

            return FootballMetaPrediction(
                probability=fallback,
                fallback_probability=fallback,
                source="FOOTBALL_V13_FALLBACK",
                model_loaded=False,
                model_path=str(self.model_path),
                confidence=confidence,
                reliability=reliability,
                reason=reason,
            )

        try:
            raw_probability = clamp(
                self._predict_with_model(features),
                0.01,
                0.99,
            )

            # Safety blend: do not let a young model fully override the
            # deterministic football consensus.
            samples = max(0, int(self.metadata.samples))
            sample_trust = clamp(samples / 500.0, 0.15, 0.85)

            quality_trust = clamp(
                self.metadata.validation_score,
                0.0,
                1.0,
            )

            model_trust = (
                sample_trust * 0.55
                + quality_trust * 0.25
                + reliability * 0.20
            )
            model_trust = clamp(model_trust, 0.15, 0.85)

            final_probability = (
                raw_probability * model_trust
                + fallback * (1.0 - model_trust)
            )
            final_probability = clamp(
                final_probability,
                0.03,
                0.92,
            )

            reason = (
                f"football meta model; raw={raw_probability:.4f}; "
                f"fallback={fallback:.4f}; "
                f"trust={model_trust:.3f}; "
                f"final={final_probability:.4f}; "
                f"samples={samples}; "
                f"validation={self.metadata.validation_score:.3f}"
            )

            return FootballMetaPrediction(
                probability=final_probability,
                fallback_probability=fallback,
                source="FOOTBALL_V13_META_MODEL",
                model_loaded=True,
                model_path=str(self.model_path),
                confidence=confidence,
                reliability=reliability,
                reason=reason,
            )

        except Exception as exc:
            reason = (
                "football meta prediction failed; "
                f"{type(exc).__name__}: {exc}; "
                f"fallback={fallback:.4f}"
            )

            return FootballMetaPrediction(
                probability=fallback,
                fallback_probability=fallback,
                source="FOOTBALL_V13_FALLBACK",
                model_loaded=True,
                model_path=str(self.model_path),
                confidence=confidence,
                reliability=reliability,
                reason=reason,
            )


def predict_football_probability(
    features: FootballFeatures,
    *,
    model_path: str = DEFAULT_MODEL_PATH,
    metadata_path: str = DEFAULT_METADATA_PATH,
) -> FootballMetaPrediction:
    model = FootballMetaModel(
        model_path=model_path,
        metadata_path=metadata_path,
    )

    return model.predict(features)


def save_football_meta_metadata(
    metadata: FootballMetaMetadata,
    path: str = DEFAULT_METADATA_PATH,
) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if not metadata.trained_at:
        metadata.trained_at = now_utc()

    file_path.write_text(
        json.dumps(
            asdict(metadata),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

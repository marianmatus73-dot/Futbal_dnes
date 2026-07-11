from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BayesianResult:
    posterior_mean: float
    confidence: float
    samples: int


def beta_binomial_update(
    wins: int,
    losses: int,
    prior_alpha: float = 10.0,
    prior_beta: float = 10.0,
) -> BayesianResult:
    wins = max(0, int(wins))
    losses = max(0, int(losses))

    alpha = prior_alpha + wins
    beta = prior_beta + losses
    total = alpha + beta

    posterior_mean = alpha / total if total > 0 else 0.5
    samples = wins + losses

    confidence = min(1.0, samples / 100.0)

    return BayesianResult(
        posterior_mean=posterior_mean,
        confidence=confidence,
        samples=samples,
    )


def bayesian_multiplier(
    wins: int,
    losses: int,
    min_weight: float = 0.90,
    max_weight: float = 1.10,
    prior_alpha: float = 10.0,
    prior_beta: float = 10.0,
) -> float:
    result = beta_binomial_update(
        wins=wins,
        losses=losses,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
    )

    deviation = result.posterior_mean - 0.5
    raw_multiplier = 1.0 + deviation * 2.0

    blended = (
        1.0 * (1.0 - result.confidence)
        + raw_multiplier * result.confidence
    )

    return round(
        max(min_weight, min(max_weight, blended)),
        4,
    )

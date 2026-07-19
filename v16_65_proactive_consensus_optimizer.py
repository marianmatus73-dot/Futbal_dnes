"""
V16.65 PROACTIVE CONSENSUS OPTIMIZER

Optimizes agent combination before final decision.
"""


def optimize_consensus(predicted_scores):
    total = sum(predicted_scores.values())

    weights = {
        agent: round(score / total, 3)
        for agent, score in predicted_scores.items()
    }

    ranking = sorted(
        predicted_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "predicted_scores": predicted_scores,
        "optimized_weights": weights,
        "agent_ranking": ranking,
        "consensus_optimized": True,
        "status": "READY"
    }
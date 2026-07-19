"""
V16.63 ADAPTIVE AGENT WEIGHTING ENGINE

Adjusts agent influence based on performance.
"""


def update_agent_weights(agent_scores):
    total = sum(agent_scores.values())

    weights = {
        agent: round(score / total, 2)
        for agent, score in agent_scores.items()
    } if total else {}

    return {
        "agent_scores": agent_scores,
        "agent_weights": weights,
        "weights_updated": True,
        "status": "READY"
    }
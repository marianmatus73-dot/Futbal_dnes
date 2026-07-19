"""
V16.81 NEXT GENERATION INTELLIGENCE REASONING ENGINE

Performs reasoning over knowledge and context.
"""


def reason_over_knowledge(nodes, context_score):
    reasoning_score = round(
        (nodes / 10) * 0.5 + context_score * 0.5,
        3
    )

    decision_support = "ENABLED" if reasoning_score >= 0.5 else "LIMITED"

    return {
        "knowledge_nodes": nodes,
        "context_score": context_score,
        "reasoning_score": reasoning_score,
        "decision_support": decision_support,
        "reasoning_active": True,
        "status": "READY"
    }
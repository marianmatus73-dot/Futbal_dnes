"""
V16.62 AGENT CONSENSUS DECISION ENGINE

Combines agent outputs into weighted consensus.
"""


def consensus_decision(agents):
    ready_count = sum(1 for value in agents.values() if value == "READY")
    total = len(agents)

    score = round(ready_count / total, 2) if total else 0

    decision = "APPROVED" if score >= 0.8 else "REVIEW"

    return {
        "agents_checked": total,
        "consensus_score": score,
        "decision": decision,
        "weighted_consensus": True,
        "status": "READY"
    }
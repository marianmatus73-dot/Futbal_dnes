
"""
V16.37 AUTONOMOUS LEARNING LOOP

Updates learning state from results and performance.
"""


def update_learning(result, previous_score):
    if result == "WIN":
        adjustment = 0.05
    elif result == "LOSS":
        adjustment = -0.05
    else:
        adjustment = 0

    new_score = max(0, min(1, previous_score + adjustment))

    return {
        "previous_score": previous_score,
        "new_score": round(new_score, 2),
        "learning_updated": True,
        "status": "READY"
    }

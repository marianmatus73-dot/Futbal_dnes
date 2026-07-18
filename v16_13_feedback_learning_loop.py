
"""
V16.13 FEEDBACK LEARNING LOOP

Compares decisions with results and prepares feedback.
"""


def evaluate_prediction(decision, result):
    if decision == "PLAY" and result == "WIN":
        score = 1
    elif decision == "PLAY" and result == "LOSS":
        score = 0
    else:
        score = None

    return {
        "decision": decision,
        "result": result,
        "feedback_score": score
    }


def update_learning(feedback):
    return {
        "feedback_received": True,
        "learning_update": "READY"
    }

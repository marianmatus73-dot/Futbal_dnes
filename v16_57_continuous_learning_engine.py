"""
V16.57 CONTINUOUS LEARNING ENGINE

Updates model state from feedback signals.
"""


def continuous_learning(feedback, model_weight):
    if feedback == "POSITIVE":
        adjustment = 0.05
    elif feedback == "NEGATIVE":
        adjustment = -0.05
    else:
        adjustment = 0

    new_weight = round(max(0.0, min(model_weight + adjustment, 2.0)), 2)

    return {
        "feedback": feedback,
        "old_weight": model_weight,
        "new_weight": new_weight,
        "learning_updated": True,
        "status": "READY"
    }
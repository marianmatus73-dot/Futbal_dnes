
"""
V16.14 MODEL CALIBRATION

Adjusts confidence based on feedback history.
"""


def calibrate(feedback_score, confidence):
    if feedback_score == 1:
        adjusted = min(confidence + 0.05, 1.0)
    elif feedback_score == 0:
        adjusted = max(confidence - 0.05, 0.0)
    else:
        adjusted = confidence

    return {
        "old_confidence": confidence,
        "new_confidence": adjusted,
        "status": "READY"
    }

def feedback_fusion(validation_score, execution_result, profit):
    feedback_score = 1.0 if execution_result == "WIN" else 0.0

    learning_signal = round(
        (validation_score / 100) * 0.5 +
        feedback_score * 0.3 +
        (1 if profit > 0 else 0) * 0.2,
        3
    )

    status = "LEARNING_UPDATE_READY" if learning_signal >= 0.8 else "REVIEW"

    return {
        "validation_score": validation_score,
        "execution_result": execution_result,
        "profit": profit,
        "feedback_score": feedback_score,
        "learning_signal": learning_signal,
        "model_update_ready": status == "LEARNING_UPDATE_READY",
        "feedback_fusion_active": True,
        "status": status
    }
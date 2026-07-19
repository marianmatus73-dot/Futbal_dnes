"""
V16.03 MASTER LEARNING UPDATE ENGINE

Updates model and strategy weights from feedback signal.
"""


def learning_update(learning_signal, old_model_weight, old_strategy_weight):
    model_increment = round(learning_signal * 0.05, 3)
    strategy_increment = round(learning_signal * 0.04, 3)

    new_model_weight = round(old_model_weight + model_increment, 3)
    new_strategy_weight = round(old_strategy_weight + strategy_increment, 3)

    return {
        "learning_signal": learning_signal,
        "old_model_weight": old_model_weight,
        "new_model_weight": new_model_weight,
        "old_strategy_weight": old_strategy_weight,
        "new_strategy_weight": new_strategy_weight,
        "weights_updated": True,
        "learning_update_active": True,
        "status": "READY"
    }
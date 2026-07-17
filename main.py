from core.football_learning_weight_optimizer_v15_55 import (
    run_learning_weight_optimizer_v15_55,
)


def run_full_pipeline_v15_55():
    return {
        "learning_optimizer": run_learning_weight_optimizer_v15_55(),
    }


if __name__ == "__main__":
    print(run_full_pipeline_v15_55())

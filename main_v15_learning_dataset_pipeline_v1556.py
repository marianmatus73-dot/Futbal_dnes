from core.football_learning_dataset_builder_v15_56 import (
    run_learning_dataset_builder_v15_56,
)


def run_full_pipeline_v15_56():
    return {
        "learning_dataset": run_learning_dataset_builder_v15_56(),
    }


if __name__ == "__main__":
    print(run_full_pipeline_v15_56())

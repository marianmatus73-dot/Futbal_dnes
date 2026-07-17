from core.football_learning_analyzer_v15_50 import (
    run_learning_analyzer_v15_50,
)


def run_full_pipeline_v15_50():

    learning = run_learning_analyzer_v15_50()

    return {
        "learning": learning,
    }


if __name__ == "__main__":
    print(run_full_pipeline_v15_50())

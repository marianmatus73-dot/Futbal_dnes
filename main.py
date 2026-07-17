from core.football_feature_source_joiner_v15_58 import (
    run_feature_source_joiner_v15_58,
)


def run_full_pipeline_v15_58():
    return {
        "feature_join": run_feature_source_joiner_v15_58(),
    }


if __name__ == "__main__":
    print(run_full_pipeline_v15_58())

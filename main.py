from core.football_feature_enrichment_v15_57 import (
    run_feature_enrichment_v15_57,
)


def run_full_pipeline_v15_57():
    report, _ = run_feature_enrichment_v15_57()
    return {
        "feature_enrichment": report,
    }


if __name__ == "__main__":
    print(run_full_pipeline_v15_57())

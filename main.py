from core.football_smart_market_probability_resolver_v15_60 import (
    run_smart_market_probability_resolver_v15_60,
)


def run_full_pipeline_v15_60():
    return {
        "smart_market_probability": run_smart_market_probability_resolver_v15_60(),
    }


if __name__ == "__main__":
    print(run_full_pipeline_v15_60())

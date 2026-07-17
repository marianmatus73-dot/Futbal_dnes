from core.football_correct_snapshot_source_loader_v15_39 import run_correct_snapshot_source_loader_v15_39
from core.football_event_split_adapter_v15_32 import run_event_split_adapter_v15_32
from core.football_source_hash_resolver_v15_37 import run_source_hash_resolver_v15_37
from core.football_opening_odds_bridge_v15_43 import run_opening_odds_bridge_v15_43
from core.football_closing_odds_bridge_v15_46 import run_closing_odds_bridge_v15_46
from core.football_opening_closing_clv_engine_v15_44 import run_clv_engine_v15_44


def run_full_pipeline_v15_46():

    _, snapshots = run_correct_snapshot_source_loader_v15_39()
    _, postmatch = run_event_split_adapter_v15_32()

    _, resolved = run_source_hash_resolver_v15_37(
        snapshots,
        postmatch,
    )

    opening_report, opening_rows = run_opening_odds_bridge_v15_43(resolved)

    closing_report, merged_rows = run_closing_odds_bridge_v15_46(
        opening_rows,
        resolved,
    )

    clv_report, _ = run_clv_engine_v15_44(merged_rows)

    return {
        "opening": opening_report,
        "closing_bridge": closing_report,
        "clv": clv_report,
    }


if __name__ == "__main__":
    print(run_full_pipeline_v15_46())

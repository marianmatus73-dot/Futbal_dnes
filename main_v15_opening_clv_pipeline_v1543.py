from core.football_correct_snapshot_source_loader_v15_39 import run_correct_snapshot_source_loader_v15_39
from core.football_event_split_adapter_v15_32 import run_event_split_adapter_v15_32
from core.football_source_hash_resolver_v15_37 import run_source_hash_resolver_v15_37
from core.football_opening_odds_bridge_v15_43 import run_opening_odds_bridge_v15_43


def run_full_pipeline_v15_43():

    _, snapshots = run_correct_snapshot_source_loader_v15_39()
    _, postmatch = run_event_split_adapter_v15_32()

    resolver_report, resolved = run_source_hash_resolver_v15_37(
        snapshots,
        postmatch,
    )

    opening_report, _ = run_opening_odds_bridge_v15_43(
        resolved
    )

    return {
        "resolver": resolver_report,
        "opening_bridge": opening_report,
    }


if __name__ == "__main__":
    print(run_full_pipeline_v15_43())

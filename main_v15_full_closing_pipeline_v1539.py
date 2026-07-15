from core.football_correct_snapshot_source_loader_v15_39 import (
    run_correct_snapshot_source_loader_v15_39
)

from core.football_event_split_adapter_v15_32 import (
    run_event_split_adapter_v15_32
)

from core.football_source_hash_resolver_v15_37 import (
    run_source_hash_resolver_v15_37
)


def run_full_pipeline_v15_39():

    source_report, snapshot_rows = (
        run_correct_snapshot_source_loader_v15_39()
    )

    event_report, postmatch_rows = (
        run_event_split_adapter_v15_32()
    )

    resolver_report, resolved = (
        run_source_hash_resolver_v15_37(
            snapshot_rows,
            postmatch_rows,
        )
    )

    return {
        "source_loader": source_report,
        "event_split": event_report,
        "resolver": resolver_report,
        "resolved_matches": len(resolved),
    }


if __name__ == "__main__":
    print(run_full_pipeline_v15_39())

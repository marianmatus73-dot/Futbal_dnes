from core.football_correct_snapshot_source_loader_v15_39 import (
    run_correct_snapshot_source_loader_v15_39
)

from core.football_event_split_adapter_v15_32 import (
    run_event_split_adapter_v15_32
)

from core.football_source_hash_resolver_v15_37 import (
    run_source_hash_resolver_v15_37
)

from core.football_closing_writer_integration_v15_41 import (
    run_closing_writer_integration_v15_41
)


def run_full_pipeline_v15_41():

    source_report, snapshots = run_correct_snapshot_source_loader_v15_39()

    event_report, postmatch_rows = run_event_split_adapter_v15_32()

    resolver_report, resolved = run_source_hash_resolver_v15_37(
        snapshots,
        postmatch_rows,
    )

    closing_report = run_closing_writer_integration_v15_41(
        resolved
    )

    return {
        "source_loader": source_report,
        "event_split": event_report,
        "resolver": resolver_report,
        "closing": closing_report,
    }


if __name__ == "__main__":
    print(run_full_pipeline_v15_41())

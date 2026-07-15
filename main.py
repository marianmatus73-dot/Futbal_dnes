from core.football_csv_snapshot_loader_v15_26 import (
    run_csv_snapshot_loader_v15_26
)

from core.football_snapshot_row_parser_adapter_v15_23 import (
    run_snapshot_row_parser_adapter_v15_23
)

from core.football_event_split_adapter_v15_32 import (
    run_event_split_adapter_v15_32
)

from core.football_smart_match_resolver_v15_29 import (
    resolve_smart_matches
)


def run_full_pipeline_v15_34():

    loader_report, snapshots = run_csv_snapshot_loader_v15_26()

    parser_report, parsed_rows = run_snapshot_row_parser_adapter_v15_23(
        snapshots=snapshots
    )

    event_report, postmatch_rows = run_event_split_adapter_v15_32()

    join_ready_rows = [
        row for row in parsed_rows
        if row.get("join_ready")
    ]

    resolved = resolve_smart_matches(
        join_ready_rows=join_ready_rows,
        source_rows=postmatch_rows
    )

    resolver_report = {
        "version": "v15.34",
        "input_snapshots": len(join_ready_rows),
        "postmatch_rows": len(postmatch_rows),
        "matched_rows": len(resolved),
        "status": "READY" if resolved else "BUILDING",
    }

    return {
        "loader": loader_report,
        "parser": parser_report,
        "event_split": event_report,
        "resolver": resolver_report,
    }


if __name__ == "__main__":
    print(run_full_pipeline_v15_34())

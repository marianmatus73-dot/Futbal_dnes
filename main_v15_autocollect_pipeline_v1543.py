from core.football_correct_snapshot_source_loader_v15_39 import (
    run_correct_snapshot_source_loader_v15_39,
)
from core.football_event_split_adapter_v15_32 import (
    run_event_split_adapter_v15_32,
)
from core.football_source_hash_resolver_v15_37 import (
    run_source_hash_resolver_v15_37,
)
from core.football_closing_writer_integration_v15_41 import (
    run_closing_writer_integration_v15_41,
)
from core.football_clv_calculator_v15_42 import (
    run_clv_calculator_v15_42,
)


def run_autocollect_pipeline_v15_43():

    source_report, snapshots = (
        run_correct_snapshot_source_loader_v15_39()
    )

    event_report, postmatch = (
        run_event_split_adapter_v15_32()
    )

    resolver_report, resolved = (
        run_source_hash_resolver_v15_37(
            snapshots,
            postmatch,
        )
    )

    closing_report = (
        run_closing_writer_integration_v15_41(
            resolved
        )
    )

    clv_report = (
        run_clv_calculator_v15_42()
    )

    return {
        "source_loader": source_report,
        "event_split": event_report,
        "resolver": resolver_report,
        "closing": closing_report,
        "clv": clv_report,
        "pipeline_status": "RUNNING_COLLECTION",
    }


if __name__ == "__main__":
    print(run_autocollect_pipeline_v15_43())

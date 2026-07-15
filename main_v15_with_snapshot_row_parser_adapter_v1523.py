# Complete V15.23 integration main

from core.football_snapshot_row_parser_adapter_v15_23 import (
    run_snapshot_row_parser_adapter_v15_23
)

# Pipeline:
#
# V15.20 Snapshot Schema Extractor
#          ↓
# V15.23 Snapshot Row Parser Adapter
#          ↓
# V15.21 Closing Odds Writer
#          ↓
# V15.19 Database Join Engine
#

def run_v15_23_pipeline():
    report, rows = run_snapshot_row_parser_adapter_v15_23([])
    return report, rows


if __name__ == "__main__":
    result = run_v15_23_pipeline()
    print(result[0])

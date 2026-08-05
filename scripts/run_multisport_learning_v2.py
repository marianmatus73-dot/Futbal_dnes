from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from pprint import pprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db_bootstrap import bootstrap_database
from core.history_restore import restore_history
from core.migration_engine import migrate_database
from core.multisport_learning_v2.manager import MultisportLearningV2Manager


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database",
        default=os.getenv("DB_FILE", "bets.db"),
    )
    parser.add_argument(
        "--history-csv",
        default=os.getenv(
            "MULTISPORT_V2_HISTORY_CSV",
            "exports/history_sport_bets.csv",
        ),
    )
    parser.add_argument(
        "--export-dir",
        default=os.getenv("EXPORT_DIR", "exports"),
    )
    args = parser.parse_args()

    print("=== MULTISPORT LEARNING V2.1 PRODUCTION ===")

    pprint(bootstrap_database(args.database))
    pprint(migrate_database(args.database))
    pprint(restore_history(args.database, args.history_csv))

    manager = MultisportLearningV2Manager(args.database)
    pprint(manager.run_all(export_dir=args.export_dir))


if __name__ == "__main__":
    main()

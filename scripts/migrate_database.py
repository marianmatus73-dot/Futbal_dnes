from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.migration_engine import migrate_database

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="bets.db")
    args = parser.parse_args()
    pprint(migrate_database(args.database))

if __name__ == "__main__":
    main()

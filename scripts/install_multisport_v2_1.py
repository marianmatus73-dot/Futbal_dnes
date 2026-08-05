from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_script(relative: str, *args: str) -> None:
    path = PROJECT_ROOT / relative
    if not path.is_file():
        raise SystemExit(f"FAILED: missing required file: {relative}")

    command = [sys.executable, str(path), *args]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    print("+", " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="bets.db")
    args = parser.parse_args()

    run_script(
        "scripts/migrate_database.py",
        "--database",
        args.database,
    )

    run_script(
        "scripts/validate_database.py",
        "--database",
        args.database,
    )

    run_script(
        "scripts/run_multisport_learning_v2.py",
        "--database",
        args.database,
    )

    print("Multisport Learning V2.1 installation: READY")


if __name__ == "__main__":
    main()

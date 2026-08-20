from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.tip_card_validation import TipCardValidationError, validate_tip_card


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a fresh production tip card")
    parser.add_argument("path", type=Path)
    parser.add_argument("--max-age-minutes", type=int, default=30)
    args = parser.parse_args()
    try:
        result = validate_tip_card(args.path, max_age_minutes=args.max_age_minutes)
    except TipCardValidationError as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

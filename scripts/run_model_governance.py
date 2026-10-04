from __future__ import annotations

import json
from pathlib import Path

from core.model_governance import build_model_governance


def _read(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    exports = Path("exports")
    payload = build_model_governance(
        _read(exports / "professional_model_table.json"),
        _read(exports / "operational_health.json"),
        exports,
    )
    print(f"Model coach: {len(payload['weekly_model_coach'])} segments; auto-apply OFF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

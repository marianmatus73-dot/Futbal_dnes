from pathlib import Path
import os

from v16_dashboard import build_dashboard

database = Path(os.getenv("DB_FILE", "bets.db"))
result = build_dashboard(
    database,
    export_dir=os.getenv("EXPORT_DIR", "exports"),
)
print("=== V16 PRODUCTION DASHBOARD REFRESH ===")
print(result)

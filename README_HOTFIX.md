# Multisport V2.1 missing-module hotfix

Copy the ZIP contents into the repository root.

Final paths must be:

```text
core/migration_engine.py
scripts/migrate_database.py
scripts/install_multisport_v2_1.py
```

Then run:

```bash
python scripts/install_multisport_v2_1.py --database bets.db
```

# Final V2.1 hotfix

Nahraď iba tieto dva súbory:

```text
scripts/validate_database.py
scripts/install_multisport_v2_1.py
```

Tento hotfix odstraňuje závislosť na:

```text
core.db_inspector
```

Potom spusti:

```bash
python scripts/install_multisport_v2_1.py --database bets.db
```

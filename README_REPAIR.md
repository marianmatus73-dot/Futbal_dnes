# Complete V2.1 repair

Rozbaľ balík priamo do koreňa repozitára a potvrď prepísanie súborov.

Balík obsahuje všetky súbory, ktoré V2.1 potrebuje:

```text
core/sqlite_helpers.py
core/migration_engine.py
core/multisport_learning_v2/__init__.py
core/multisport_learning_v2/schema.py
core/multisport_learning_v2/metrics.py
core/multisport_learning_v2/meta.py
core/multisport_learning_v2/dashboard.py
core/multisport_learning_v2/manager.py
scripts/run_multisport_learning_v2.py
```

Potom spusti iba:

```bash
python scripts/run_multisport_learning_v2.py --database bets.db
```

Installer už nie je potrebný na testovanie V2.1.

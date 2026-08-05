# Multisport Learning V2.1 Production Edition

Samostatný produkčný modul pre:

- baseball
- basketball
- tennis
- hockey
- MMA
- NFL

Modul:

- neimportuje `main.py`,
- nevyžaduje `python-dotenv`,
- dokáže vytvoriť minimálnu schému `sport_bets`,
- dokáže obnoviť históriu z CSV,
- kontroluje databázu,
- vykonáva bezpečné migrácie,
- vytvára JSON, CSV a HTML dashboard,
- je pripravený pre GitHub Actions.

## Inštalácia

Rozbaľ balík do koreňa projektu.

Výsledná štruktúra:

```text
core/
  multisport_learning_v2/
  db_bootstrap.py
  history_restore.py
  db_inspector.py
  migration_engine.py
  sqlite_helpers.py

scripts/
  run_multisport_learning_v2.py
  inspect_database.py
  migrate_database.py
  validate_database.py
  install_multisport_v2_1.py

.github/workflows/
  multisport_learning_v2_1.yml
```

## Produkčné spustenie

```bash
python scripts/run_multisport_learning_v2.py --database bets.db
```

## Kontrola databázy

```bash
python scripts/inspect_database.py --database bets.db
python scripts/validate_database.py --database bets.db
```

## Migrácia

```bash
python scripts/migrate_database.py --database bets.db
```

## Inštalácia a test

```bash
python scripts/install_multisport_v2_1.py --database bets.db
```

## CSV história

Predvolený zdroj:

```text
exports/history_sport_bets.csv
```

Ak súbor existuje, bootstrap ho importuje do `sport_bets`.

## Environment

```env
MULTISPORT_LEARNING_V2_ENABLED=1
MULTISPORT_V2_HISTORY_CSV=exports/history_sport_bets.csv
EXPORT_DIR=exports
DB_FILE=bets.db
```

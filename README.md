# Multisport Betting Engine

## Run

```bash
python main.py --sport football --dry-run
python main.py --sport tennis --dry-run
python main.py --sport all --dry-run
python main.py --sport basketball --dry-run
python main.py --sport hockey --dry-run
```

## Structure

```text
main.py
core/
  config.py
  market.py
  odds_api.py
  registry.py
  reporting.py
  staking.py
  types.py
sports/
  base.py
  football.py
  tennis.py
  basketball.py
  hockey.py
```

`main.py` only controls routing. Each sport has its own independent model.

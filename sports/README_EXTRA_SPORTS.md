# Extra Sports Quant Pack

Tento balík pridáva nové športové moduly v rovnakom štýle ako tenis/hokej/basketbal:

- `sports/baseball.py`
- `sports/mma.py`
- `sports/nfl.py`
- `sports/esports.py`

Každý modul používa spoločnú infraštruktúru:
- `sport_bets`
- `sport_odds_snapshots`
- `sport_decision_audit`
- `sport_bookmaker_stats`
- `sport_elo_ratings`
- CLV update
- settlement cez Odds API scores endpoint
- bookmaker grading
- ELO adjustment
- analytics report

## Dôležité

Tieto moduly sa spustia len vtedy, keď ich máš registrované v `main.py`.

Ak tvoj `main.py` má ručný zoznam modulov, treba doplniť importy napríklad:

```python
from sports.baseball import BaseballModule
from sports.mma import MMAModule
from sports.nfl import NFLModule
from sports.esports import EsportsModule
```

a do zoznamu modulov pridať:

```python
BaseballModule(),
MMAModule(),
NFLModule(),
EsportsModule(),
```

Ak tvoj `main.py` automaticky načítava všetky súbory zo `sports/`, netreba nič meniť.

## YAML

Obsah súboru `extra_sports_yaml_snippet.yml` vlož do `env:` sekcie v `.github/workflows/analyza.yml`.

Odporúčanie: nechaj `SPORT_KEY_AUTO_DISCOVERY: 1`, aby sa neaktívne ligy automaticky preskočili.

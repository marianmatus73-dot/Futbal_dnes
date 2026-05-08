# Football Betting Model v10 PROFI

Technický nástroj na vyhľadávanie value tipov vo futbale. Nie je to garancia zisku. Používaj iba s vlastným risk manažmentom.

## Čo je nové vo v10

- konzervatívnejší value scoring
- kontrola extrémnych kurzov a podozrivo vysokého edge
- explicitná timezone `Europe/Bratislava`
- automatická football-data sezóna s možnosťou override
- kontrola `Over/Under 2.5` podľa reálnej API hranice `point`
- deduplikácia na najlepší kurz pre rovnaký zápas/tip
- portfólio limity: denná expozícia, expozícia na zápas, expozícia na ligu
- fair odds, implied probability, EV v EUR a risk level v reporte
- CSV aj JSON export
- CLI prepínače pre dry-run, backtest, ligu, bank a top počet tipov

## Inštalácia

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements_v10.txt
cp env_v10_example.txt .env
```

Vyplň v `.env` aspoň:

```env
ODDS_API_KEY=...
AKTUALNY_BANK=1000
```

## Použitie

```bash
python main_v10_profi_betting.py --dry-run
python main_v10_profi_betting.py --league "Premier League" --dry-run
python main_v10_profi_betting.py --settle-only
python main_v10_profi_betting.py --backtest --backtest-days 180
python main_v10_profi_betting.py --bank 500 --min-edge 0.06 --top 10 --no-email
```

## Dôležité limity

Skutočný historický backtest podľa uzatváracích kurzov vyžaduje archivovať odds snapshoty pred zápasom. Táto verzia robí DB backtest iba nad tipmi, ktoré už skript v minulosti uložil a vyhodnotil.

## Odporúčaný režim

Najprv spúšťaj 2 až 4 týždne iba:

```bash
python main_v10_profi_betting.py --dry-run --no-email
```

Potom sleduj rozdiel medzi modelovým edge a reálnym výsledkom. Až keď máš dostatočný sample, zapínaj reálne staking rozhodnutia.

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import io, os, logging, joblib
from scipy.stats import poisson
from datetime import datetime, timedelta
from dotenv import load_dotenv

# ================= CONFIG =================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
load_dotenv()

API_ODDS_KEY = os.getenv('ODDS_API_KEY')

# Bezpečné načítanie BANKU (ošetrenie prázdnych hodnôt)
bank_env = os.getenv('AKTUALNY_BANK', '1000')
if not bank_env or bank_env.strip() == "":
    BANK = 1000.0
else:
    try:
        BANK = float(bank_env.strip())
    except ValueError:
        BANK = 1000.0

KELLY_FRAC = 0.25
HISTORY_FILE = "historia_tipov.csv"
MODEL_FILE = "ai_model.pkl"

# Rozšírený zoznam líg
LIGY = {
    'Premier League': {'csv': 'E0', 'api': 'soccer_epl', 'ha': 0.35},
    'La Liga': {'csv': 'SP1', 'api': 'soccer_spain_la_liga', 'ha': 0.38},
    'Bundesliga': {'csv': 'D1', 'api': 'soccer_germany_bundesliga', 'ha': 0.40},
    'Serie A': {'csv': 'I1', 'api': 'soccer_italy_serie_a', 'ha': 0.30},
    'Ligue 1': {'csv': 'F1', 'api': 'soccer_france_ligue_one', 'ha': 0.32},
    'Eredivisie': {'csv': 'N1', 'api': 'soccer_netherlands_eredivisie', 'ha': 0.42},   
    'Championship': {'csv': 'E1', 'api': 'soccer_efl_champ', 'ha': 0.28},
    'Primeira Liga': {'csv': 'P1', 'api': 'soccer_portugal_primeira_liga', 'ha': 0.35},
    'Jupiler Pro League': {'csv': 'B1', 'api': 'soccer_belgium_pro_league', 'ha': 0.38},
    'Super Lig': {'csv': 'T1', 'api': 'soccer_turkey_super_league', 'ha': 0.40},
    'Scottish Premiership': {'csv': 'SC0', 'api': 'soccer_scotland_premier_league', 'ha': 0.36}
}

# ================= MODEL & MATH =================
async def fetch_csv(session, url):
    for _ in range(3):
        try:
            async with session.get(url) as r:
                if r.status == 200:
                    return pd.read_csv(io.StringIO((await r.read()).decode('utf-8', errors='ignore')))
        except Exception as e:
            logging.warning(f"Retry CSV: {e}")
            await asyncio.sleep(1)
    return None

def calculate_team_strength(df):
    league_home_avg = df['FTHG'].mean()
    league_away_avg = df['FTAG'].mean()
    teams = {}
    for team in set(df['HomeTeam']):
        home = df[df['HomeTeam'] == team]
        away = df[df['AwayTeam'] == team]
        teams[team] = {
            'attack_home': home['FTHG'].mean() / league_home_avg,
            'defense_home': home['FTAG'].mean() / league_away_avg,
            'attack_away': away['FTAG'].mean() / league_away_avg,
            'defense_away': away['FTHG'].mean() / league_home_avg
        }
    return teams, league_home_avg, league_away_avg

def expected_goals(home, away, teams, avg_h, avg_a, ha):
    h, a_team = teams.get(home), teams.get(away)
    if not h or not a_team: return avg_h, avg_a
    lh = h['attack_home'] * a_team['defense_away'] * avg_h + ha
    la = a_team['attack_away'] * h['defense_home'] * avg_a
    return lh, la

def poisson_probs(lh, la):
    matrix = np.zeros((6, 6))
    for x in range(6):
        for y in range(6):
            matrix[x, y] = poisson.pmf(x, lh) * poisson.pmf(y, la)
    matrix /= matrix.sum()
    return {
        '1': np.sum(np.tril(matrix, -1)),
        'X': np.sum(np.diag(matrix)),
        '2': np.sum(np.triu(matrix, 1)),
        'Over 2.5': sum(matrix[x, y] for x in range(6) for y in range(6) if x+y > 2.5)
    }

# ================= AI & STRATEGY =================
def train_model():
    if not os.path.exists(HISTORY_FILE): return None, None
    try:
        df = pd.read_csv(HISTORY_FILE).dropna(subset=['Vysledok', 'Edge', 'Kurz', 'lh', 'la'])
        # Potrebujeme aspoň 50-100 vyhodnotených zápasov (V/P)
        df = df[df['Vysledok'].isin(['V', 'P'])].copy()
        if len(df) < 50: return None, None

        df['win'] = (df['Vysledok'] == 'V').astype(int)
        features = ['Edge', 'Kurz', 'lh', 'la']
        
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=150, max_depth=4, verbosity=-1)
        model.fit(df[features], df['win'])
        joblib.dump((model, features), MODEL_FILE)
        return model, features
    except Exception as e:
        logging.error(f"AI Tréning chyba: {e}")
        return None, None

def kelly(prob, odds):
    k = ((odds - 1) * prob - (1 - prob)) / (odds - 1)
    return max(0, k * KELLY_FRAC)

# ================= CORE =================
async def process_league(session, name, cfg, model, features):
    url = f"https://www.football-data.co.uk/mmz4281/2526/{cfg['csv']}.csv"
    df = await fetch_csv(session, url)
    if df is None: return []

    df = df.dropna(subset=['FTHG','FTAG'])
    teams, avg_h, avg_a = calculate_team_strength(df)

    async with session.get(f'https://api.the-odds-api.com/v4/sports/{cfg["api"]}/odds/',
                           params={'apiKey': API_ODDS_KEY, 'regions': 'eu', 'markets': 'h2h,totals'}) as r:
        if r.status != 200: return []
        odds_data = await r.json()

    bets, now = [], datetime.utcnow()

    for m in odds_data:
        h, a = m['home_team'], m['away_team']
        try:
            t = datetime.strptime(m['commence_time'], "%Y-%m-%dT%H:%M:%SZ")
        except: continue

        if not (now <= t <= now + timedelta(hours=48)): continue

        lh, la = expected_goals(h, a, teams, avg_h, avg_a, cfg['ha'])
        probs = poisson_probs(lh, la)

        for bk in m['bookmakers']:
            for mk in bk['markets']:
                for out in mk['outcomes']:
                    lbl = '1' if out['name']==h else ('2' if out['name']==a else ('X' if out['name']=='Draw' else 'Over 2.5'))
                    if lbl not in probs: continue

                    odds = out['price']
                    edge = (probs[lbl] * odds) - 1

                    # Filtre pre value bet
                    if 0.03 <= edge <= 0.25:
                        if model:
                            row = pd.DataFrame([{'Edge': edge, 'Kurz': odds, 'lh': lh, 'la': la}])
                            if model.predict(row[features])[0] == 0: continue

                        stake = kelly(probs[lbl], odds) * BANK
                        bets.append({
                            'Datum': t.strftime('%d.%m.%Y'),
                            'Zápas': f"{h} vs {a}",
                            'Tip': lbl,
                            'Kurz': odds,
                            'Edge': round(edge, 4),
                            'lh': round(lh, 2),
                            'la': round(la, 2),
                            'Vklad': round(stake, 2),
                            'Vysledok': ''
                        })
    return bets

# ================= MAIN =================
async def main():
    if not API_ODDS_KEY:
        logging.error("Chýba API kľúč")
        return

    async with aiohttp.ClientSession() as session:
        model, features = train_model()
        tasks = [process_league(session, n, c, model, features) for n, c in LIGY.items()]
        results = await asyncio.gather(*tasks)
        bets = [b for sub in results for b in sub]

        if bets:
            new_df = pd.DataFrame(bets)
            if os.path.exists(HISTORY_FILE):
                hist_df = pd.read_csv(HISTORY_FILE)
                final_df = pd.concat([hist_df, new_df]).drop_duplicates(subset=['Zápas', 'Tip', 'Datum'], keep='first')
            else:
                final_df = new_df
            
            final_df.to_csv(HISTORY_FILE, index=False)
            logging.info(f"Uložené tipy: {len(bets)}")
        else:
            logging.info("Žiadne nové tipy.")

if __name__ == "__main__":
    asyncio.run(main())

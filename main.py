import asyncio, aiohttp, pandas as pd, numpy as np, io, os, logging, smtplib, joblib
from scipy.stats import poisson
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

# --- CONFIG ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
load_dotenv()

# Tieto premenné si skript potiahne zo Secrets na GitHube
API_ODDS_KEY = os.getenv('ODDS_API_KEY')
GMAIL_USER = os.getenv('GMAIL_USER')
GMAIL_PASSWORD = os.getenv('GMAIL_PASSWORD')
GMAIL_RECEIVER = os.getenv('GMAIL_RECEIVER', GMAIL_USER)

# --- OPRAVA CHYBY S PRÁZDNYM BANKOM ---
bank_env = os.getenv('AKTUALNY_BANK', '1000')

if not bank_env or bank_env.strip() == "":
    BANK = 1000.0
else:
    try:
        # Odstránime prípadné biele znaky a skúsime prevod na číslo
        BANK = float(bank_env.strip())
    except ValueError:
        BANK = 1000.0

KELLY_FRAC = 0.10
# --------------------------------------

HISTORY_FILE = "historia_tipov.csv"
MODEL_FILE = "ai_model.pkl"

# --- ROZŠÍRENÝ ZOZNAM LÍG (v4.0) ---
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

# --- MATEMATICKÉ JADRO (Dixon-Coles korekcia) ---
def rho_correction(x, y, lh, la, rho):
    if x == 0 and y == 0: return 1 - (lh * la * rho)
    if x == 0 and y == 1: return 1 + (lh * rho)
    if x == 1 and y == 0: return 1 + (la * rho)
    if x == 1 and y == 1: return 1 - rho
    return 1

def poisson_probs_v4(lh, la, rho=-0.05):
    matrix = np.zeros((10, 10))
    for x in range(10):
        for y in range(10):
            p_x = poisson.pmf(x, lh)
            p_y = poisson.pmf(y, la)
            matrix[x, y] = p_x * p_y * rho_correction(x, y, lh, la, rho)
    
    matrix /= matrix.sum()

    return {
        '1': np.sum(np.tril(matrix, -1)),
        'X': np.sum(np.diag(matrix)),
        '2': np.sum(np.triu(matrix, 1)),
        'Over 2.5': np.sum([matrix[x, y] for x in range(10) for y in range(10) if x + y > 2.5])
    }

def safe_kelly(prob, odds):
    kelly = ((odds - 1) * prob - (1 - prob)) / (odds - 1)
    return max(0, min(kelly * KELLY_FRAC, 0.02))

def get_elo(df):
    ratings = {}
    for _, row in df.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        ratings.setdefault(h, 1500); ratings.setdefault(a, 1500)
        exp = 1 / (1 + 10 ** ((ratings[a] - ratings[h]) / 400))
        s = 1 if row['FTHG'] > row['FTAG'] else (0 if row['FTHG'] < row['FTAG'] else 0.5)
        ratings[h] += 20 * (s - exp); ratings[a] += 20 * ((1 - s) - (1 - exp))
    return ratings

# --- AI TRÉNING (LightGBM) ---
def train_model():
    if not os.path.exists(HISTORY_FILE): return None, None
    try:
        df = pd.read_csv(HISTORY_FILE)
        df = df[df['Vysledok'].isin(['V','P'])].copy()
        if len(df) < 50: return None, None

        df['EdgeNum'] = pd.to_numeric(df['Edge'], errors='coerce')
        df['KurzNum'] = pd.to_numeric(df['Kurz'], errors='coerce')
        df['LogOdds'] = np.log(df['KurzNum'])
        df['InvOdds'] = 1 / df['KurzNum']
        df['win'] = (df['Vysledok'] == 'V').astype(int)
        df = df.dropna()

        features = ['EdgeNum','KurzNum','LogOdds','InvOdds']
        
        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.03, max_depth=4, verbosity=-1)
        model.fit(df[features], df['win'])
        
        joblib.dump((model, features), MODEL_FILE)
        return model, features
    except Exception as e:
        logging.error(f"AI Model Error: {e}")
        return None, None

# --- CORE LOGIC ---
async def process_league(session, name, cfg, model, features):
    # Dynamicky ťaháme dáta pre sezónu 25/26
    url = f"https://www.football-data.co.uk/mmz4281/2526/{cfg['csv']}.csv"
    try:
        async with session.get(url) as r:
            if r.status != 200: return []
            df = pd.read_csv(io.StringIO((await r.read()).decode('utf-8', errors='ignore')))
    except: return []
    
    df = df.dropna(subset=['FTHG','FTAG'])
    elo = get_elo(df)
    avg_h, avg_a = df['FTHG'].mean(), df['FTAG'].mean()

    async with session.get(f'https://api.the-odds-api.com/v4/sports/{cfg["api"]}/odds/', 
                           params={'apiKey': API_ODDS_KEY, 'regions': 'eu', 'markets': 'h2h,totals'}) as r:
        if r.status != 200: return []
        odds_data = await r.json()

    now, bets = datetime.utcnow(), []

    for m in odds_data:
        h, a = m['home_team'], m['away_team']
        t = datetime.strptime(m['commence_time'], "%Y-%m-%dT%H:%M:%SZ")
        if not (now <= t <= now + timedelta(hours=48)): continue

        e_diff = (elo.get(h,1500) - elo.get(a,1500)) / 1000
        probs = poisson_probs_v4(avg_h + cfg['ha'] + e_diff, avg_a - e_diff)

        for bk in m['bookmakers']:
            for mk in bk['markets']:
                for out in mk['outcomes']:
                    lbl = '1' if out['name']==h else ('2' if out['name']==a else ('X' if out['name']=='Draw' else 'Over 2.5'))
                    if lbl not in probs: continue
                    
                    o_val, edge = out['price'], probs[lbl] * out['price'] - 1
                    
                    # Filtre: Edge 5% - 45%
                    if 0.05 <= edge <= 0.45:
                        if model:
                            row = pd.DataFrame([{'EdgeNum': edge*100, 'KurzNum': o_val, 'LogOdds': np.log(o_val), 'InvOdds': 1/o_val}])
                            if model.predict(row[features])[0] == 0: continue

                        bets.append({
                            'Datum': t.strftime('%d.%m.%Y'), 'Zápas': f"{h} vs {a}", 'Tip': lbl, 
                            'Kurz': o_val, 'Edge': round(edge*100,1), 'Vklad': round(safe_kelly(probs[lbl], o_val) * BANK, 2), 'Vysledok': ''
                        })
    return bets

# --- HLAVNÝ SPÚŠŤAČ ---
async def main():
    if not API_ODDS_KEY:
        logging.error("Chýba API kľúč!")
        return

    async with aiohttp.ClientSession() as session:
        model, features = train_model()
        tasks = [process_league(session, n, c, model, features) for n, c in LIGY.items()]
        results = await asyncio.gather(*tasks)
        all_bets = [b for sub in results for b in sub]

        if all_bets:
            new_df = pd.DataFrame(all_bets)
            if os.path.exists(HISTORY_FILE):
                hist_df = pd.read_csv(HISTORY_FILE)
                final_df = pd.concat([hist_df, new_df]).drop_duplicates(subset=['Zápas','Tip'], keep='first')
            else:
                final_df = new_df
            
            final_df.to_csv(HISTORY_FILE, index=False)
            logging.info(f"✅ Hotovo. Nájdených {len(all_bets)} tipov.")
        else:
            logging.info("❌ Žiadne nové výhodné tipy sa nenašli.")

if __name__ == "__main__":
    asyncio.run(main())

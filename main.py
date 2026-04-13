import asyncio, aiohttp, pandas as pd, numpy as np, io, os, smtplib, logging, joblib
from scipy.stats import poisson
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from fuzzywuzzy import process
from xgboost import XGBClassifier

# --- 1. KONFIGURÁCIA ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
load_dotenv()

API_ODDS_KEY = os.getenv('ODDS_API_KEY')
GMAIL_USER = os.getenv('GMAIL_USER')
GMAIL_PASSWORD = os.getenv('GMAIL_PASSWORD')
GMAIL_RECEIVER = os.getenv('GMAIL_RECEIVER', GMAIL_USER)
AKTUALNY_BANK = float(os.getenv('AKTUALNY_BANK', 100))
HISTORY_FILE = "historia_tipov.csv"
MODEL_FILE = "ai_model.pkl"
KELLY_FRAC = 0.10

LIGY_CONFIG = {
    '⚽ Premier League':   {'csv': 'E0',  'api': 'soccer_epl', 'ha': 0.35},
    '⚽ Championship':     {'csv': 'E1',  'api': 'soccer_efl_champ', 'ha': 0.30},
    '⚽ La Liga':          {'csv': 'SP1', 'api': 'soccer_spain_la_liga', 'ha': 0.38},
    '⚽ Bundesliga':       {'csv': 'D1',  'api': 'soccer_germany_bundesliga', 'ha': 0.40},
    '⚽ Serie A':          {'csv': 'I1',  'api': 'soccer_italy_serie_a', 'ha': 0.30},
    '⚽ Ligue 1':          {'csv': 'F1',  'api': 'soccer_france_ligue_one', 'ha': 0.35},
    '⚽ Eredivisie':       {'csv': 'N1',  'api': 'soccer_netherlands_eredivisie', 'ha': 0.40}
}

# --- 2. AI MOZOG (XGBOOST) ---
def train_ai_model():
    if not os.path.exists(HISTORY_FILE): return None
    try:
        df = pd.read_csv(HISTORY_FILE)
        df = df[df['Vysledok'].isin(['V', 'P'])].copy()
        if len(df) < 100: return None # Minimálny počet dát pre štart

        df['win'] = (df['Vysledok'] == 'V').astype(int)
        df['EdgeNum'] = df['Edge'].str.replace('%','').astype(float)
        df['KurzNum'] = pd.to_numeric(df['Kurz'], errors='coerce')
        df = df.dropna(subset=['EdgeNum', 'KurzNum', 'win'])

        X = df[['EdgeNum', 'KurzNum']]
        y = df['win']

        model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05)
        model.fit(X, y)
        joblib.dump(model, MODEL_FILE)
        return model
    except Exception as e:
        logging.error(f"AI Training Error: {e}")
        return None

# --- 3. ELO SYSTÉM ---
def get_elo(df):
    ratings = {}
    for _, row in df.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        ratings.setdefault(h, 1500); ratings.setdefault(a, 1500)
        exp = 1/(1+10**((ratings[a]-ratings[h])/400))
        score = 1 if row['FTHG'] > row['FTAG'] else (0 if row['FTHG'] < row['FTAG'] else 0.5)
        ratings[h] += 20*(score-exp); ratings[a] += 20*((1-score)-(1-exp))
    return ratings

# --- 4. ANALÝZA ---
async def analyzuj():
    model = train_ai_model()
    async with aiohttp.ClientSession() as session:
        all_bets, now_utc = [], datetime.utcnow()

        for liga, cfg in LIGY_CONFIG.items():
            url_csv = f"https://www.football-data.co.uk/mmz4281/2526/{cfg['csv']}.csv"
            async with session.get(url_csv) as r_csv:
                if r_csv.status != 200: continue
                raw_data = await r_csv.read()
                df_stats = pd.read_csv(io.StringIO(raw_data.decode('utf-8')))
            
            elo = get_elo(df_stats)
            avg_h, avg_a = df_stats['FTHG'].mean(), df_stats['FTAG'].mean()
            
            async with session.get(f'https://api.the-odds-api.com/v4/sports/{cfg["api"]}/odds/', params={'apiKey':API_ODDS_KEY,'regions':'eu','markets':'h2h,totals'}) as r_odds:
                if r_odds.status != 200: continue
                odds_json = await r_odds.json()
                
                for m in odds_json:
                    try:
                        home, away = m['home_team'], m['away_team']
                        m_t = datetime.strptime(m['commence_time'], "%Y-%m-%dT%H:%M:%SZ")
                        if not (now_utc <= m_t <= now_utc + timedelta(hours=36)): continue

                        # Poisson + ELO adjustment
                        elo_diff = (elo.get(home, 1500) - elo.get(away, 1500)) / 1000
                        lh = (df_stats[df_stats['HomeTeam']==home]['FTHG'].mean() or avg_h) + cfg['ha'] + elo_diff
                        la = (df_stats[df_stats['AwayTeam']==away]['FTAG'].mean() or avg_a) - elo_diff
                        
                        matrix = np.outer(poisson.pmf(np.arange(10), max(0.1, lh)), poisson.pmf(np.arange(10), max(0.1, la)))
                        
                        # Over 2.5 s 20% penalizáciou
                        p_over = (1 - np.sum([matrix[i,j] for i in range(10) for j in range(10) if i+j < 2.5])) * 0.80
                        probs = {'1': np.sum(np.tril(matrix, -1)), 'X': np.sum(np.diag(matrix)), '2': np.sum(np.triu(matrix, 1)), 'Over 2.5': p_over}

                        for bk in m['bookmakers']:
                            for mk in bk['markets']:
                                for out in mk['outcomes']:
                                    lbl = '1' if out['name']==home else ('2' if out['name']==away else ('X' if out['name']=='Draw' else 'Over 2.5'))
                                    if lbl in probs:
                                        kurz = out['price']
                                        edge = (probs[lbl] * kurz) - 1
                                        
                                        if 0.05 <= edge <= 0.40 and kurz <= 5.0:
                                            # --- AI FILTER ---
                                            if model is not None:
                                                prediction = model.predict(pd.DataFrame([[edge*100, kurz]], columns=['EdgeNum', 'KurzNum']))[0]
                                                if prediction == 0: continue # AI zamietlo tip

                                            vklad = round(min(max(0, (((kurz-1)*probs[lbl]-(1-probs[lbl]))/(kurz-1))*KELLY_FRAC), 0.02)*AKTUALNY_BANK, 2)
                                            if kurz > 2.5: vklad = min(vklad, 12.0)
                                            
                                            all_bets.append({'Zápas': f"{home} vs {away}", 'Tip': lbl, 'Kurz': kurz, 'Edge': f"{round(edge*100,1)}%", 'Vklad': f"{vklad}€"})
                    except: continue

        if all_bets:
            final_df = pd.DataFrame(all_bets).sort_values('Vklad', ascending=False).drop_duplicates(subset=['Zápas'])
            posli_email(final_df.to_dict('records'))

def posli_email(bets):
    msg = MIMEMultipart(); msg['Subject'] = f"🤖 AI HYBRID REPORT - {len(bets)} tipov"; msg['To'] = GMAIL_RECEIVER
    html = "<h3>🚀 Model v3.0 (Poisson + ELO + XGBoost Filter)</h3><table border='1' style='border-collapse:collapse;'>"
    html += "<tr style='background:#eee;'><th>Zápas</th><th>Tip</th><th>Kurz</th><th>Edge</th><th>Vklad</th></tr>"
    for b in bets:
        html += f"<tr><td>{b['Zápas']}</td><td>{b['Tip']}</td><td>{b['Kurz']}</td><td>{b['Edge']}</td><td>{b['Vklad']}</td></tr>"
    msg.attach(MIMEText(html + "</table>", 'html'))
    with smtplib.SMTP('smtp.gmail.com', 587) as s:
        s.starttls(); s.login(GMAIL_USER, GMAIL_PASSWORD); s.send_message(msg)

if __name__ == "__main__":
    asyncio.run(analyzuj())

import asyncio, aiohttp, pandas as pd, numpy as np, io, os, smtplib, logging, joblib
from scipy.stats import poisson
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from xgboost import XGBClassifier

# --- 1. KONFIGURÁCIA ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
load_dotenv()

API_ODDS_KEY = os.getenv('ODDS_API_KEY')
GMAIL_USER = os.getenv('GMAIL_USER')
GMAIL_PASSWORD = os.getenv('GMAIL_PASSWORD')
GMAIL_RECEIVER = os.getenv('GMAIL_RECEIVER', GMAIL_USER)
AKTUALNY_BANK = float(os.getenv('AKTUALNY_BANK', 1000))
HISTORY_FILE = "historia_tipov.csv"
MODEL_FILE = "ai_model.pkl"
KELLY_FRAC = 0.10

LIGY_CONFIG = {
    '⚽ Premier League':   {'csv': 'E0',  'api': 'soccer_epl', 'sport': 'futbal', 'ha': 0.35},
    '⚽ Championship':     {'csv': 'E1',  'api': 'soccer_efl_champ', 'sport': 'futbal', 'ha': 0.30},
    '⚽ La Liga':          {'csv': 'SP1', 'api': 'soccer_spain_la_liga', 'sport': 'futbal', 'ha': 0.38},
    '⚽ Bundesliga':       {'csv': 'D1',  'api': 'soccer_germany_bundesliga', 'sport': 'futbal', 'ha': 0.40},
    '⚽ Serie A':          {'csv': 'I1',  'api': 'soccer_italy_serie_a', 'sport': 'futbal', 'ha': 0.30},
    '⚽ Ligue 1':          {'csv': 'F1',  'api': 'soccer_france_ligue_one', 'sport': 'futbal', 'ha': 0.35},
    '⚽ Eredivisie':       {'csv': 'N1',  'api': 'soccer_netherlands_eredivisie', 'sport': 'futbal', 'ha': 0.40},
    '🏒 NHL':              {'csv': 'NHL', 'api': 'icehockey_nhl', 'sport': 'hokej', 'ha': 0.05}
}

# --- 2. VYHODNOCOVANIE ---
async def vyhodnot_vysledky(session):
    if not os.path.exists(HISTORY_FILE): return
    try:
        df = pd.read_csv(HISTORY_FILE)
        df['Vysledok'] = df['Vysledok'].fillna('')
        mask = df['Vysledok'].astype(str).str.strip() == ''
        if not mask.any(): return

        logging.info(f"🤖 AI: Kontrola výsledkov...")
        for liga, cfg in LIGY_CONFIG.items():
            url = f"https://www.football-data.co.uk/mmz4281/2526/{cfg['csv']}.csv" if cfg['sport'] == 'futbal' else f"https://raw.githubusercontent.com/pavel-jara/hockey-data/master/data/{cfg['csv']}_2025.csv"
            async with session.get(url) as r:
                if r.status != 200: continue
                df_res = pd.read_csv(io.StringIO((await r.read()).decode('utf-8', errors='ignore')))
                if df_res.empty: continue
                if cfg['sport'] == 'hokej': df_res = df_res.rename(columns={'HT':'HomeTeam','AT':'AwayTeam','HG':'FTHG','AG':'FTAG'})

                for idx, row in df[mask].iterrows():
                    teams = str(row['Zápas']).split(' vs ')
                    if len(teams) < 2: continue
                    h_t, a_t = teams[0].strip(), teams[1].strip()
                    res_row = df_res[df_res['HomeTeam'].str.contains(h_t[:4], na=False, case=False) & df_res['AwayTeam'].str.contains(a_t[:4], na=False, case=False)]
                    if not res_row.empty:
                        last = res_row.iloc[-1]
                        gh, ga = last['FTHG'], last['FTAG']
                        tip, res = str(row['Tip']), ''
                        if tip == '1': res = 'V' if gh > ga else 'P'
                        elif tip == '2': res = 'V' if ga > gh else 'P'
                        elif tip == 'X': res = 'V' if gh == ga else 'P'
                        elif 'Over 2.5' in tip: res = 'V' if (gh + ga) > 2.5 else 'P'
                        if res: df.at[idx, 'Vysledok'] = res
        df.to_csv(HISTORY_FILE, index=False)
    except Exception as e: logging.error(f"Chyba vyhodnotenia: {e}")

# --- 3. AI TRÉNING ---
def train_ai_model():
    if not os.path.exists(HISTORY_FILE): return None
    try:
        df = pd.read_csv(HISTORY_FILE)
        df = df[df['Vysledok'].isin(['V', 'P'])].copy()
        if len(df) < 20: return None
        df['win'] = (df['Vysledok'] == 'V').astype(int)
        df['EdgeNum'] = pd.to_numeric(df['Edge'].astype(str).str.replace('%',''), errors='coerce')
        df['KurzNum'] = pd.to_numeric(df['Kurz'], errors='coerce')
        train_data = df.dropna(subset=['EdgeNum', 'KurzNum', 'win'])
        if len(train_data) < 20: return None
        model = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1)
        model.fit(train_data[['EdgeNum', 'KurzNum']], train_data['win'])
        return model
    except: return None

# --- 4. ANALÝZA ---
def get_elo(df):
    r = {}
    if df.empty: return r
    for _, row in df.iterrows():
        h, a = str(row['HomeTeam']), str(row['AwayTeam'])
        r.setdefault(h, 1500); r.setdefault(a, 1500)
        exp = 1/(1+10**((r[a]-r[h])/400))
        s = 1 if row['FTHG'] > row['FTAG'] else (0 if row['FTHG'] < row['FTAG'] else 0.5)
        r[h] += 20*(s-exp); r[a] += 20*((1-s)-(1-exp))
    return r

async def analyzuj():
    async with aiohttp.ClientSession() as session:
        await vyhodnot_vysledky(session)
        model = train_ai_model()
        all_bets, now_utc = [], datetime.utcnow()

        for liga, cfg in LIGY_CONFIG.items():
            try:
                url = f"https://www.football-data.co.uk/mmz4281/2526/{cfg['csv']}.csv" if cfg['sport'] == 'futbal' else f"https://raw.githubusercontent.com/pavel-jara/hockey-data/master/data/{cfg['csv']}_2025.csv"
                async with session.get(url) as r_csv:
                    if r_csv.status != 200: continue
                    df_stats = pd.read_csv(io.StringIO((await r_csv.read()).decode('utf-8', errors='ignore')))
                
                if df_stats.empty or len(df_stats) < 5: continue
                if cfg['sport'] == 'hokej': df_stats = df_stats.rename(columns={'HT':'HomeTeam','AT':'AwayTeam','HG':'FTHG','AG':'FTAG'})
                
                df_stats = df_stats.dropna(subset=['FTHG', 'FTAG'])
                elo_ratings = get_elo(df_stats)
                avg_h, avg_a = df_stats['FTHG'].mean(), df_stats['FTAG'].mean()
                
                async with session.get(f'https://api.the-odds-api.com/v4/sports/{cfg["api"]}/odds/', params={'apiKey':API_ODDS_KEY,'regions':'eu','markets':'h2h,totals'}) as r_odds:
                    if r_odds.status != 200: continue
                    for m in await r_odds.json():
                        h, a = m['home_team'], m['away_team']
                        m_t = datetime.strptime(m['commence_time'], "%Y-%m-%dT%H:%M:%SZ")
                        if not (now_utc <= m_t <= now_utc + timedelta(hours=48)): continue
                        e_diff = (elo_ratings.get(h, 1500) - elo_ratings.get(a, 1500)) / 1000
                        lh = (df_stats[df_stats['HomeTeam']==h]['FTHG'].mean() or avg_h) + cfg['ha'] + e_diff
                        la = (df_stats[df_stats['AwayTeam']==a]['FTAG'].mean() or avg_a) - e_diff
                        matrix = np.outer(poisson.pmf(np.arange(10), max(0.1, lh)), poisson.pmf(np.arange(10), max(0.1, la)))
                        p_ov = (1 - np.sum([matrix[i,j] for i in range(10) for j in range(10) if i+j < 2.5]))
                        probs = {'1': np.sum(np.tril(matrix, -1)), 'X': np.sum(np.diag(matrix)), '2': np.sum(np.triu(matrix, 1)), 'Over 2.5': p_ov}

                        for bk in m['bookmakers']:
                            for mk in bk['markets']:
                                for out in mk['outcomes']:
                                    lbl = '1' if out['name']==h else ('2' if out['name']==a else ('X' if out['name']=='Draw' else 'Over 2.5'))
                                    if lbl in probs:
                                        k, edge = out['price'], (probs[lbl] * out['price']) - 1
                                        if 0.05 <= edge <= 0.45:
                                            if model is not None and model.predict(pd.DataFrame([[edge*100, k]], columns=['EdgeNum', 'KurzNum']))[0] == 0: continue
                                            vklad = round(min(max(0, (((k-1)*probs[lbl]-(1-probs[lbl]))/(k-1))*KELLY_FRAC), 0.02)*AKTUALNY_BANK, 2)
                                            all_bets.append({'Datum': m_t.strftime('%d.%m.%Y'), 'Zápas': f"{h} vs {a}", 'Tip': lbl, 'Kurz': k, 'Edge': f"{round(edge*100,1)}%", 'Vklad': f"{vklad}€", 'Sport': cfg['sport'], 'Vysledok': ''})
            except: continue

        if all_bets:
            new_df = pd.DataFrame(all_bets).drop_duplicates(subset=['Zápas', 'Tip'])
            if os.path.exists(HISTORY_FILE):
                final_df = pd.concat([pd.read_csv(HISTORY_FILE), new_df]).drop_duplicates(subset=['Zápas', 'Tip'], keep='first')
            else: final_df = new_df
            final_df.to_csv(HISTORY_FILE, index=False)
            posli_email(new_df.to_dict('records'))

def posli_email(bets):
    if not GMAIL_USER or not GMAIL_PASSWORD: return
    msg = MIMEMultipart(); msg['Subject'] = f"🤖 AI REPORT - {len(bets)} tipov"; msg['To'] = GMAIL_RECEIVER
    html = f"<h3>🚀 Model v3.6</h3><table border='1' style='border-collapse:collapse; width:100%;'>"
    html += "<tr style='background:#333; color:white;'><th>Sport</th><th>Zápas</th><th>Tip</th><th>Kurz</th><th>Edge</th><th>Vklad</th></tr>"
    for b in bets: html += f"<tr><td>{b['Sport']}</td><td>{b['Zápas']}</td><td>{b['Tip']}</td><td>{b['Kurz']}</td><td>{b['Edge']}</td><td>{b['Vklad']}</td></tr>"
    msg.attach(MIMEText(html + "</table>", 'html'))
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.starttls(); s.login(GMAIL_USER, GMAIL_PASSWORD); s.send_message(msg)
    except: pass

if __name__ == "__main__":
    asyncio.run(analyzuj())

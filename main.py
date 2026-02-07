import asyncio, aiohttp, pandas as pd, numpy as np, io, os, smtplib, logging
from scipy.stats import poisson
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from fuzzywuzzy import process

# --- 1. KONFIGURÁCIA ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
load_dotenv()

API_ODDS_KEY = os.getenv('ODDS_API_KEY')
GMAIL_USER = os.getenv('GMAIL_USER')
GMAIL_PASSWORD = os.getenv('GMAIL_PASSWORD')
GMAIL_RECEIVER = os.getenv('GMAIL_RECEIVER', GMAIL_USER)
AKTUALNY_BANK = float(os.getenv('AKTUALNY_BANK', 1000))
HISTORY_FILE = "historia_tipov.csv"
KELLY_FRAC = 0.25 

LIGY_CONFIG = {
    '⚽ Premier League':   {'csv': 'E0',  'api': 'soccer_epl', 'sport': 'futbal', 'ha': 0.25},
    '⚽ Championship':     {'csv': 'E1',  'api': 'soccer_efl_champ', 'sport': 'futbal', 'ha': 0.20},
    '⚽ La Liga':          {'csv': 'SP1', 'api': 'soccer_spain_la_liga', 'sport': 'futbal', 'ha': 0.28},
    '⚽ La Liga 2':        {'csv': 'SP2', 'api': 'soccer_spain_segunda_division', 'sport': 'futbal', 'ha': 0.22},
    '⚽ Bundesliga':       {'csv': 'D1',  'api': 'soccer_germany_bundesliga', 'sport': 'futbal', 'ha': 0.30},
    '⚽ Bundesliga 2':     {'csv': 'D2',  'api': 'soccer_germany_2_bundesliga', 'sport': 'futbal', 'ha': 0.25},
    '⚽ Serie A':          {'csv': 'I1',  'api': 'soccer_italy_serie_a', 'sport': 'futbal', 'ha': 0.22},
    '⚽ Serie B':          {'csv': 'I2',  'api': 'soccer_italy_serie_b', 'sport': 'futbal', 'ha': 0.18},
    '⚽ Ligue 1':          {'csv': 'F1',  'api': 'soccer_france_ligue_one', 'sport': 'futbal', 'ha': 0.25},
    '⚽ Ligue 2':          {'csv': 'F2',  'api': 'soccer_france_ligue_two', 'sport': 'futbal', 'ha': 0.20},
    '⚽ Eredivisie':       {'csv': 'N1',  'api': 'soccer_netherlands_eredivisie', 'sport': 'futbal', 'ha': 0.35},
    '⚽ Liga Portugal':    {'csv': 'P1',  'api': 'soccer_portugal_primeira_liga', 'sport': 'futbal', 'ha': 0.30},
    '⚽ Süper Lig (TR)':   {'csv': 'T1',  'api': 'soccer_turkey_super_league', 'sport': 'futbal', 'ha': 0.32},
    '⚽ Belgicko Jupiler': {'csv': 'B1',  'api': 'soccer_belgium_first_division', 'sport': 'futbal', 'ha': 0.28},
    '🏒 NHL':              {'csv': 'NHL', 'api': 'icehockey_nhl', 'sport': 'hokej', 'ha': 0.15},
    '🏒 Česko Extraliga':  {'csv': 'CZE', 'api': 'icehockey_czech_extraliga', 'sport': 'hokej', 'ha': 0.30},
    '🏒 Slovensko':        {'csv': 'SVK', 'api': 'icehockey_slovakia_extraliga', 'sport': 'hokej', 'ha': 0.35},
    '🏒 Nemecko DEL':      {'csv': 'GER', 'api': 'icehockey_germany_del', 'sport': 'hokej', 'ha': 0.28},
    '🏒 Švédsko SHL':      {'csv': 'SWE', 'api': 'icehockey_sweden_shl', 'sport': 'hokej', 'ha': 0.25},
    '🏒 Fínsko Liiga':     {'csv': 'FIN', 'api': 'icehockey_finland_liiga', 'sport': 'hokej', 'ha': 0.22},
    '🎾 ATP Tenis':        {'csv': 'ATP', 'api': 'tennis_atp', 'sport': 'tenis', 'ha': 0}
}

# --- 2. VYHODNOCOVANIE ---
async def vyhodnot_vysledky(session):
    if not os.path.exists(HISTORY_FILE): return ""
    try:
        df = pd.read_csv(HISTORY_FILE)
        if df.empty: return ""
        if 'Vysledok' not in df.columns: df['Vysledok'] = None
        df['Vysledok'] = df['Vysledok'].astype(object)
        
        updates = 0
        # Vyhodnocovanie funguje pre futbal z football-data.co.uk (sezóna 25/26)
        async with session.get("https://www.football-data.co.uk/mmz4281/2526/E0.csv") as r:
            if r.status == 200:
                res_data = pd.read_csv(io.StringIO((await r.read()).decode('utf-8')))
                for idx, row in df.iterrows():
                    if pd.isna(row['Vysledok']) or row['Vysledok'] == "":
                        match_res = res_data[res_data['HomeTeam'] == row['Zápas'].split(' vs ')[0]]
                        if not match_res.empty:
                            fthg, ftag = match_res.iloc[-1]['FTHG'], match_res.iloc[-1]['FTAG']
                            tip, vyhra = str(row['Tip']), False
                            if tip == '1' and fthg > ftag: vyhra = True
                            elif tip == '2' and ftag > fthg: vyhra = True
                            elif tip == 'X' and fthg == ftag: vyhra = True
                            elif 'Over' in tip and (fthg + ftag) > float(tip.split()[-1]): vyhra = True
                            elif 'Under' in tip and (fthg + ftag) < float(tip.split()[-1]): vyhra = True
                            df.at[idx, 'Vysledok'] = 'V' if vyhra else 'P'
                            updates += 1
        if updates > 0:
            df.to_csv(HISTORY_FILE, index=False)
            return f"<p style='color:green;'>✅ Vyhodnotených {updates} starších zápasov.</p>"
    except Exception as e: logging.error(f"Chyba vyhodnotenia: {e}")
    return ""

# --- 3. TÝŽDENNÝ SUMÁR ---
def posli_tyzdenny_sumar():
    if datetime.now().weekday() != 6: return # Len v nedeľu
    if not os.path.exists(HISTORY_FILE): return
    try:
        df = pd.read_csv(HISTORY_FILE)
        df_eval = df[df['Vysledok'].isin(['V', 'P'])].copy()
        if df_eval.empty: return
        
        df_eval['Vklad_num'] = df_eval['Vklad'].str.replace('€', '').astype(float)
        df_eval['Zisk'] = df_eval.apply(lambda x: (x['Vklad_num']*x['Kurz']-x['Vklad_num']) if x['Vysledok']=='V' else -x['Vklad_num'], axis=1)
        
        celkovy = round(df_eval['Zisk'].sum(), 2)
        html = f"<h2>📊 Týždenný Sumár</h2><p>Zisk/Strata: <b>{celkovy}€</b></p><p>Tipov: {len(df_eval)}</p>"
        
        msg = MIMEMultipart()
        msg['Subject'] = f"📊 SUMÁR: {celkovy}€"
        msg['To'] = GMAIL_RECEIVER if GMAIL_RECEIVER else GMAIL_USER
        msg.attach(MIMEText(html, 'html'))
        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.starttls(); s.login(GMAIL_USER, GMAIL_PASSWORD); s.send_message(msg)
    except Exception as e: logging.error(f"Sumar error: {e}")

# --- 4. HLAVNÁ ANALÝZA ---
async def fetch_csv(session, liga, cfg):
    try:
        now = datetime.now()
        if cfg['sport'] == 'futbal':
            sez = f"{now.strftime('%y')}{(now.year + 1) % 100:02d}" if now.month >= 8 else f"{(now.year - 1) % 100:02d}{now.strftime('%y')}"
            url = f"https://www.football-data.co.uk/mmz4281/{sez}/{cfg['csv']}.csv"
        else:
            url = f"https://raw.githubusercontent.com/pavel-jara/hockey-data/master/data/{cfg['csv']}_2025.csv"
        async with session.get(url) as r:
            return liga, await r.read() if r.status == 200 else None
    except: return liga, None

def spracuj_stats(content, cfg):
    try:
        df = pd.read_csv(io.StringIO(content.decode('utf-8', errors='ignore')))
        if cfg['sport'] == 'hokej':
            df = df.rename(columns={'home_team':'HomeTeam','away_team':'AwayTeam','HG':'FTHG','AG':'FTAG','HT':'HomeTeam','AT':'AwayTeam'})
        df = df.dropna(subset=['FTHG', 'FTAG'])
        avg_h, avg_a = df['FTHG'].mean(), df['FTAG'].mean()
        stats = pd.DataFrame(index=list(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())))
        h_s, a_s = df.groupby('HomeTeam').agg({'FTHG':'mean', 'FTAG':'mean'}), df.groupby('AwayTeam').agg({'FTAG':'mean', 'FTHG':'mean'})
        stats['AH'], stats['DH'] = h_s['FTHG']/avg_h, h_s['FTAG']/avg_a
        stats['AA'], stats['DA'] = a_s['FTAG']/avg_a, a_s['FTHG']/avg_h
        return stats.fillna(1.0), avg_h, avg_a
    except: return None, 0, 0

async def analyzuj():
    async with aiohttp.ClientSession() as session:
        log_vys = await vyhodnot_vysledky(session)
        csv_res = await asyncio.gather(*(fetch_csv(session, l, c) for l, c in LIGY_CONFIG.items()))
        all_bets, now_utc = [], datetime.utcnow()

        for liga, content in csv_res:
            if not content: continue
            cfg = LIGY_CONFIG[liga]
            stats, ah_avg, aa_avg = spracuj_stats(content, cfg)
            if stats is None: continue

            lim_h = 36 if cfg['sport'] == 'hokej' else 24
            async with session.get(f'https://api.the-odds-api.com/v4/sports/{cfg["api"]}/odds/', 
                                  params={'apiKey':API_ODDS_KEY,'regions':'eu','markets':'h2h,totals'}) as r:
                if r.status != 200: continue
                for m in await r.json():
                    try:
                        m_t = datetime.strptime(m['commence_time'], "%Y-%m-%dT%H:%M:%SZ")
                        if not (now_utc <= m_t <= now_utc + timedelta(hours=lim_h)): continue
                        c1_m, c2_m = process.extractOne(m['home_team'], stats.index), process.extractOne(m['away_team'], stats.index)
                        if c1_m[1] < 70 or c2_m[1] < 70: continue
                        c1, c2 = c1_m[0], c2_m[0]
                        lh, la = stats.at[c1,'AH']*stats.at[c2,'DA']*ah_avg + cfg['ha'], stats.at[c2,'AA']*stats.at[c1,'DH']*aa_avg
                        matrix = np.outer(poisson.pmf(np.arange(12), max(0.1, lh)), poisson.pmf(np.arange(12), max(0.1, la)))
                        lim = 5.5 if cfg['sport'] == 'hokej' else 2.5
                        probs = {'1': np.sum(np.tril(matrix, -1)), 'X': np.sum(np.diag(matrix)), '2': np.sum(np.triu(matrix, 1)),
                                 f'Over {lim}': 1 - np.sum([matrix[i,j] for i in range(12) for j in range(12) if i+j < lim])}

                        min_e = 0.01 if cfg['sport'] == 'hokej' else 0.05
                        for bk in m.get('bookmakers', []):
                            for mk in bk.get('markets', []):
                                for out in mk['outcomes']:
                                    lbl = f"{out['name']} {lim}" if mk['key']=='totals' and out.get('point')==lim else ('1' if out['name']==m['home_team'] else ('2' if out['name']==m['away_team'] else 'X'))
                                    if lbl in probs:
                                        edge = (probs[lbl] * out['price']) - 1
                                        if min_e <= edge <= 0.45:
                                            v = round(min(max(0, (((out['price']-1)*probs[lbl]-(1-probs[lbl]))/(out['price']-1))*KELLY_FRAC), 0.02)*AKTUALNY_BANK, 2)
                                            all_bets.append({'Zápas':f"{c1} vs {c2}", 'Tip':lbl, 'Kurz':out['price'], 'Edge':f"{round(edge*100,1)}%", 'Vklad':f"{v}€", 'Sport':cfg['sport']})
                    except: continue

        if all_bets:
            final = pd.DataFrame(all_bets).sort_values('Edge', ascending=False).drop_duplicates(subset=['Zápas', 'Tip']).to_dict('records')
            uloz_a_posli(final, log_vys)

def uloz_a_posli(bets, log_v):
    df_new = pd.DataFrame(bets)
    df_new['Datum'] = datetime.now().strftime('%d.%m.%Y')
    if os.path.exists(HISTORY_FILE):
        df_old = pd.read_csv(HISTORY_FILE)
        df_new = pd.concat([df_old, df_new]).drop_duplicates(subset=['Zápas', 'Tip', 'Datum'])
    df_new.to_csv(HISTORY_FILE, index=False)
    
    prijemca = GMAIL_RECEIVER if GMAIL_RECEIVER else GMAIL_USER
    if not prijemca: return
    
    msg = MIMEMultipart()
    msg['Subject'] = f"🚀 AI REPORT - {len(bets)} tipov"
    msg['To'] = prijemca
    html = f"{log_v}<h3>🎯 Nové Value Bets</h3><table border='1' style='border-collapse:collapse; width:100%;'>"
    html += "<tr style='background:#333; color:white;'><th>Šport</th><th>Zápas</th><th>Tip</th><th>Kurz</th><th>Edge</th><th>Vklad</th></tr>"
    for b in bets:
        icon = "🏒" if b['Sport'] == 'hokej' else "⚽"
        html += f"<tr><td>{icon}</td><td>{b['Zápas']}</td><td>{b['Tip']}</td><td>{b['Kurz']}</td><td>{b['Edge']}</td><td>{b['Vklad']}</td></tr>"
    msg.attach(MIMEText(html + "</table>", 'html'))
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.starttls(); s.login(GMAIL_USER, GMAIL_PASSWORD); s.send_message(msg)
    except Exception as e: logging.error(f"Email error: {e}")

if __name__ == "__main__":
    asyncio.run(analyzuj())
    posli_tyzdenny_sumar()

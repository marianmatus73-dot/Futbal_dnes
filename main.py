import asyncio, aiohttp, pandas as pd, numpy as np, io, os, smtplib, logging, sys
from scipy.stats import poisson
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from fuzzywuzzy import process

# --- 1. KONFIGURÁCIA A LOGOVANIE ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
load_dotenv()

API_ODDS_KEY = os.getenv('ODDS_API_KEY')
GMAIL_USER = os.getenv('GMAIL_USER')
GMAIL_PASSWORD = os.getenv('GMAIL_PASSWORD')
GMAIL_RECEIVER = os.getenv('GMAIL_RECEIVER', GMAIL_USER)
AKTUALNY_BANK = float(os.getenv('AKTUALNY_BANK', 1000))
HISTORY_FILE = "historia_tipov.csv"
KELLY_FRAC = 0.12  # Konzervatívny prístup pre stabilnejší rast

LIGY_CONFIG = {
    # --- FUTBAL ---
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
    
    # --- HOKEJ ---
    '🏒 NHL':              {'csv': 'NHL', 'api': 'icehockey_nhl', 'sport': 'hokej', 'ha': 0.12},
    '🏒 Česko Extraliga':  {'csv': 'CZE', 'api': 'icehockey_czech_extraliga', 'sport': 'hokej', 'ha': 0.20},
    '🏒 Slovensko':        {'csv': 'SVK', 'api': 'icehockey_slovakia_extraliga', 'sport': 'hokej', 'ha': 0.25},
    '🏒 Nemecko DEL':      {'csv': 'GER', 'api': 'icehockey_germany_del', 'sport': 'hokej', 'ha': 0.22},
    '🏒 Švédsko SHL':      {'csv': 'SWE', 'api': 'icehockey_sweden_shl', 'sport': 'hokej', 'ha': 0.18},
    '🏒 Fínsko Liiga':     {'csv': 'FIN', 'api': 'icehockey_finland_liiga', 'sport': 'hokej', 'ha': 0.15}
}

# --- 2. VYHODNOCOVANIE ---
async def vyhodnot_vysledky(session):
    if not os.path.exists(HISTORY_FILE): return ""
    try:
        df = pd.read_csv(HISTORY_FILE)
        if df.empty: return ""
        df['Vysledok'] = df['Vysledok'].astype(object)
        
        updates = 0
        # Rozšírený zoznam líg pre sťahovanie výsledkov
        league_codes = ['E0', 'E1', 'SP1', 'SP2', 'D1', 'D2', 'I1', 'I2', 'F1', 'F2', 'N1', 'P1', 'T1', 'B1']
        
        for code in league_codes:
            url = f"https://www.football-data.co.uk/mmz4281/2526/{code}.csv"
            async with session.get(url) as r:
                if r.status == 200:
                    res_data = pd.read_csv(io.StringIO((await r.read()).decode('utf-8')))
                    for idx, row in df.iterrows():
                        if pd.isna(row.get('Vysledok')) or row.get('Vysledok') == "":
                            home_team = str(row['Zápas']).split(' vs ')[0]
                            match_res = res_data[res_data['HomeTeam'] == home_team]
                            if not match_res.empty:
                                fthg, ftag = match_res.iloc[-1]['FTHG'], match_res.iloc[-1]['FTAG']
                                tip = str(row['Tip'])
                                vyhra = False
                                if tip == '1' and fthg > ftag: vyhra = True
                                elif tip == '2' and ftag > fthg: vyhra = True
                                elif tip == 'X' and fthg == ftag: vyhra = True
                                elif 'Over' in tip and (fthg + ftag) > 2.5: vyhra = True
                                elif 'Under' in tip and (fthg + ftag) < 2.5: vyhra = True
                                df.at[idx, 'Vysledok'] = 'V' if vyhra else 'P'
                                updates += 1
        if updates > 0:
            df.to_csv(HISTORY_FILE, index=False)
            return f"<p style='color:green;'>✅ Automaticky vyhodnotených {updates} zápasov.</p>"
    except Exception as e: logging.error(f"Chyba vyhodnotenia: {e}")
    return ""

# --- 3. JADRO ANALÝZY ---
def spracuj_stats(content, cfg):
    try:
        df = pd.read_csv(io.StringIO(content.decode('utf-8', errors='ignore')))
        if cfg['sport'] == 'hokej':
            df = df.rename(columns={'home_team':'HomeTeam','away_team':'AwayTeam','HG':'FTHG','AG':'FTAG','HT':'HomeTeam','AT':'AwayTeam'})
        df = df.dropna(subset=['FTHG', 'FTAG'])
        avg_h, avg_a = df['FTHG'].mean(), df['FTAG'].mean()
        stats = pd.DataFrame(index=list(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())))
        h_s = df.groupby('HomeTeam').agg({'FTHG':'mean', 'FTAG':'mean'})
        a_s = df.groupby('AwayTeam').agg({'FTAG':'mean', 'FTHG':'mean'})
        stats['AH'] = h_s['FTHG'] / avg_h; stats['DH'] = h_s['FTAG'] / avg_a
        stats['AA'] = a_s['FTAG'] / avg_a; stats['DA'] = a_s['FTHG'] / avg_h
        return stats.fillna(1.0), avg_h, avg_a
    except: return None, 0, 0

async def analyzuj():
    async with aiohttp.ClientSession() as session:
        log_vys = await vyhodnot_vysledky(session)
        all_bets, now_utc = [], datetime.utcnow()

        for liga, cfg in LIGY_CONFIG.items():
            # URL pre historické dáta
            url_csv = f"https://www.football-data.co.uk/mmz4281/2526/{cfg['csv']}.csv" if cfg['sport']=='futbal' else f"https://raw.githubusercontent.com/pavel-jara/hockey-data/master/data/{cfg['csv']}_2025.csv"
            
            async with session.get(url_csv) as r_csv:
                if r_csv.status != 200: continue
                content = await r_csv.read()
                stats, ah_avg, aa_avg = spracuj_stats(content, cfg)
            
            if stats is None: continue

            # Odds API
            async with session.get(f'https://api.the-odds-api.com/v4/sports/{cfg["api"]}/odds/', 
                                  params={'apiKey':API_ODDS_KEY,'regions':'eu','markets':'h2h,totals'}) as r_odds:
                if r_odds.status != 200: continue
                for m in await r_odds.json():
                    try:
                        m_t = datetime.strptime(m['commence_time'], "%Y-%m-%dT%H:%M:%SZ")
                        if not (now_utc <= m_t <= now_utc + timedelta(hours=36)): continue
                        
                        c1_m, c2_m = process.extractOne(m['home_team'], stats.index), process.extractOne(m['away_team'], stats.index)
                        if c1_m[1] < 75 or c2_m[1] < 75: continue
                        c1, c2 = c1_m[0], c2_m[0]
                        
                        lh, la = stats.at[c1,'AH']*stats.at[c2,'DA']*ah_avg + cfg['ha'], stats.at[c2,'AA']*stats.at[c1,'DH']*aa_avg
                        matrix = np.outer(poisson.pmf(np.arange(10), max(0.1, lh)), poisson.pmf(np.arange(10), max(0.1, la)))
                        
                        probs = {'1': np.sum(np.tril(matrix, -1)), 'X': np.sum(np.diag(matrix)), '2': np.sum(np.triu(matrix, 1)),
                                 'Over 2.5': 1 - np.sum([matrix[i,j] for i in range(10) for j in range(10) if i+j < 2.5])}

                        for bk in m.get('bookmakers', []):
                            # Filtrujeme len relevantné stávkové kancelárie (voliteľné)
                            for mk in bk.get('markets', []):
                                for out in mk['outcomes']:
                                    lbl = '1' if out['name']==m['home_team'] else ('2' if out['name']==m['away_team'] else ('X' if out['name']=='Draw' else 'Over 2.5'))
                                    if lbl in probs:
                                        edge = (probs[lbl] * out['price']) - 1
                                        
                                        # --- FILTRE A POISTKY ---
                                        if out['price'] > 5.0: continue # Ignorujeme kurzy nad 5.0
                                        if edge > 0.45: continue        # Ignorujeme anomálie (pravdepodobne chyby v dátach)
                                        
                                        # Zastropovanie Edge pre Kellyho výpočet (limitujeme risk)
                                        effective_edge = min(edge, 0.25)
                                        
                                        if effective_edge >= 0.05:
                                            vklad = round(min(max(0, (((out['price']-1)*probs[lbl]-(1-probs[lbl]))/(out['price']-1))*KELLY_FRAC), 0.02)*AKTUALNY_BANK, 2)
                                            all_bets.append({
                                                'Zápas': f"{c1} vs {c2}",
                                                'Tip': lbl,
                                                'Kurz': out['price'],
                                                'Edge': f"{round(edge*100,1)}%",
                                                'Vklad': f"{vklad}€",
                                                'Sport': cfg['sport']
                                            })
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
    
    msg = MIMEMultipart()
    msg['Subject'] = f"🚀 AI REPORT - {len(bets)} tipov"
    msg['To'] = GMAIL_RECEIVER
    
    html = f"{log_v}<h3>🎯 Filtrované Value Bets (Max Kurz 5.0, Max Edge 25%)</h3>"
    html += "<table border='1' style='border-collapse:collapse; width:100%;'>"
    html += "<tr style='background:#333; color:white;'><th>Šport</th><th>Zápas</th><th>Tip</th><th>Kurz</th><th>Edge</th><th>Vklad</th></tr>"
    for b in bets:
        icon = '🏒' if b['Sport'] == 'hokej' else '⚽'
        html += f"<tr><td>{icon}</td><td>{b['Zápas']}</td><td>{b['Tip']}</td><td>{b['Kurz']}</td><td>{b['Edge']}</td><td>{b['Vklad']}</td></tr>"
    html += "</table>"
    
    msg.attach(MIMEText(html, 'html'))
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.starttls()
            s.login(GMAIL_USER, GMAIL_PASSWORD)
            s.send_message(msg)
    except Exception as e:
        logging.error(f"Chyba pri odosielaní emailu: {e}")

if __name__ == "__main__":
    asyncio.run(analyzuj())

import asyncio, aiohttp, pandas as pd, numpy as np, io, os, smtplib, csv, logging
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
    '⚽ La Liga':          {'csv': 'SP1', 'api': 'soccer_spain_la_liga', 'sport': 'futbal', 'ha': 0.28},
    '⚽ Bundesliga':       {'csv': 'D1',  'api': 'soccer_germany_bundesliga', 'sport': 'futbal', 'ha': 0.30},
    '⚽ Serie A':          {'csv': 'I1',  'api': 'soccer_italy_serie_a', 'sport': 'futbal', 'ha': 0.22},
    '⚽ Ligue 1':          {'csv': 'F1',  'api': 'soccer_france_ligue_one', 'sport': 'futbal', 'ha': 0.25},
    '⚽ Eredivisie':       {'csv': 'N1',  'api': 'soccer_netherlands_eredivisie', 'sport': 'futbal', 'ha': 0.35},
    '⚽ Liga Portugal':    {'csv': 'P1',  'api': 'soccer_portugal_primeira_liga', 'sport': 'futbal', 'ha': 0.30},
    '⚽ Süper Lig':        {'csv': 'T1',  'api': 'soccer_turkey_super_league', 'sport': 'futbal', 'ha': 0.32},
    '🏒 NHL':              {'csv': 'NHL', 'api': 'icehockey_nhl', 'sport': 'hokej', 'ha': 0.15},
    '🏒 Česko Extraliga':  {'csv': 'CZE', 'api': 'icehockey_czech_extraliga', 'sport': 'hokej', 'ha': 0.30},
    '🏒 Nemecko DEL':      {'csv': 'GER', 'api': 'icehockey_germany_del', 'sport': 'hokej', 'ha': 0.28},
    '🏒 Švédsko SHL':      {'csv': 'SWE', 'api': 'icehockey_sweden_shl', 'sport': 'hokej', 'ha': 0.25},
    '🏒 Fínsko Liiga':     {'csv': 'FIN', 'api': 'icehockey_finland_liiga', 'sport': 'hokej', 'ha': 0.22},
    '🏒 Slovensko':        {'csv': 'SVK', 'api': 'icehockey_slovakia_extraliga', 'sport': 'hokej', 'ha': 0.35},
    '🏀 NBA':              {'csv': 'NBA', 'api': 'basketball_nba', 'sport': 'basketbal', 'ha': 3.1},
    '🎾 ATP Tenis':        {'csv': 'ATP', 'api': 'tennis_atp', 'sport': 'tenis', 'ha': 0}
}

# --- 2. FUNKCIE ---

def uloz_a_clv(new_bets):
    clv_alerts = []
    headers = ['Datum', 'Zápas', 'Tip', 'Kurz_Start', 'CLV', 'Edge', 'Vklad', 'Sport', 'Vysledok']
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            if len(df.columns) != len(headers): df = pd.DataFrame(columns=headers)
        except: df = pd.DataFrame(columns=headers)
    else: df = pd.DataFrame(columns=headers)

    for b in new_bets:
        mask = (df['Zápas'] == b['Zápas']) & (df['Tip'] == b['Tip'])
        if df[mask].empty:
            new_row = {'Datum': datetime.now().strftime('%d.%m.%Y'), 'Zápas': b['Zápas'], 'Tip': b['Tip'], 'Kurz_Start': b['Kurz'], 'CLV': 100.0, 'Edge': b['Edge'], 'Vklad': b['Vklad'], 'Sport': b['Sport'], 'Vysledok': ''}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            idx = df[mask].index[0]
            try:
                start_p = float(df.at[idx, 'Kurz_Start'])
                clv_val = (start_p / b['Kurz']) * 100
                df.at[idx, 'CLV'] = round(clv_val, 1)
                if clv_val > 102.5: clv_alerts.append({'Z': b['Zápas'], 'T': b['Tip'], 'P': f"{round(clv_val-100,1)}%"})
            except: pass
    df.to_csv(HISTORY_FILE, index=False)
    return clv_alerts

async def fetch_csv(session, liga, cfg):
    try:
        now = datetime.now()
        if cfg['sport'] == 'futbal':
            sez = f"{now.strftime('%y')}{(now.year + 1) % 100:02d}" if now.month >= 8 else f"{(now.year - 1) % 100:02d}{now.strftime('%y')}"
            url = f"https://www.football-data.co.uk/mmz4281/{sez}/{cfg['csv']}.csv"
        elif cfg['sport'] == 'hokej':
            repo = "martineon/nhl-historical-data/master/data/nhl_results_2025.csv" if cfg['csv'] == 'NHL' else f"pavel-jara/hockey-data/master/data/{cfg['csv']}_2025.csv"
            url = f"https://raw.githubusercontent.com/{repo}"
        elif cfg['sport'] == 'tenis':
            url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{now.year}.csv"
        else: url = "https://raw.githubusercontent.com/fivethirtyeight/nba-model/master/nba_elo.csv"
        async with session.get(url, timeout=10) as r:
            if r.status == 200: return liga, await r.read()
        return liga, None
    except: return liga, None

def spracuj_stats(content, cfg):
    try:
        df = pd.read_csv(io.StringIO(content.decode('utf-8', errors='ignore')))
        if cfg['sport'] == 'tenis':
            w, l = df['winner_name'].value_counts(), df['loser_name'].value_counts()
            players = list(set(df['winner_name'].dropna()) | set(df['loser_name'].dropna()))
            stats = pd.DataFrame(index=players)
            stats['WinRate'] = [(w.get(p,0)+1)/(w.get(p,0)+l.get(p,0)+2) for p in players]
            return stats, 0, 0
        df = df.rename(columns={'home_team':'HomeTeam','away_team':'AwayTeam','home_goals':'FTHG','away_goals':'FTAG','team1':'HomeTeam','team2':'AwayTeam','score1':'FTHG','score2':'FTAG'})
        avg_h, avg_a = df['FTHG'].mean(), df['FTAG'].mean()
        all_teams = list(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
        stats = pd.DataFrame(index=all_teams)
        h_stats = df.groupby('HomeTeam').agg({'FTHG':'mean', 'FTAG':'mean'})
        a_stats = df.groupby('AwayTeam').agg({'FTAG':'mean', 'FTHG':'mean'})
        stats['AH'] = h_stats['FTHG'] / avg_h; stats['DH'] = h_stats['FTAG'] / avg_a
        stats['AA'] = a_stats['FTAG'] / avg_a; stats['DA'] = a_stats['FTHG'] / avg_h
        return stats.fillna(1.0), avg_h, avg_a
    except: return None, 0, 0

async def analyzuj():
    async with aiohttp.ClientSession() as session:
        csv_results = await asyncio.gather(*(fetch_csv(session, l, c) for l, c in LIGY_CONFIG.items()))
        all_bets = []
        now_utc, limit_utc = datetime.utcnow(), datetime.utcnow() + timedelta(hours=24)

        for liga, content in csv_results:
            if not content: continue
            cfg = LIGY_CONFIG[liga]; stats, ah, aa = spracuj_stats(content, cfg)
            if stats is None: continue
            async with session.get(f'https://api.the-odds-api.com/v4/sports/{cfg["api"]}/odds/', params={'apiKey':API_ODDS_KEY,'regions':'eu','markets':'h2h,totals'}) as r:
                if r.status != 200: continue
                matches = await r.json()

            for m in matches:
                m_time = datetime.strptime(m['commence_time'], "%Y-%m-%dT%H:%M:%SZ")
                if not (now_utc <= m_time <= limit_utc): continue
                c1_m, c2_m = process.extractOne(m['home_team'], stats.index), process.extractOne(m['away_team'], stats.index)
                if not c1_m or not c2_m or c1_m[1] < 70: continue
                c1, c2 = c1_m[0], c2_m[0]
                
                probs, lim = {}, (5.5 if cfg['sport']=='hokej' else 2.5)
                if cfg['sport'] == 'tenis':
                    w1, w2 = stats.at[c1,'WinRate'], stats.at[c2,'WinRate']
                    probs = {'1': w1/(w1+w2), '2': w2/(w1+w2)}
                else:
                    lh, la = (stats.at[c1,'AH']*stats.at[c2,'DA']*ah + cfg['ha']), (stats.at[c2,'AA']*stats.at[c1,'DH']*aa)
                    pu = sum(poisson.pmf(i,lh)*poisson.pmf(j,la) for i in range(12) for j in range(12) if i+j < lim)
                    probs = {'1':sum(poisson.pmf(i,lh)*poisson.pmf(j,la) for i in range(12) for j in range(i)), 'X':sum(poisson.pmf(i,lh)*poisson.pmf(i,la) for i in range(12)), '2':sum(poisson.pmf(i,lh)*poisson.pmf(j,la) for i in range(12) for j in range(i+1,12)), f'Over {lim}':1-pu, f'Under {lim}':pu}

                match_best_odds = {}
                for bk in m.get('bookmakers', []):
                    for mk in bk.get('markets', []):
                        for out in mk['outcomes']:
                            lbl = f"{out['name']} {lim}" if mk['key']=='totals' and out.get('point')==lim else ('1' if out['name']==m['home_team'] else ('2' if out['name']==m['away_team'] else 'X'))
                            if lbl in probs:
                                prob, price = probs[lbl], out['price']
                                edge = (prob * price) - 1
                                if 0.05 <= edge <= 0.45:
                                    if lbl not in match_best_odds or price > match_best_odds[lbl]['Kurz']:
                                        v = round(min(max(0, (((price-1)*prob-(1-prob))/(price-1))*KELLY_FRAC), 0.02)*AKTUALNY_BANK, 2)
                                        match_best_odds[lbl] = {'Zápas':f"{c1} vs {c2}", 'Tip':lbl, 'Kurz':price, 'Edge':f"{round(edge*100,1)}%", 'Vklad':f"{v}€", 'Sport':cfg['sport'], 'Skóre':f"{round(lh,1)}:{round(la,1)}" if cfg['sport']!='tenis' else ""}
                all_bets.extend(match_best_odds.values())

        if all_bets:
            final_bets = pd.DataFrame(all_bets).sort_values('Kurz', ascending=False).drop_duplicates(subset=['Zápas', 'Tip']).to_dict('records')
            alerts = uloz_a_clv(final_bets)
            posli_email(final_bets, alerts)

def posli_email(bets, alerts):
    msg = MIMEMultipart(); msg['Subject'] = f"🚀 AI PREHĽADNÝ REPORT ({len(bets)} tipov)"; msg['From'], msg['To'] = GMAIL_USER, GMAIL_RECEIVER
    html = "<h2>🎯 AI VALUE BETS (Prečistené)</h2>"
    if alerts:
        html += "<div style='background:#e3f2fd; padding:15px; border-left:5px solid #2196f3;'><h3>🔥 SMART MONEY (CLV)</h3>"
        for a in alerts[:5]: html += f"• <b>{a['Z']}</b>: {a['T']} (Kurz klesol o {a['P']})<br>"
        html += "</div><br>"
    html += "<table border='1' style='border-collapse:collapse; width:100%; text-align:center; font-family:sans-serif;'><tr style='background:#eeeeee;'><th>Zápas</th><th>Tip</th><th>Kurz</th><th>Edge</th><th>Vklad</th><th>Exp.</th></tr>"
    for b in sorted(bets, key=lambda x: x['Edge'], reverse=True)[:35]:
        html += f"<tr><td>{b['Zápas']}</td><td><b>{b['Tip']}</b></td><td>{b['Kurz']}</td><td style='color:green;'>{b['Edge']}</td><td>{b['Vklad']}</td><td>{b['Skóre']}</td></tr>"
    msg.attach(MIMEText(html + "</table>", 'html'))
    with smtplib.SMTP('smtp.gmail.com', 587) as s:
        s.starttls(); s.login(GMAIL_USER, GMAIL_PASSWORD); s.send_message(msg)

if __name__ == "__main__":
    asyncio.run(analyzuj())

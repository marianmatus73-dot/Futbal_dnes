import asyncio
import aiohttp
import pandas as pd
import numpy as np
import io
import os
import smtplib
import logging
from scipy.stats import poisson, norm
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from fuzzywuzzy import process

# --- 1. KONFIGURÁCIA ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

API_ODDS_KEY = os.getenv('ODDS_API_KEY')
GMAIL_USER = os.getenv('GMAIL_USER')
GMAIL_PASSWORD = os.getenv('GMAIL_PASSWORD')
GMAIL_RECEIVER = os.getenv('GMAIL_RECEIVER', GMAIL_USER)
AKTUALNY_BANK = float(os.getenv('AKTUALNY_BANK', 1000))
HISTORY_FILE = "historia_tipov.csv"

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
    '🏀 NBA':              {'csv': 'NBA', 'api': 'basketball_nba', 'sport': 'basketbal', 'ha': 3.1},
    '🎾 ATP Tenis':        {'csv': 'ATP', 'api': 'tennis_atp', 'sport': 'tenis', 'ha': 0}
}

# --- 2. POMOCNÉ FUNKCIE ---

def fuzzy_match_team(name, choices):
    if choices is None or len(choices) == 0: return None
    match, score = process.extractOne(name, choices)
    return match if score >= 85 else None

async def fetch_csv(session, liga, cfg):
    try:
        now = datetime.now()
        if cfg['sport'] == 'futbal':
            sez = f"{now.strftime('%y')}{(now.year + 1) % 100:02d}" if now.month >= 8 else f"{(now.year - 1) % 100:02d}{now.strftime('%y')}"
            url = f"https://www.football-data.co.uk/mmz4281/{sez}/{cfg['csv']}.csv"
        elif cfg['sport'] == 'basketbal':
            url = "https://raw.githubusercontent.com/alexno62/NBA-Data/master/nba_games_stats.csv"
        elif cfg['sport'] == 'tenis':
            url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2025.csv"
        elif cfg['csv'] == 'NHL':
            url = f"https://raw.githubusercontent.com/martineon/nhl-historical-data/master/data/nhl_results_2025.csv"
        
        async with session.get(url, timeout=15) as r:
            return liga, await r.read() if r.status == 200 else None
    except: return liga, None

def update_history_results(raw_content, liga_name, sport):
    if not os.path.isfile(HISTORY_FILE): return ""
    try:
        df = pd.read_csv(io.StringIO(raw_content.decode('utf-8', errors='ignore')))
        historia = pd.read_csv(HISTORY_FILE)
        if 'Vysledok' not in historia.columns: historia['Vysledok'] = None
        
        mask = (historia['Liga'] == liga_name) & (historia['Vysledok'].isna())
        wins, losses = 0, 0

        for idx, row in historia[mask].iterrows():
            zapas_str = row['Zápas']
            p1, p2 = zapas_str.split(' vs ')
            
            if sport == 'tenis':
                # Tenis hľadáme podľa víťaza a porazeného
                res = df[(df['winner_name'].str.contains(p1, case=False, na=False) & df['loser_name'].str.contains(p2, case=False, na=False)) |
                         (df['winner_name'].str.contains(p2, case=False, na=False) & df['loser_name'].str.contains(p1, case=False, na=False))].iloc[-1:]
                if not res.empty:
                    real_winner = res.iloc[0]['winner_name']
                    status = "WIN" if row['Tip'].lower() in real_winner.lower() else "LOSS"
                    historia.at[idx, 'Vysledok'] = status
                    if status == "WIN": wins += 1
                    else: losses += 1
            else:
                # Futbal, Hokej, Basketbal (vyhľadávanie podľa Home/Away)
                res = df[(df.iloc[:, 0:10].astype(str).apply(lambda x: x.str.contains(p1, case=False).any(), axis=1))].iloc[-1:]
                if not res.empty:
                    # Skúšame rôzne názvy stĺpcov pre góly (FTHG/FTAG alebo home_goals/away_goals)
                    gh = res.iloc[0].get('FTHG', res.iloc[0].get('home_goals', res.iloc[0].get('TeamPoints', 0)))
                    ga = res.iloc[0].get('FTAG', res.iloc[0].get('away_goals', res.iloc[0].get('OpponentPoints', 0)))
                    
                    real_winner_label = '1' if gh > ga else ('2' if ga > gh else 'X')
                    tip_label = '1' if row['Tip'] == p1 else ('2' if row['Tip'] == p2 else 'X')
                    status = "WIN" if tip_label == real_winner_label else "LOSS"
                    historia.at[idx, 'Vysledok'] = status
                    if status == "WIN": wins += 1 
                    else: losses += 1
        
        historia.to_csv(HISTORY_FILE, index=False)
        return f"<li>{liga_name}: {wins}W - {losses}L</li>" if (wins+losses) > 0 else ""
    except Exception as e: 
        logger.error(f"Chyba pri vyhodnotení {liga_name}: {e}")
        return ""

def spracuj_stats(content, cfg):
    try:
        df = pd.read_csv(io.StringIO(content.decode('utf-8', errors='ignore')))
        sport = cfg['sport']
        
        if sport == 'tenis':
            wins = df['winner_name'].value_counts()
            losses = df['loser_name'].value_counts()
            players = set(df['winner_name']) | set(df['loser_name'])
            stats = pd.DataFrame(index=list(players))
            stats['WinRate'] = [(wins.get(p, 0) + 1) / (wins.get(p, 0) + losses.get(p, 0) + 2) for p in players]
            return stats, 0, 0
        
        if sport == 'basketbal':
            df = df.rename(columns={'Team': 'HomeTeam', 'Opponent': 'AwayTeam', 'TeamPoints': 'FTHG', 'OpponentPoints': 'FTAG', 'Date': 'Date'})
        elif sport == 'hokej':
            df = df.rename(columns={'home_team': 'HomeTeam', 'away_team': 'AwayTeam', 'home_goals': 'FTHG', 'away_goals': 'FTAG', 'date': 'Date'})
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])
        
        avg_h, avg_a = df['FTHG'].mean(), df['FTAG'].mean()
        h = df.groupby('HomeTeam').apply(lambda x: pd.Series({
            'AH': x['FTHG'].mean()/avg_h, 'DH': x['FTAG'].mean()/avg_a, 'Last': x['Date'].max()
        }), include_groups=False)
        a = df.groupby('AwayTeam').apply(lambda x: pd.Series({
            'AA': x['FTAG'].mean()/avg_a, 'DA': x['FTHG'].mean()/avg_h, 'Last': x['Date'].max()
        }), include_groups=False)
        return h.join(a, how='outer').fillna(1.0), avg_h, avg_a
    except Exception as e:
        logger.error(f"Chyba spracovania štatistík: {e}")
        return None, 0, 0

# --- 3. HLAVNÝ PROCES ---

async def analyzuj():
    async with aiohttp.ClientSession() as session:
        csv_results = await asyncio.gather(*(fetch_csv(session, l, c) for l, c in LIGY_CONFIG.items()))
        all_bets, summary_html = [], ""
        now_utc = datetime.utcnow()

        for liga, content in csv_results:
            if not content: continue
            cfg = LIGY_CONFIG[liga]
            
            # 1. Vyhodnotenie histórie
            summary_html += update_history_results(content, liga, cfg['sport'])
            
            # 2. Výpočet štatistík
            stats_out = spracuj_stats(content, cfg)
            if stats_out[0] is None: continue
            stats, avg_h, avg_a = stats_out
            
            # API volanie pre kurzy na najbližších 24h
            params = {'apiKey': API_ODDS_KEY, 'regions': 'eu', 'markets': 'h2h',
                      'commenceTimeFrom': now_utc.isoformat() + 'Z',
                      'commenceTimeTo': (now_utc + timedelta(hours=24)).isoformat() + 'Z'}

            async with session.get(f'https://api.the-odds-api.com/v4/sports/{cfg["api"]}/odds/', params=params) as r:
                if r.status != 200: continue
                matches = await r.json()

            for m in matches:
                c1, c2 = fuzzy_match_team(m['home_team'], stats.index), fuzzy_match_team(m['away_team'], stats.index)
                if not (c1 and c2): continue
                
                if cfg['sport'] == 'tenis':
                    w1, w2 = stats.at[c1, 'WinRate'], stats.at[c2, 'WinRate']
                    p = {'1': w1/(w1+w2), '2': w2/(w1+w2)}
                else:
                    lh, la = stats.at[c1,'AH']*stats.at[c2,'DA']*avg_h, stats.at[c2,'AA']*stats.at[c1,'DH']*avg_a
                    
                    if cfg['sport'] == 'futbal': lh += cfg['ha']
                    if cfg['sport'] in ['hokej', 'basketbal']:
                        if (now_utc - stats.at[c1, 'Last']).days < 2: lh *= 0.94
                        if (now_utc - stats.at[c2, 'Last']).days < 2: la *= 0.94

                    if cfg['sport'] == 'basketbal':
                        p_h = 1 - norm.cdf(0, loc=(lh - la + cfg['ha']), scale=12)
                        p = {'1': p_h, '2': 1-p_h}
                    else: # Futbal / Hokej
                        p = {'1': sum(poisson.pmf(x,lh)*poisson.pmf(y,la) for x in range(10) for y in range(x)),
                             'X': sum(poisson.pmf(x,lh)*poisson.pmf(x,la) for x in range(10)),
                             '2': sum(poisson.pmf(x,lh)*poisson.pmf(y,la) for y in range(10) for x in range(y))}
                
                for bk in m.get('bookmakers', []):
                    for mk in bk.get('markets', []):
                        for out in mk['outcomes']:
                            label = '1' if out['name']==m['home_team'] else ('2' if out['name']==m['away_team'] else 'X')
                            prob = p.get(label, 0)
                            edge = (prob * out['price']) - 1
                            
                            if 0.03 <= edge <= 0.30:
                                kelly = ((out['price'] - 1) * prob - (1 - prob)) / (out['price'] - 1)
                                risk_adj = 1 / (out['price'] ** 0.5)
                                vklad_pct = min(max(0, kelly * 0.05 * risk_adj), 0.02)
                                
                                bet = {'Datum': now_utc.strftime('%Y-%m-%d'), 'Liga': liga, 
                                       'Zápas': f"{m['home_team']} vs {m['away_team']}", 'Tip': out['name'], 
                                       'Kurz': out['price'], 'Edge': f"{round(edge*100,1)}%",
                                       'Vklad': f"{round(vklad_pct*100,2)}% ({round(vklad_pct*AKTUALNY_BANK,2)}€)",
                                       'Vysledok': None}
                                all_bets.append(bet)

        # 4. Export a Email
        if all_bets:
            df_bets = pd.DataFrame(all_bets)
            df_bets.to_csv(HISTORY_FILE, mode='a', header=not os.path.exists(HISTORY_FILE), index=False)
            
            top = sorted(all_bets, key=lambda x: float(x['Edge'].replace('%','')), reverse=True)[:7]
            html = f"<h3>📈 Bilancia výsledkov:</h3><ul>{summary_html if summary_html else 'Čakáme na výsledky...'}</ul>"
            html += "<h3>✅ Nové AI Tipy na dnes:</h3><table border='1' style='width:100%; border-collapse:collapse;'>"
            html += "<tr style='background:#eee;'><th>Liga</th><th>Zápas</th><th>Tip</th><th>Kurz</th><th>Edge</th><th>Vklad</th></tr>"
            for b in top:
                html += f"<tr><td>{b['Liga']}</td><td>{b['Zápas']}</td><td>{b['Tip']}</td><td>{b['Kurz']}</td><td style='color:green;'>{b['Edge']}</td><td>{b['Vklad']}</td></tr>"
            
            msg = MIMEMultipart(); msg['Subject'] = f"📊 AI Betting Report V4.1 - {now_utc.strftime('%d.%m.')}"; msg['From'] = GMAIL_USER; msg['To'] = GMAIL_RECEIVER
            msg.attach(MIMEText(html + "</table>", 'html'))
            with smtplib.SMTP('smtp.gmail.com', 587) as s:
                s.starttls(); s.login(GMAIL_USER, GMAIL_PASSWORD); s.send_message(msg)

if __name__ == "__main__":
    asyncio.run(analyzuj())

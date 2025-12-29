import asyncio
import aiohttp
import pandas as pd
import numpy as np
import io
import os
import smtplib
import logging
from scipy.stats import poisson, norm
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from fuzzywuzzy import process

# --- 1. KONFIGURÁCIA A LOGOVANIE ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

API_ODDS_KEY = os.getenv('ODDS_API_KEY')
GMAIL_USER = os.getenv('GMAIL_USER')
GMAIL_PASSWORD = os.getenv('GMAIL_PASSWORD')
GMAIL_RECEIVER = os.getenv('GMAIL_RECEIVER', GMAIL_USER)
AKTUALNY_BANK = float(os.getenv('AKTUALNY_BANK', 1000))

LIGY_CONFIG = {
    '⚽ Premier League':   {'csv': 'E0',  'api': 'soccer_epl', 'sport': 'futbal'},
    '⚽ La Liga':          {'csv': 'SP1', 'api': 'soccer_spain_la_liga', 'sport': 'futbal'},
    '⚽ Bundesliga':       {'csv': 'D1',  'api': 'soccer_germany_bundesliga', 'sport': 'futbal'},
    '⚽ Serie A':          {'csv': 'I1',  'api': 'soccer_italy_serie_a', 'sport': 'futbal'},
    '⚽ Ligue 1':          {'csv': 'F1',  'api': 'soccer_france_ligue_one', 'sport': 'futbal'},
    '⚽ Eredivisie':       {'csv': 'N1',  'api': 'soccer_netherlands_eredivisie', 'sport': 'futbal'},
    '⚽ Liga Portugal':    {'csv': 'P1',  'api': 'soccer_portugal_primeira_liga', 'sport': 'futbal'},
    '⚽ Süper Lig':        {'csv': 'T1',  'api': 'soccer_turkey_super_league', 'sport': 'futbal'},
    '🏒 NHL':              {'csv': 'NHL', 'api': 'icehockey_nhl', 'sport': 'hokej'},
    '🏀 NBA':              {'csv': 'NBA', 'api': 'basketball_nba', 'sport': 'basketbal'},
    '🎾 ATP Tenis':        {'csv': 'ATP', 'api': 'tennis_atp', 'sport': 'tenis'}
}

KELLY_FRACTION = 0.1
MAX_BANK_PCT = 0.02

# --- 2. POMOCNÉ FUNKCIE ---

def fuzzy_match_team(name, choices):
    if choices is None or len(choices) == 0: return None
    # Zvýšená hranica na 85, aby sa nepomiešali tímy (napr. Paris FC vs PSG)
    match, score = process.extractOne(name, choices)
    return match if score >= 85 else None

async def fetch_csv(session, liga, cfg):
    try:
        url = ""
        now = datetime.now()
        if cfg['sport'] == 'futbal':
            sez = f"{now.strftime('%y')}{(now.year + 1) % 100:02d}" if now.month >= 8 else f"{(now.year - 1) % 100:02d}{now.strftime('%y')}"
            url = f"https://www.football-data.co.uk/mmz4281/{sez}/{cfg['csv']}.csv"
        elif cfg['sport'] == 'basketbal':
            url = "https://raw.githubusercontent.com/alexno62/NBA-Data/master/nba_games_stats.csv"
        elif cfg['sport'] == 'tenis':
            url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{now.year}.csv"
        elif cfg['csv'] == 'NHL':
            url = f"https://raw.githubusercontent.com/martineon/nhl-historical-data/master/data/nhl_results_{now.year}.csv"
        
        async with session.get(url, timeout=12) as r:
            return liga, await r.read() if r.status == 200 else None
    except: return liga, None

def spracuj_stats(content, sport):
    try:
        df = pd.read_csv(io.StringIO(content.decode('utf-8', errors='ignore')))
        if sport == 'tenis':
            wins = df['winner_name'].value_counts()
            losses = df['loser_name'].value_counts()
            players = set(df['winner_name']) | set(df['loser_name'])
            stats = pd.DataFrame(index=list(players))
            stats['WinRate'] = [wins.get(p, 0) / (wins.get(p, 0) + losses.get(p, 0) + 1) for p in players]
            return stats, 0, 0
        
        if sport == 'basketbal':
            df = df.rename(columns={'Team': 'HomeTeam', 'Opponent': 'AwayTeam', 'TeamPoints': 'FTHG', 'OpponentPoints': 'FTAG'})
        elif sport == 'hokej':
            df = df.rename(columns={'home_team': 'HomeTeam', 'away_team': 'AwayTeam', 'home_goals': 'FTHG', 'away_goals': 'FTAG'})
        
        df['W'] = np.linspace(0.8, 1.2, len(df))
        avg_h, avg_a = (df['FTHG']*df['W']).sum()/df['W'].sum(), (df['FTAG']*df['W']).sum()/df['W'].sum()
        h = df.groupby('HomeTeam').apply(lambda x: pd.Series({'AH': (x['FTHG']*x['W']).mean()/avg_h, 'DH': (x['FTAG']*x['W']).mean()/avg_a}), include_groups=False)
        a = df.groupby('AwayTeam').apply(lambda x: pd.Series({'AA': (x['FTAG']*x['W']).mean()/avg_a, 'DA': (x['FTHG']*x['W']).mean()/avg_h}), include_groups=False)
        return h.join(a, how='outer').fillna(1.0), avg_h, avg_a
    except: return None, 0, 0

def odosli_email(text):
    msg = MIMEMultipart()
    msg['Subject'] = f"🎯 AI Betting Report - {datetime.now().strftime('%d.%m. %H:%M')}"
    msg['From'] = GMAIL_USER
    msg['To'] = GMAIL_RECEIVER
    msg.attach(MIMEText(text, 'html'))
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.starttls()
            s.login(GMAIL_USER, GMAIL_PASSWORD)
            s.send_message(msg)
        logger.info("✅ Email úspešne odoslaný.")
    except Exception as e: logger.error(f"❌ Email zlyhal: {e}")

# --- 3. HLAVNÝ PROCES ---

async def analyzuj():
    async with aiohttp.ClientSession() as session:
        csv_results = await asyncio.gather(*(fetch_csv(session, l, c) for l, c in LIGY_CONFIG.items()))
        all_bets = []
        processed_match_ids = set()

        for liga, content in csv_results:
            if not content: continue
            cfg = LIGY_CONFIG[liga]
            stats, avg_h, avg_a = spracuj_stats(content, cfg['sport'])
            
            async with session.get(f'https://api.the-odds-api.com/v4/sports/{cfg["api"]}/odds/', params={'apiKey': API_ODDS_KEY, 'regions': 'eu', 'markets': 'h2h'}) as r:
                if r.status != 200: continue
                matches = await r.json()

            for m in matches:
                # Unikátny identifikátor zápasu pre elimináciu duplicít
                m_id = f"{m['home_team']}_{m['away_team']}_{liga}"
                if m_id in processed_match_ids: continue
                
                c1, c2 = fuzzy_match_team(m['home_team'], stats.index), fuzzy_match_team(m['away_team'], stats.index)
                if c1 and c2:
                    if cfg['sport'] == 'tenis':
                        w1, w2 = stats.at[c1, 'WinRate'], stats.at[c2, 'WinRate']
                        p = {'1': w1/(w1+w2), '2': w2/(w1+w2)}
                    elif cfg['sport'] == 'basketbal':
                        lh, la = stats.at[c1,'AH']*stats.at[c2,'DA']*avg_h, stats.at[c2,'AA']*stats.at[c1,'DH']*avg_a
                        p_h = 1 - norm.cdf(0, loc=(lh - la), scale=12)
                        p = {'1': p_h, '2': 1-p_h}
                    else:
                        lh, la = stats.at[c1,'AH']*stats.at[c2,'DA']*avg_h, stats.at[c2,'AA']*stats.at[c1,'DH']*avg_a
                        p = {'1': sum(poisson.pmf(x,lh)*poisson.pmf(y,la) for x in range(10) for y in range(x)),
                             'X': sum(poisson.pmf(x,lh)*poisson.pmf(x,la) for x in range(10)),
                             '2': sum(poisson.pmf(x,lh)*poisson.pmf(y,la) for y in range(10) for x in range(y))}
                    
                    # Najlepšia príležitosť v rámci zápasu
                    match_bets = []
                    for bk in m.get('bookmakers', []):
                        for mk in bk.get('markets', []):
                            if mk['key'] == 'h2h':
                                for out in mk['outcomes']:
                                    label = '1' if out['name']==m['home_team'] else ('2' if out['name']==m['away_team'] else 'X')
                                    prob = p.get(label, 0)
                                    edge = (prob * out['price']) - 1
                                    
                                    # Kelly Criterion pre vklad
                                    f_star = ((out['price'] - 1) * prob - (1 - prob)) / (out['price'] - 1)
                                    vklad_pct = min(max(0, f_star * KELLY_FRACTION), MAX_BANK_PCT)
                                    
                                    # Filter proti nereálnym Edge (nad 100%)
                                    if 0.02 < edge < 1.0:
                                        match_bets.append({
                                            'Liga': liga, 'Zápas': f"{m['home_team']} vs {m['away_team']}", 
                                            'Tip': out['name'], 'Kurz': out['price'], 'Edge': edge,
                                            'Vklad': f"{round(vklad_pct*100,2)}% ({round(vklad_pct * AKTUALNY_BANK, 2)}€)"
                                        })
                    
                    if match_bets:
                        # Pridaj len ten úplne najlepší tip z tohto zápasu
                        all_bets.append(max(match_bets, key=lambda x: x['Edge']))
                        processed_match_ids.add(m_id)

        if all_bets:
            top = sorted(all_bets, key=lambda x: x['Edge'], reverse=True)[:3]
            html = "<h2>Top 3 AI Príležitosti</h2><table border='1' style='border-collapse:collapse; width:100%; text-align:center;'>"
            html += "<tr style='background:#2c3e50; color:white;'><th>Liga</th><th>Zápas</th><th>Tip</th><th>Kurz</th><th>Edge</th><th>Vklad</th></tr>"
            for b in top:
                edge_val = round(b['Edge']*100,1)
                color = "green" if edge_val < 40 else "orange"
                html += f"<tr><td>{b['Liga']}</td><td>{b['Zápas']}</td><td>{b['Tip']}</td><td>{b['Kurz']}</td>"
                html += f"<td style='color:{color}; font-weight:bold;'>{edge_val}%</td><td>{b['Vklad']}</td></tr>"
            odosli_email(html + "</table>")
        else:
            logger.warning("Nenašli sa žiadne bezpečné tipy.")

if __name__ == "__main__":
    asyncio.run(analyzuj())

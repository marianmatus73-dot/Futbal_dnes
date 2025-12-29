import asyncio
import aiohttp
import pandas as pd
import numpy as np
import io
import os
import smtplib
import pytz
import logging
from scipy.stats import poisson, norm
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from fuzzywuzzy import process

# --- 1. KONFIGURÁCIA ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
env = {k: os.getenv(k) for k in ['ODDS_API_KEY', 'GMAIL_USER', 'GMAIL_PASSWORD']}
API_ODDS_KEY = env['ODDS_API_KEY']
GMAIL_USER = env['GMAIL_USER']
GMAIL_PASSWORD = env['GMAIL_PASSWORD']
GMAIL_RECEIVER = os.getenv('GMAIL_RECEIVER', GMAIL_USER)
AKTUALNY_BANK = float(os.getenv('AKTUALNY_BANK', 1000))

# KOMPLETNÝ ZOZNAM LÍG
LIGY_CONFIG = {
    '⚽ Premier League':   {'csv': 'E0',  'api': 'soccer_epl', 'sport': 'futbal'},
    '⚽ Championship':     {'csv': 'E1',  'api': 'soccer_efl_champ', 'sport': 'futbal'},
    '⚽ La Liga':          {'csv': 'SP1', 'api': 'soccer_spain_la_liga', 'sport': 'futbal'},
    '⚽ Bundesliga':       {'csv': 'D1',  'api': 'soccer_germany_bundesliga', 'sport': 'futbal'},
    '⚽ Serie A':          {'csv': 'I1',  'api': 'soccer_italy_serie_a', 'sport': 'futbal'},
    '⚽ Ligue 1':          {'csv': 'F1',  'api': 'soccer_france_ligue_one', 'sport': 'futbal'},
    '⚽ Eredivisie':       {'csv': 'N1',  'api': 'soccer_netherlands_eredivisie', 'sport': 'futbal'},
    '⚽ Liga Portugal':    {'csv': 'P1',  'api': 'soccer_portugal_primeira_liga', 'sport': 'futbal'},
    '⚽ Jupiler League':   {'csv': 'B1',  'api': 'soccer_belgium_first_division', 'sport': 'futbal'},
    '⚽ Süper Lig':        {'csv': 'T1',  'api': 'soccer_turkey_super_league', 'sport': 'futbal'},
    '🏒 NHL':              {'csv': 'NHL', 'api': 'icehockey_nhl', 'sport': 'hokej'},
    '🏀 NBA':              {'csv': 'NBA', 'api': 'basketball_nba', 'sport': 'basketbal'},
    '🎾 ATP Tenis':        {'csv': 'ATP', 'api': 'tennis_atp', 'sport': 'tenis'}
}

KELLY_FRACTION = 0.15
MAX_BANK_PCT = 0.02
MIN_VALUE_EDGE = 0.02

# --- 2. POMOCNÉ FUNKCIE ---

def vypocitaj_kelly(pravd, kurz):
    if kurz <= 1 or pravd <= 0: return 0, 0
    f_star = ((kurz - 1) * pravd - (1 - pravd)) / (kurz - 1)
    vklad_pct = min(max(0, f_star * KELLY_FRACTION), MAX_BANK_PCT)
    return round(vklad_pct * 100, 2), round(vklad_pct * AKTUALNY_BANK, 2)

def fuzzy_match_team(name, choices):
    if choices is None or len(choices) == 0: return None
    m, s = process.extractOne(name, choices)
    return m if s >= 75 else None

# --- 3. ASYNCHRÓNNE SŤAHOVANIE DÁT ---

async def fetch_csv(session, liga, cfg):
    try:
        url = ""
        if cfg['sport'] == 'futbal':
            now = datetime.now()
            sez_str = f"{now.strftime('%y')}{(now.year + 1) % 100:02d}" if now.month >= 8 else f"{(now.year - 1) % 100:02d}{now.strftime('%y')}"
            url = f"https://www.football-data.co.uk/mmz4281/{sez_str}/{cfg['csv']}.csv"
        elif cfg['sport'] == 'basketbal':
            url = "https://raw.githubusercontent.com/alexno62/NBA-Data/master/nba_games_stats.csv"
        elif cfg['sport'] == 'tenis':
            url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{datetime.now().year}.csv"
        elif cfg['csv'] == 'NHL':
            sez = datetime.now().year if datetime.now().month < 9 else datetime.now().year + 1
            url = f"https://raw.githubusercontent.com/martineon/nhl-historical-data/master/data/nhl_results_{sez}.csv"

        async with session.get(url, timeout=15) as resp:
            if resp.status == 200:
                content = await resp.read()
                return liga, content
    except:
        logger.error(f"Nepodarilo sa stiahnuť dáta pre {liga}")
    return liga, None

# --- 4. VÝPOČTY ŠTATISTÍK ---

def spracuj_stats(liga, content, sport):
    try:
        df = pd.read_csv(io.StringIO(content.decode('utf-8', errors='ignore')))
        if sport == 'tenis':
            wins = df['winner_name'].value_counts()
            losses = df['loser_name'].value_counts()
            players = set(df['winner_name']) | set(df['loser_name'])
            stats = pd.DataFrame(index=list(players))
            stats['WinRate'] = [wins.get(p, 0) / (wins.get(p, 0) + losses.get(p, 0) + 1) for p in players]
            return stats, 0, 0
        
        # Premenovanie stĺpcov pre zjednotenie
        if sport == 'basketbal':
            df = df.rename(columns={'Team': 'HomeTeam', 'Opponent': 'AwayTeam', 'TeamPoints': 'FTHG', 'OpponentPoints': 'FTAG'})
        elif sport == 'hokej':
            df = df.rename(columns={'home_team': 'HomeTeam', 'away_team': 'AwayTeam', 'home_goals': 'FTHG', 'away_goals': 'FTAG'})

        df['Weight'] = np.linspace(0.8, 1.2, len(df))
        avg_h = (df['FTHG'] * df['Weight']).sum() / df['Weight'].sum()
        avg_a = (df['FTAG'] * df['Weight']).sum() / df['Weight'].sum()
        
        h_stats = df.groupby('HomeTeam').apply(lambda x: pd.Series({'Att_H': (x['FTHG']*x['Weight']).mean()/avg_h, 'Def_H': (x['FTAG']*x['Weight']).mean()/avg_a}))
        a_stats = df.groupby('AwayTeam').apply(lambda x: pd.Series({'Att_A': (x['FTAG']*x['Weight']).mean()/avg_a, 'Def_A': (x['FTHG']*x['Weight']).mean()/avg_h}))
        return h_stats.join(a_stats, how='outer').fillna(1.0), avg_h, avg_a
    except:
        return None, 0, 0

# --- 5. HLAVNÁ LOGIKA ---

async def analyzuj():
    async with aiohttp.ClientSession() as session:
        # Sťahovanie CSV dát paralelne
        csv_tasks = [fetch_csv(session, liga, cfg) for liga, cfg in LIGY_CONFIG.items()]
        csv_results = await asyncio.gather(*csv_tasks)
        
        all_bets = []
        
        for liga, content in csv_results:
            if not content: continue
            cfg = LIGY_CONFIG[liga]
            stats, avg_h, avg_a = spracuj_stats(liga, content, cfg['sport'])
            if stats is None: continue

            # Získanie kurzov
            odds_url = f'https://api.the-odds-api.com/v4/sports/{cfg["api"]}/odds/'
            async with session.get(odds_url, params={'apiKey': API_ODDS_KEY, 'regions': 'eu', 'markets': 'h2h'}) as resp:
                matches = await resp.json() if resp.status == 200 else []

            for m in matches:
                t1, t2 = m['home_team'], m['away_team']
                c1, c2 = fuzzy_match_team(t1, stats.index), fuzzy_match_team(t2, stats.index)
                if not c1 or not c2: continue

                # Predikcia
                if cfg['sport'] == 'tenis':
                    w1, w2 = stats.at[c1, 'WinRate'], stats.at[c2, 'WinRate']
                    p = {'1': w1/(w1+w2), '2': w2/(w1+w2)}
                elif cfg['sport'] == 'basketbal':
                    lh = stats.at[c1, 'Att_H'] * stats.at[c2, 'Def_A'] * avg_h
                    la = stats.at[c2, 'Att_A'] * stats.at[c1, 'Def_H'] * avg_a
                    prob_h = 1 - norm.cdf(0, loc=(lh - la), scale=12)
                    p = {'1': prob_h, '2': 1-prob_h}
                else: # Futbal + Hokej
                    lh = stats.at[c1, 'Att_H'] * stats.at[c2, 'Def_A'] * avg_h
                    la = stats.at[c2, 'Att_A'] * stats.at[c1, 'Def_H'] * avg_a
                    p = {'1': 0, 'X': 0, '2': 0}
                    for x in range(10):
                        for y in range(10):
                            prob = poisson.pmf(x, lh) * poisson.pmf(y, la)
                            if x > y: p['1'] += prob
                            elif x == y: p['X'] += prob
                            else: p['2'] += prob

                # Kontrola hodnoty u bookmakerov
                for bk in m.get('bookmakers', []):
                    h2h = next((mk for mk in bk['markets'] if mk['key'] == 'h2h'), None)
                    if h2h:
                        for out in h2h['outcomes']:
                            key = '1' if out['name'] == t1 else ('2' if out['name'] == t2 else 'X')
                            prob = p.get(key, 0)
                            edge = (prob * out['price']) - 1
                            if MIN_VALUE_EDGE < edge < 0.4:
                                pct, suma = vypocitaj_kelly(prob, out['price'])
                                if pct > 0:
                                    all_bets.append({
                                        'Liga': liga, 'Zápas': f"{t1}-{t2}", 'Tip': f"{out['name']}",
                                        'Kurz': out['price'], 'Edge': f"{round(edge*100,1)}%", 'Vklad': f"{pct}% ({suma}€)"
                                    })

        # Zoradenie a odoslanie (len top 5 najlepších)
        top_bets = sorted(all_bets, key=lambda x: float(x['Edge'].replace('%','')), reverse=True)[:5]
        # Tu zavoláš funkciu odosli_email_vylepseny z predchádzajúceho kódu...
        print(f"Nájdených {len(top_bets)} tipov.")

if __name__ == "__main__":
    asyncio.run(analyzuj())

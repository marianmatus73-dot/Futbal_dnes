import requests
import pandas as pd
import smtplib
import os
import io
import time
from scipy.stats import poisson
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from fuzzywuzzy import process

# --- 1. NASTAVENIA ---
load_dotenv()

API_ODDS_KEY = os.getenv('ODDS_API_KEY')
GMAIL_USER = os.getenv('GMAIL_USER')
GMAIL_PASSWORD = os.getenv('GMAIL_PASSWORD')
GMAIL_RECEIVER = os.getenv('GMAIL_RECEIVER', GMAIL_USER)

LIGY_CONFIG = {
    '🇬🇧 Premier League':   {'csv': 'E0',  'api': 'soccer_epl'},
    '🇬🇧 Championship':     {'csv': 'E1',  'api': 'soccer_efl_champ'},
    '🇪🇸 La Liga':          {'csv': 'SP1', 'api': 'soccer_spain_la_liga'},
    '🇩🇪 Bundesliga':       {'csv': 'D1',  'api': 'soccer_germany_bundesliga'},
    '🇮🇹 Serie A':          {'csv': 'I1',  'api': 'soccer_italy_serie_a'},
    '🇫🇷 Ligue 1':          {'csv': 'F1',  'api': 'soccer_france_ligue_one'},
    '🇳🇱 Eredivisie':       {'csv': 'N1',  'api': 'soccer_netherlands_eredivisie'},
    '🇵🇹 Liga Portugal':    {'csv': 'P1',  'api': 'soccer_portugal_primeira_liga'},
    '🇧🇪 Jupiler League':   {'csv': 'B1',  'api': 'soccer_belgium_first_division'},
    '🇹🇷 Süper Lig':        {'csv': 'T1',  'api': 'soccer_turkey_super_league'},
    '🇺🇸 NHL':              {'csv': 'NHL', 'api': 'icehockey_nhl'}
}

MIN_VALUE_EDGE = 0.05
KELLY_FRACTION = 0.2
MAX_BANK_PCT = 0.02

# --- 2. POMOCNÉ FUNKCIE ---

def ziskaj_nhl_rok():
    now = datetime.now()
    return str(now.year + 1) if now.month >= 10 else str(now.year)

def ziskaj_futbal_sezonu():
    now = datetime.now()
    if now.month >= 8:
        return f"{now.strftime('%y')}{(now.year + 1) % 100:02d}"
    return f"{(now.year - 1) % 100:02d}{now.strftime('%y')}"

AKTUALNA_SEZONA_FUTBAL = ziskaj_futbal_sezonu()
AKTUALNA_SEZONA_NHL = ziskaj_nhl_rok()

# --- 3. JADRO MODELU ---

def vypocitaj_kelly(pravdepodobnost, kurz):
    if kurz <= 1 or pravdepodobnost <= 0: return 0
    b = kurz - 1
    f_star = (b * pravdepodobnost - (1 - pravdepodobnost)) / b
    return round(min(max(0, f_star * KELLY_FRACTION), MAX_BANK_PCT) * 100, 2)

def vypocitaj_silu_timov(df):
    if df.empty or len(df) < 10: return None, 0, 0
    avg_h, avg_a = df['FTHG'].mean(), df['FTAG'].mean()
    h_stats = df.groupby('HomeTeam')[['FTHG', 'FTAG']].mean()
    a_stats = df.groupby('AwayTeam')[['FTAG', 'FTHG']].mean()
    teams = pd.DataFrame(index=h_stats.index)
    teams['Att_H'] = h_stats['FTHG'] / avg_h
    teams['Def_H'] = h_stats['FTAG'] / avg_a
    teams_away = pd.DataFrame(index=a_stats.index)
    teams_away['Att_A'] = a_stats['FTAG'] / avg_a
    teams_away['Def_A'] = a_stats['FTHG'] / avg_h
    return teams.join(teams_away, how='outer').fillna(1.0), avg_h, avg_a

def predikuj_vsetko(home, away, stats, avg_h, avg_a, sport='futbal'):
    try:
        lamb_h = stats.at[home, 'Att_H'] * stats.at[away, 'Def_A'] * avg_h
        lamb_a = stats.at[away, 'Att_A'] * stats.at[home, 'Def_H'] * avg_a
    except KeyError: return None

    res = {'1': 0, 'X': 0, '2': 0, 'over': 0, 'under': 0}
    limit = 2.5 if sport == 'futbal' else 5.5
    max_g = 12 if sport == 'nhl' else 8

    for x in range(max_g):
        for y in range(max_g):
            p = poisson.pmf(x, lamb_h) * poisson.pmf(y, lamb_a)
            if x > y: res['1'] += p
            elif x == y: res['X'] += p
            else: res['2'] += p
            if (x + y) > limit: res['over'] += p
            else: res['under'] += p
    return res

# --- 4. DÁTA A API ---

def stiahni_csv_data(liga_kod):
    url = f"https://raw.githubusercontent.com/lbenz730/NHL_Draft_Analysis/master/data/nhl_scores_{AKTUALNA_SEZONA_NHL}.csv" if liga_kod == 'NHL' else f"https://www.football-data.co.uk/mmz4281/{AKTUALNA_SEZONA_FUTBAL}/{liga_kod}.csv"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200: return pd.DataFrame()
        df = pd.read_csv(io.StringIO(r.content.decode('utf-8', errors='ignore')))
        if liga_kod == 'NHL':
            df = df.rename(columns={'home_team': 'HomeTeam', 'away_team': 'AwayTeam', 'home_goals': 'FTHG', 'away_goals': 'FTAG'})
        return df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].dropna()
    except: return pd.DataFrame()

def ziskaj_kurzy(sport_key):
    url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds/'
    params = {'apiKey': API_ODDS_KEY, 'regions': 'eu', 'markets': 'h2h,totals', 'oddsFormat': 'decimal'}
    try:
        r = requests.get(url, params=params, timeout=10)
        return r.json() if r.status_code == 200 else []
    except: return []

def fuzzy_match_team(api_name, csv_teams):
    match, score = process.extractOne(api_name, csv_teams)
    return match if score >= 75 else None

# --- 5. HLAVNÝ PROCES ---

def spustit_analyzu():
    all_potential_bets = []
    print(f"🔄 Analýza spustená: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    
    for liga, cfg in LIGY_CONFIG.items():
        df = stiahni_csv_data(cfg['csv'])
        stats, avg_h, avg_a = vypocitaj_silu_timov(df)
        if stats is None: continue

        matches = ziskaj_kurzy(cfg['api'])
        sport_type = 'nhl' if 'NHL' in liga else 'futbal'
        goly_limit = 5.5 if sport_type == 'nhl' else 2.5

        for m in matches:
            api_h, api_a = m['home_team'], m['away_team']
            start_time = datetime.fromisoformat(m['commence_time'].replace('Z', '')).strftime('%H:%M')
            csv_h, csv_a = fuzzy_match_team(api_h, stats.index), fuzzy_match_team(api_a, stats.index)
            if not csv_h or not csv_a: continue

            probs = predikuj_vsetko(csv_h, csv_a, stats, avg_h, avg_a, sport_type)
            if not probs: continue

            for bookie in m['bookmakers']:
                h2h = next((mk for mk in bookie['markets'] if mk['key'] == 'h2h'), None)
                if h2h:
                    for out in h2h['outcomes']:
                        p = probs['1'] if out['name'] == api_h else (probs['2'] if out['name'] == api_a else probs['X'])
                        edge = (p * out['price']) - 1
                        if edge > 0:
                            all_potential_bets.append({
                                'Čas': start_time, 'Liga': liga, 'Zápas': f"{api_h} vs {api_a}", 
                                'Typ': '⚖️ 1X2', 'Tip': out['name'], 'Kurz': out['price'], 
                                'Edge': edge, 'Pravd': p
                            })

                totals = next((mk for mk in bookie['markets'] if mk['key'] == 'totals'), None)
                if totals:
                    for out in totals['outcomes']:
                        if out.get('point') == goly_limit:
                            p = probs['over'] if out['name'].lower() == 'over' else probs['under']
                            edge = (p * out['price']) - 1
                            if edge > 0:
                                all_potential_bets.append({
                                    'Čas': start_time, 'Liga': liga, 'Zápas': f"{api_h} vs {api_a}", 
                                    'Tip': f"{out['name']} {goly_limit}", 'Typ': '⚽ Góly', 'Kurz': out['price'], 
                                    'Edge': edge, 'Pravd': p
                                })

    if all_potential_bets:
        df_all = pd.DataFrame(all_potential_bets).sort_values(by='Edge', ascending=False)
        top_bets = df_all[df_all['Edge'] >= MIN_VALUE_EDGE]
        # Garancia aspoň 3 tipov
        if len(top_bets) < 3: top_bets = df_all.head(3)

        final_data = []
        for _, r in top_bets.iterrows():
            final_data.append({
                'Čas': r['Čas'], 'Liga': r['Liga'], 'Zápas': r['Zápas'], 'Typ': r['Typ'], 
                'Tip': r['Tip'], 'Model': f"{r['Pravd']:.1%}", 'Kurz': r['Kurz'], 
                'Edge': f"{r['Edge']:.1%}", 'Vklad': f"<b>{vypocitaj_kelly(r['Pravd'], r['Kurz'])}%</b>"
            })
        odosli_email(final_data)
    else:
        # Odoslať email o prázdnom dni
        odosli_email([], ziadne_tipy=True)

def odosli_email(data, ziadne_tipy=False):
    style = """<style>
        table {border-collapse: collapse; width: 100%; font-family: sans-serif;}
        th {background-color: #1a237e; color: white; padding: 12px; text-align: center;}
        td {padding: 10px; border-bottom: 1px solid #ddd; text-align: center;}
        tr:nth-child(even) {background-color: #f8f9fa;}
        .type-goals {color: #e65100; font-weight: bold;}
        .type-1x2 {color: #0277bd; font-weight: bold;}
    </style>"""

    if ziadne_tipy:
        html = "<html><body><h2>🎯 AI Stávkový Report</h2><p>Dnes nie sú žiadne výhodné príležitosti.</p></body></html>"
        subject = f"💤 AI Report: Žiadne tipy ({datetime.now().strftime('%d.%m')})"
    else:
        df = pd.DataFrame(data)
        df['Typ'] = df['Typ'].apply(lambda x: f'<span class="{"type-goals" if "⚽" in x else "type-1x2"}">{x}</span>')
        html = f"<html><body><h2 style='color: #1a237e;'>🎯 AI Stávkový Report</h2>{style}{df.to_html(index=False, escape=False)}</body></html>"
        subject = f"🚀 TOP AI Tipy: {len(data)} šancí ({datetime.now().strftime('%d.%m')})"
    
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = GMAIL_USER
    msg['To'] = GMAIL_RECEIVER
    msg.attach(MIMEText(html, 'html'))
    
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.starttls(); s.login(GMAIL_USER, GMAIL_PASSWORD); s.send_message(msg)
        print("📧 Email odoslaný!")
    except Exception as e: print(f"Chyba: {e}")

if __name__ == "__main__":
    spustit_analyzu()

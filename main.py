import requests
import pandas as pd
import smtplib
import os
import io
import pytz
from scipy.stats import poisson
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
from fuzzywuzzy import process

# --- 1. NASTAVENIA & ENV ---
load_dotenv()

def get_required_env(keys):
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Chýbajú env premenné: {', '.join(missing)}")
    return {k: os.getenv(k) for k in keys}

env = get_required_env(['ODDS_API_KEY', 'GMAIL_USER', 'GMAIL_PASSWORD'])
API_ODDS_KEY = env['ODDS_API_KEY']
GMAIL_USER = env['GMAIL_USER']
GMAIL_PASSWORD = env['GMAIL_PASSWORD']
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

MIN_VALUE_EDGE = 0.01  # Znížené, aby sme mali z čoho vyberať Top 3
KELLY_FRACTION = 0.2
MAX_BANK_PCT = 0.02

# --- 2. POMOCNÉ FUNKCIE ---

def get_local_time(utc_str):
    try:
        utc_dt = datetime.fromisoformat(utc_str.replace('Z', '')).replace(tzinfo=pytz.utc)
        local_tz = pytz.timezone('Europe/Bratislava')
        return utc_dt.astimezone(local_tz).strftime('%H:%M')
    except:
        return '??:??'

def vypocitaj_kelly(pravdepodobnost, kurz):
    if kurz <= 1 or pravdepodobnost <= 0: return 0
    b = kurz - 1
    f_star = (b * pravdepodobnost - (1 - pravdepodobnost)) / b
    frac = min(max(0, f_star * KELLY_FRACTION), MAX_BANK_PCT)
    return round(frac * 100, 2)

# --- 3. DÁTA A MODEL ---

def stiahni_csv_data(liga_kod):
    now = datetime.now()
    if liga_kod == 'NHL':
        sezona = str(now.year) if now.month < 9 else str(now.year + 1)
        url = f"https://raw.githubusercontent.com/martineon/nhl-historical-data/master/data/nhl_results_{sezona}.csv"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200: return pd.DataFrame()
            df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
            df = df.rename(columns={'home_team': 'HomeTeam', 'away_team': 'AwayTeam', 'home_goals': 'FTHG', 'away_goals': 'FTAG'})
            return df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].dropna()
        except: return pd.DataFrame()
    else:
        sez_str = f"{now.strftime('%y')}{(now.year + 1) % 100:02d}" if now.month >= 8 else f"{(now.year - 1) % 100:02d}{now.strftime('%y')}"
        url = f"https://www.football-data.co.uk/mmz4281/{sez_str}/{liga_kod}.csv"
        try:
            r = requests.get(url, timeout=10)
            df = pd.read_csv(io.StringIO(r.content.decode('utf-8', errors='ignore')))
            return df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].dropna()
        except: return pd.DataFrame()

def vypocitaj_silu_timov(df):
    if df.empty or len(df) < 10: return None, 0, 0
    df = df.copy()
    split_point = int(len(df) * 0.75)
    df['Weight'] = 1.0
    df.iloc[split_point:, df.columns.get_loc('Weight')] = 1.3
    avg_h = (df['FTHG'] * df['Weight']).sum() / df['Weight'].sum()
    avg_a = (df['FTAG'] * df['Weight']).sum() / df['Weight'].sum()
    h_stats = df.groupby('HomeTeam').apply(lambda x: pd.Series({
        'Att_H': ((x['FTHG'] * x['Weight']).sum() / x['Weight'].sum()) / avg_h,
        'Def_H': ((x['FTAG'] * x['Weight']).sum() / x['Weight'].sum()) / avg_a
    }))
    a_stats = df.groupby('AwayTeam').apply(lambda x: pd.Series({
        'Att_A': ((x['FTAG'] * x['Weight']).sum() / x['Weight'].sum()) / avg_a,
        'Def_A': ((x['FTHG'] * x['Weight']).sum() / x['Weight'].sum()) / avg_h
    }))
    return h_stats.join(a_stats, how='outer').fillna(1.0), avg_h, avg_a

def predikuj_vsetko(home, away, stats, avg_h, avg_a, sport='futbal'):
    if home not in stats.index or away not in stats.index: return None
    lamb_h = stats.at[home, 'Att_H'] * stats.at[away, 'Def_A'] * avg_h
    lamb_a = stats.at[away, 'Att_A'] * stats.at[home, 'Def_H'] * avg_a
    res = {'1': 0, 'X': 0, '2': 0, 'over': 0, 'under': 0}
    limit = 2.5 if sport == 'futbal' else 5.5
    for x in range(12):
        for y in range(12):
            p = poisson.pmf(x, lamb_h) * poisson.pmf(y, lamb_a)
            if x > y: res['1'] += p
            elif x == y: res['X'] += p
            else: res['2'] += p
            if (x + y) > limit: res['over'] += p
            else: res['under'] += p
    sum_1x2 = res['1'] + res['X'] + res['2']
    res['1'] /= sum_1x2; res['X'] /= sum_1x2; res['2'] /= sum_1x2
    if sport == 'nhl':
        res['ML1'] = res['1'] + (res['X'] * 0.51)
        res['ML2'] = res['2'] + (res['X'] * 0.49)
    return res

def ziskaj_kurzy(sport_key):
    url = f'https://api.the-odds-api.com/v4/sports/{sport_key}/odds/'
    params = {'apiKey': API_ODDS_KEY, 'regions': 'eu', 'markets': 'h2h,totals', 'oddsFormat': 'decimal'}
    try:
        r = requests.get(url, params=params, timeout=15)
        return r.json() if r.status_code == 200 else []
    except: return []

def fuzzy_match_team(api_name, csv_teams):
    if csv_teams is None or len(csv_teams) == 0: return None
    match, score = process.extractOne(api_name, csv_teams)
    return match if score >= 75 else None

def odosli_email(data, ziadne_tipy=False):
    style = "<style>table {border-collapse: collapse; width: 100%; font-family: sans-serif;} th {background: #1a237e; color: white; padding: 10px;} td {padding: 8px; border-bottom: 1px solid #ddd; text-align: center;} .vklad {color: #d32f2f; font-weight: bold;}</style>"
    if ziadne_tipy:
        html = "<html><body><p>Dnes žiadne hodnotné príležitosti.</p></body></html>"
        subject = f"💤 AI Report: {datetime.now().strftime('%d.%m')}"
    else:
        df = pd.DataFrame(data)
        html = f"<html><body><h2 style='color: #1a237e;'>🎯 TOP 3 AI Tipy dňa</h2>{style}{df.to_html(index=False, escape=False)}</body></html>"
        subject = f"🚀 TOP 3 AI Tipy - {datetime.now().strftime('%d.%m')}"
    msg = MIMEMultipart(); msg['Subject'] = subject; msg['From'] = GMAIL_USER; msg['To'] = GMAIL_RECEIVER
    msg.attach(MIMEText(html, 'html'))
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.starttls(); s.login(GMAIL_USER, GMAIL_PASSWORD); s.send_message(msg)
    except Exception as e: print(f"Email error: {e}")

# --- 4. HLAVNÝ SPÚŠŤAČ ---

def spustit_analyzu():
    all_potential_bets = []
    print(f"🚀 Štart analýzy: {datetime.now().strftime('%H:%M')}")

    for liga, cfg in LIGY_CONFIG.items():
        df = stiahni_csv_data(cfg['csv'])
        stats, avg_h, avg_a = vypocitaj_silu_timov(df)
        matches = ziskaj_kurzy(cfg['api'])
        if not matches or stats is None: continue
        sport_type = 'nhl' if 'NHL' in liga else 'futbal'

        for m in matches:
            api_h, api_a = m['home_team'], m['away_team']
            csv_h = fuzzy_match_team(api_h, stats.index)
            csv_a = fuzzy_match_team(api_a, stats.index)
            if not csv_h or not csv_a: continue
            probs = predikuj_vsetko(csv_h, csv_a, stats, avg_h, avg_a, sport_type)
            if not probs: continue

            for bookie in m.get('bookmakers', []):
                h2h = next((mk for mk in bookie.get('markets', []) if mk['key'] == 'h2h'), None)
                if h2h:
                    for out in h2h['outcomes']:
                        p = probs['ML1'] if (sport_type=='nhl' and out['name']==api_h) else (probs['1'] if out['name']==api_h else (probs['2'] if out['name']==api_a else probs['X']))
                        edge = (p * out['price']) - 1
                        if edge > 0:
                            all_potential_bets.append({
                                'Čas': get_local_time(m['commence_time']), 'Zápas': f"{api_h}-{api_a}",
                                'Tip': out['name'], 'Kurz': out['price'], 'Edge': round(edge * 100, 2), 
                                'Vklad': f"<span class='vklad'>{vypocitaj_kelly(p, out['price'])}%</span>"
                            })
                
                totals = next((mk for mk in bookie.get('markets', []) if mk['key'] == 'totals'), None)
                if totals:
                    limit = 5.5 if sport_type == 'nhl' else 2.5
                    for out in totals['outcomes']:
                        if out.get('point') == limit:
                            p = probs['over'] if out['name'].lower() == 'over' else probs['under']
                            edge = (p * out['price']) - 1
                            if edge > 0:
                                all_potential_bets.append({
                                    'Čas': get_local_time(m['commence_time']), 'Zápas': f"{api_h}-{api_a}",
                                    'Tip': f"{out['name']} {limit}", 'Kurz': out['price'], 'Edge': round(edge * 100, 2), 
                                    'Vklad': f"<span class='vklad'>{vypocitaj_kelly(p, out['price'])}%</span>"
                                })

    if all_potential_bets:
        # ZORADENIE A VÝBER TOP 5
        final_top = sorted(all_potential_bets, key=lambda x: x['Edge'], reverse=True)[:5]
        # Formátovanie Edge pre tabuľku
        for item in final_top: item['Edge'] = f"{item['Edge']}%"
        odosli_email(final_top)
    else:
        odosli_email([], ziadne_tipy=True)
    print("✅ Hotovo.")

if __name__ == "__main__":
    spustit_analyzu()

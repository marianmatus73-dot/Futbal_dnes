import asyncio, aiohttp, pandas as pd, numpy as np, io, os, logging
from scipy.stats import poisson
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fuzzywuzzy import process

# --- 1. KONFIGURÁCIA ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
load_dotenv()

API_ODDS_KEY = os.getenv('ODDS_API_KEY')
AKTUALNY_BANK = float(os.getenv('AKTUALNY_BANK', 1000))
KELLY_FRAC = 0.20  

LIGY_HOKEJ = {
    '🏒 NHL':               {'csv': 'NHL', 'api': 'icehockey_nhl', 'ha': 0.15},
    '🏒 Česko Extraliga':   {'csv': 'CZE', 'api': 'icehockey_czech_extraliga', 'ha': 0.28},
    '🏒 Slovensko':         {'csv': 'SVK', 'api': 'icehockey_slovakia_extraliga', 'ha': 0.35},
    '🏒 Nemecko DEL':       {'csv': 'GER', 'api': 'icehockey_germany_del', 'ha': 0.25},
    '🏒 Švédsko SHL':       {'csv': 'SWE', 'api': 'icehockey_sweden_shl', 'ha': 0.22},
    '🏒 Fínsko Liiga':      {'csv': 'FIN', 'api': 'icehockey_finland_liiga', 'ha': 0.20}
}

# --- 2. MATEMATICKÉ JADRO ---

def analyzuj_hokej_skore(lh, la, limit=5.5):
    """Vektorizovaný Poisson upravený pre hokejové limity."""
    k = np.arange(15) # Hokej málokedy presiahne 14 gólov na tím
    p_h = poisson.pmf(k, lh)
    p_a = poisson.pmf(k, la)
    matrix = np.outer(p_h, p_a)
    
    prob_1 = np.sum(np.tril(matrix, -1))
    prob_X = np.sum(np.diag(matrix))
    prob_2 = np.sum(np.triu(matrix, 1))
    
    rows, cols = np.indices(matrix.shape)
    prob_under = np.sum(matrix[rows + cols < limit])
    
    return {'1': prob_1, 'X': prob_X, '2': prob_2, f'Under {limit}': prob_under, f'Over {limit}': 1 - prob_under}

# --- 3. DÁTA A SPRACOVANIE ---

async def fetch_hokej_csv(session, liga, cfg):
    try:
        curr_year = datetime.now().year
        # Dynamické linky pre hokejové repozitáre
        if cfg['csv'] == 'NHL':
            url = f"https://raw.githubusercontent.com/martineon/nhl-historical-data/master/data/nhl_results_{curr_year}.csv"
        else:
            url = f"https://raw.githubusercontent.com/pavel-jara/hockey-data/master/data/{cfg['csv']}_{curr_year}.csv"
        
        async with session.get(url, timeout=12) as r:
            if r.status == 200: return liga, await r.read()
            return liga, None
    except: return liga, None

def ziskaj_hokej_stats(content):
    try:
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        # Mapovanie stĺpcov (unifikácia rôznych CSV zdrojov)
        df = df.rename(columns={'home_team':'HT','away_team':'AT','home_goals':'HG','away_goals':'AG','team1':'HT','team2':'AT','score1':'HG','score2':'AG'})
        
        avg_h, avg_a = df['HG'].mean(), df['AG'].mean()
        teams = list(set(df['HT'].unique()) | set(df['AT'].unique()))
        stats = pd.DataFrame(index=teams)
        
        h_stats = df.groupby('HT').agg({'HG':'mean', 'AG':'mean'})
        a_stats = df.groupby('AT').agg({'AG':'mean', 'HG':'mean'})
        
        stats['AH'] = h_stats['HG'] / avg_h # Attack Home
        stats['DH'] = h_stats['AG'] / avg_a # Defense Home
        stats['AA'] = a_stats['AG'] / avg_a # Attack Away
        stats['DA'] = a_stats['HG'] / avg_h # Defense Away
        return stats.fillna(1.0), avg_h, avg_a
    except: return None, 0, 0



async def spust_hokej():
    async with aiohttp.ClientSession() as session:
        csv_data = await asyncio.gather(*(fetch_hokej_csv(session, l, c) for l, c in LIGY_HOKEJ.items()))
        
        for liga, content in csv_data:
            if not content: continue
            cfg = LIGY_HOKEJ[liga]
            stats, avg_h, avg_a = ziskaj_hokej_stats(content)
            if stats is None: continue

            # Odds API pre konkrétnu hokejovú ligu
            async with session.get(f'https://api.the-odds-api.com/v4/sports/{cfg["api"]}/odds/', 
                                  params={'apiKey': API_ODDS_KEY, 'regions': 'eu', 'markets': 'h2h,totals'}) as r:
                if r.status != 200: continue
                matches = await r.json()

            for m in matches:
                # Fuzzy matching mien tímov
                c1_m = process.extractOne(m['home_team'], stats.index)
                c2_m = process.extractOne(m['away_team'], stats.index)
                
                if c1_m and c2_m and c1_m[1] > 65 and c2_m[1] > 65:
                    h, a = c1_m[0], c2_m[0]
                    
                    # Výpočet lambda (očakávané góly)
                    lh = (stats.at[h,'AH'] * stats.at[a,'DA'] * avg_h) + cfg['ha']
                    la = (stats.at[a,'AA'] * stats.at[h,'DH'] * avg_a)
                    
                    probs = analyzuj_hokej_skore(lh, la, limit=5.5)
                    
                    # Vyhodnotenie Edge
                    for bk in m['bookmakers']:
                        for mk in bk['markets']:
                            for out in mk['outcomes']:
                                key = '1' if out['name'] == m['home_team'] else ('2' if out['name'] == m['away_team'] else 'X')
                                if mk['key'] == 'totals':
                                    key = f"{out['name']} 5.5"
                                
                                if key in probs:
                                    prob, price = probs[key], out['price']
                                    edge = (prob * price) - 1
                                    
                                    if 0.06 <= edge <= 0.40:
                                        vklad = round(((edge / (price - 1)) * KELLY_FRAC) * AKTUALNY_BANK, 2)
                                        print(f"🏒 {liga}: {h}-{a} | Tip: {key} | Kurz: {price} | Edge: {round(edge*100,1)}% | Vklad: {vklad}€")

if __name__ == "__main__":
    asyncio.run(spust_hokej())

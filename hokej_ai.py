import asyncio, aiohttp, pandas as pd, numpy as np, io, os, logging
from scipy.stats import poisson
from datetime import datetime
from dotenv import load_dotenv
from fuzzywuzzy import process

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

async def fetch_hokej_csv(session, liga, cfg):
    curr_year = datetime.now().year
    urls = []
    if cfg['csv'] == 'NHL':
        urls = [f"https://raw.githubusercontent.com/martineon/nhl-historical-data/master/data/nhl_results_{curr_year}.csv",
                f"https://raw.githubusercontent.com/martineon/nhl-historical-data/master/data/nhl_results_{curr_year-1}.csv"]
    else:
        urls = [f"https://raw.githubusercontent.com/pavel-jara/hockey-data/master/data/{cfg['csv']}_{curr_year}.csv",
                f"https://raw.githubusercontent.com/pavel-jara/hockey-data/master/data/{cfg['csv']}_{curr_year-1}.csv"]

    for url in urls:
        try:
            async with session.get(url, timeout=10) as r:
                if r.status == 200:
                    logging.info(f"✅ Hokej {liga}: Načítané dáta z {url}")
                    return liga, await r.read()
        except: continue
    return liga, None

def ziskaj_stats(content):
    try:
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        df = df.rename(columns={'home_team':'HT','away_team':'AT','home_goals':'HG','away_goals':'AG','team1':'HT','team2':'AT','score1':'HG','score2':'AG'})
        avg_h, avg_a = df['HG'].mean(), df['AG'].mean()
        stats = pd.DataFrame(index=list(set(df['HT'].unique()) | set(df['AT'].unique())))
        h_stats = df.groupby('HT').agg({'HG':'mean', 'AG':'mean'})
        a_stats = df.groupby('AT').agg({'AG':'mean', 'HG':'mean'})
        stats['AH'] = h_stats['HG'] / avg_h; stats['DH'] = h_stats['AG'] / avg_a
        stats['AA'] = a_stats['AG'] / avg_a; stats['DA'] = a_stats['HG'] / avg_h
        return stats.fillna(1.0), avg_h, avg_a
    except: return None, 0, 0

async def spust_hokej():
    async with aiohttp.ClientSession() as session:
        csv_data = await asyncio.gather(*(fetch_hokej_csv(session, l, c) for l, c in LIGY_HOKEJ.items()))
        for liga, content in csv_data:
            if not content: continue
            cfg = LIGY_HOKEJ[liga]; stats, ah, aa = ziskaj_stats(content)
            if stats is None: continue
            async with session.get(f'https://api.the-odds-api.com/v4/sports/{cfg["api"]}/odds/', params={'apiKey':API_ODDS_KEY,'regions':'eu','markets':'h2h'}) as r:
                if r.status != 200: continue
                for m in await r.json():
                    c1_m, c2_m = process.extractOne(m['home_team'], stats.index), process.extractOne(m['away_team'], stats.index)
                    if c1_m and c2_m and c1_m[1] > 65:
                        lh = (stats.at[c1_m[0],'AH']*stats.at[c2_m[0],'DA']*ah)+cfg['ha']
                        la = (stats.at[c2_m[0],'AA']*stats.at[c1_m[0],'DH']*aa)
                        p1 = poisson.pmf(np.arange(15), lh); p2 = poisson.pmf(np.arange(15), la)
                        prob1 = np.sum(np.tril(np.outer(p1, p2), -1))
                        for bk in m['bookmakers']:
                            for out in bk['markets'][0]['outcomes']:
                                if out['name'] == m['home_team']:
                                    edge = (prob1 * out['price']) - 1
                                    if 0.05 < edge < 0.4:
                                        v = round(((edge/(out['price']-1))*KELLY_FRAC)*AKTUALNY_BANK, 2)
                                        print(f"🏒 {liga}: {m['home_team']} @{out['price']} | Edge: {round(edge*100,1)}% | Vklad: {v}€")

if __name__ == "__main__":
    asyncio.run(spust_hokej())

import asyncio, aiohttp, pandas as pd, numpy as np, io, os
from scipy.stats import poisson
from datetime import datetime
from fuzzywuzzy import process

LIGY_HOKEJ = {
    '🏒 NHL (USA)':           {'csv': 'NHL', 'api': 'icehockey_nhl', 'ha': 0.15},
    '🏒 Extraliga (CZE)':     {'csv': 'CZE', 'api': 'icehockey_czech_extraliga', 'ha': 0.28},
    '🏒 Extraliga (SVK)':     {'csv': 'SVK', 'api': 'icehockey_slovakia_extraliga', 'ha': 0.35},
    '🏒 DEL (GER)':           {'csv': 'GER', 'api': 'icehockey_germany_del', 'ha': 0.25},
    '🏒 SHL (SWE)':           {'csv': 'SWE', 'api': 'icehockey_sweden_shl', 'ha': 0.22},
    '🏒 Liiga (FIN)':         {'csv': 'FIN', 'api': 'icehockey_finland_liiga', 'ha': 0.20}
}

async def ziskaj_hokej_tipy():
    api_key = os.getenv('ODDS_API_KEY')
    bank = float(os.getenv('AKTUALNY_BANK', 1000))
    nase_tipy = []
    curr = datetime.now().year
    async with aiohttp.ClientSession() as session:
        for liga, cfg in LIGY_HOKEJ.items():
            url = f"https://raw.githubusercontent.com/martineon/nhl-historical-data/master/data/nhl_results_{curr}.csv" if cfg['csv'] == 'NHL' else f"https://raw.githubusercontent.com/pavel-jara/hockey-data/master/data/{cfg['csv']}_{curr}.csv"
            async with session.get(url) as r:
                if r.status != 200: continue
                df = pd.read_csv(io.StringIO((await r.read()).decode('utf-8'))).rename(columns={'home_team':'HT','away_team':'AT','home_goals':'HG','away_goals':'AG','team1':'HT','team2':'AT','score1':'HG','score2':'AG'})
            
            avg_h, avg_a = df['HG'].mean(), df['AG'].mean()
            stats = pd.DataFrame(index=list(set(df['HT'].unique()) | set(df['AT'].unique())))
            h_s = df.groupby('HT').agg({'HG':'mean', 'AG':'mean'}); a_s = df.groupby('AT').agg({'AG':'mean', 'HG':'mean'})
            stats['AH'] = h_s['HG']/avg_h; stats['DH'] = h_s['AG']/avg_a; stats['AA'] = a_s['AG']/avg_a; stats['DA'] = a_s['HG']/avg_h
            stats = stats.fillna(1.0)

            async with session.get(f'https://api.the-odds-api.com/v4/sports/{cfg["api"]}/odds/', 
                                  params={'apiKey':api_key,'regions':'eu','markets':'h2h,totals'}) as r:
                if r.status == 200:
                    matches = await r.json()
                    for m in matches:
                        c1 = process.extractOne(m['home_team'], stats.index)
                        c2 = process.extractOne(m['away_team'], stats.index)
                        if c1 and c2 and c1[1] > 70:
                            lh, la = (stats.at[c1[0],'AH']*stats.at[c2[0],'DA']*avg_h)+cfg['ha'], (stats.at[c2[0],'AA']*stats.at[c1[0],'DH']*avg_a)
                            matrix = np.outer(poisson.pmf(np.arange(15), lh), poisson.pmf(np.arange(15), la))
                            for bk in m['bookmakers']:
                                for mk in bk['markets']:
                                    for out in mk['outcomes']:
                                        prob, label = 0, ""
                                        if mk['key'] == 'h2h':
                                            if out['name'] == m['home_team']: prob, label = np.sum(np.tril(matrix, -1)), "1"
                                            elif out['name'] == m['away_team']: prob, label = np.sum(np.triu(matrix, 1)), "2"
                                            else: prob, label = np.sum(np.diag(matrix)), "X"
                                        elif mk['key'] == 'totals':
                                            limit = out.get('point', 5.5)
                                            p_under = np.sum(matrix[np.indices((15,15))[0] + np.indices((15,15))[1] < limit])
                                            prob = (1 - p_under) if out['name'].lower() == 'over' else p_under
                                            label = f"{out['name']} {limit}"
                                        
                                        edge = (prob * out['price']) - 1
                                        if 0.05 < edge < 0.45:
                                            v = round(((edge/(out['price']-1))*0.1)*bank, 2)
                                            nase_tipy.append({'Šport': '🏒 Hokej', 'Zápas': f"{c1[0]} vs {c2[0]}", 'Tip': label, 'Kurz': out['price'], 'Edge': f"{round(edge*100,1)}%", 'Vklad': f"{v}€", 'Očak. skóre': f"{round(lh,1)}:{round(la,1)}"})
    return nase_tipy

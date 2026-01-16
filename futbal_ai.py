import asyncio, aiohttp, pandas as pd, numpy as np, io, os
from scipy.stats import poisson
from datetime import datetime
from fuzzywuzzy import process

LIGY_FUTBAL = {
    '⚽ Premier League (ENG)': {'csv': 'E0', 'api': 'soccer_epl', 'ha': 0.25},
    '⚽ Championship (ENG)':   {'csv': 'E1', 'api': 'soccer_efl_champ', 'ha': 0.20},
    '⚽ League 1 (ENG)':       {'csv': 'E2', 'api': 'soccer_england_league1', 'ha': 0.18},
    '⚽ League 2 (ENG)':       {'csv': 'E3', 'api': 'soccer_england_league2', 'ha': 0.15},
    '⚽ Bundesliga (GER)':     {'csv': 'D1', 'api': 'soccer_germany_bundesliga', 'ha': 0.30},
    '⚽ Bundesliga 2 (GER)':   {'csv': 'D2', 'api': 'soccer_germany_bundesliga2', 'ha': 0.25},
    '⚽ La Liga (ESP)':        {'csv': 'SP1', 'api': 'soccer_spain_la_liga', 'ha': 0.28},
    '⚽ La Liga 2 (ESP)':      {'csv': 'SP2', 'api': 'soccer_spain_segunda_division', 'ha': 0.20},
    '⚽ Serie A (ITA)':        {'csv': 'I1', 'api': 'soccer_italy_serie_a', 'ha': 0.22},
    '⚽ Serie B (ITA)':        {'csv': 'I2', 'api': 'soccer_italy_serie_b', 'ha': 0.18},
    '⚽ Ligue 1 (FRA)':        {'csv': 'F1', 'api': 'soccer_france_ligue_one', 'ha': 0.25},
    '⚽ Ligue 2 (FRA)':        {'csv': 'F2', 'api': 'soccer_france_ligue_two', 'ha': 0.20},
    '⚽ Eredivisie (NED)':     {'csv': 'N1', 'api': 'soccer_netherlands_eredivisie', 'ha': 0.30},
    '⚽ Pro League (BEL)':     {'csv': 'B1', 'api': 'soccer_belgium_first_division', 'ha': 0.28},
    '⚽ Premiership (SCO)':    {'csv': 'SC0', 'api': 'soccer_scotland_premier_league', 'ha': 0.35},
    '⚽ Süper Lig (TUR)':      {'csv': 'T1', 'api': 'soccer_turkey_super_league', 'ha': 0.32}
}

async def ziskaj_futbal_tipy():
    api_key = os.getenv('ODDS_API_KEY')
    bank = float(os.getenv('AKTUALNY_BANK', 1000))
    nase_tipy = []
    async with aiohttp.ClientSession() as session:
        for liga, cfg in LIGY_FUTBAL.items():
            rok = datetime.now().strftime('%y')
            sezona = f"{rok}{str(int(rok)+1)}"
            url = f"https://www.football-data.co.uk/mmz4281/{sezona}/{cfg['csv']}.csv"
            async with session.get(url) as r:
                if r.status != 200: continue
                df = pd.read_csv(io.StringIO((await r.read()).decode('utf-8')))
            
            avg_h, avg_a = df['FTHG'].mean(), df['FTAG'].mean()
            stats = pd.DataFrame(index=list(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())))
            h_s = df.groupby('HomeTeam').agg({'FTHG':'mean', 'FTAG':'mean'})
            a_s = df.groupby('AwayTeam').agg({'FTAG':'mean', 'FTHG':'mean'})
            stats['AH'] = h_s['FTHG']/avg_h; stats['DH'] = h_s['FTAG']/avg_a
            stats['AA'] = a_s['FTAG']/avg_a; stats['DA'] = a_s['FTHG']/avg_h
            stats = stats.fillna(1.0)

            async with session.get(f'https://api.the-odds-api.com/v4/sports/{cfg["api"]}/odds/', 
                                  params={'apiKey':api_key,'regions':'eu','markets':'h2h,totals'}) as r:
                if r.status == 200:
                    matches = await r.json()
                    for m in matches:
                        c1 = process.extractOne(m['home_team'], stats.index)
                        c2 = process.extractOne(m['away_team'], stats.index)
                        if c1 and c2 and c1[1] > 75:
                            lh, la = (stats.at[c1[0],'AH']*stats.at[c2[0],'DA']*avg_h)+cfg['ha'], (stats.at[c2[0],'AA']*stats.at[c1[0],'DH']*avg_a)
                            matrix = np.outer(poisson.pmf(np.arange(10), lh), poisson.pmf(np.arange(10), la))
                            for bk in m['bookmakers']:
                                for mk in bk['markets']:
                                    for out in mk['outcomes']:
                                        prob, label = 0, ""
                                        if mk['key'] == 'h2h':
                                            if out['name'] == m['home_team']: prob, label = np.sum(np.tril(matrix, -1)), "1"
                                            elif out['name'] == m['away_team']: prob, label = np.sum(np.triu(matrix, 1)), "2"
                                            else: prob, label = np.sum(np.diag(matrix)), "X"
                                        elif mk['key'] == 'totals':
                                            limit = out.get('point', 2.5)
                                            p_under = np.sum(matrix[np.indices((10,10))[0] + np.indices((10,10))[1] < limit])
                                            prob = (1 - p_under) if out['name'].lower() == 'over' else p_under
                                            label = f"{out['name']} {limit}"
                                        
                                        edge = (prob * out['price']) - 1
                                        if 0.05 < edge < 0.45:
                                            v = round(((edge/(out['price']-1))*0.1)*bank, 2)
                                            nase_tipy.append({'Šport': '⚽ Futbal', 'Zápas': f"{c1[0]} vs {c2[0]}", 'Tip': label, 'Kurz': out['price'], 'Edge': f"{round(edge*100,1)}%", 'Vklad': f"{v}€", 'Očak. skóre': f"{round(lh,1)}:{round(la,1)}"})
    return nase_tipy

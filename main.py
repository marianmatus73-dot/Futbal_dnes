import asyncio, os, smtplib, pandas as pd
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

from futbal_ai import ziskaj_futbal_tipy
from hokej_ai import ziskaj_hokej_tipy
from tenis_ai import ziskaj_tenis_tipy

load_dotenv()

async def hlavny_proces():
    print(f"🚀 Štart analýzy: {datetime.now().strftime('%d.%m %H:%M')}")
    
    ulohy = [ziskaj_futbal_tipy(), ziskaj_hokej_tipy(), ziskaj_tenis_tipy()]
    vysledky = await asyncio.gather(*ulohy)
    vsetky_tipy = [tip for podzoznam in vysledky for tip in podzoznam]

    if not vsetky_tipy:
        print("📭 Dnes žiadne výhodné stávky s požadovaným Edge.")
        return

    df = pd.DataFrame(vsetky_tipy)
    
    # Zoradenie podľa Edge
    df['Edge_Num'] = df['Edge'].str.replace('%', '').astype(float)
    df = df.sort_values(by='Edge_Num', ascending=False).drop(columns=['Edge_Num'])

    posli_email(df)

def posli_email(df):
    msg = MIMEMultipart()
    msg['Subject'] = f"🏆 AI VALUE BETS: {len(df)} tipov ({datetime.now().strftime('%d.%m')})"
    msg['From'] = os.getenv('GMAIL_USER')
    msg['To'] = os.getenv('GMAIL_RECEIVER')

    # CSS štýl pre krajšiu tabuľku v maily
    html = f"""
    <html>
    <head>
        <style>
            table {{ border-collapse: collapse; width: 100%; font-family: sans-serif; }}
            th {{ background: #2c3e50; color: white; padding: 10px; text-align: center; }}
            td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            tr:nth-child(even) {{ background: #f9f9f9; }}
            .high-edge {{ color: #27ae60; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h2>🎯 AI Analýza - Výhodné stávky</h2>
        {df.to_html(index=False, escape=False)}
        <p><small>Očak. skóre je vypočítané na základe Poissonovej distribúcie (xG).</small></p>
    </body>
    </html>
    """
    msg.attach(MIMEText(html, 'html'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(os.getenv('GMAIL_USER'), os.getenv('GMAIL_PASSWORD'))
            server.send_message(msg)
            print(f"📧 Report s {len(df)} tipmi odoslaný!")
    except Exception as e:
        print(f"❌ Email error: {e}")

if __name__ == "__main__":
    asyncio.run(hlavny_proces())

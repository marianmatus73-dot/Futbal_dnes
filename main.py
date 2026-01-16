import asyncio, os, smtplib, pandas as pd
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

# Importujeme funkcie z ostatných súborov
from futbal_ai import ziskaj_futbal_tipy
from hokej_ai import ziskaj_hokej_tipy
from tenis_ai import ziskaj_tenis_tipy

load_dotenv()

async def hlavny_proces():
    print(f"--- 🚀 Štart kompletnej analýzy: {datetime.now().strftime('%d.%m.%Y %H:%M')} ---")
    
    # Spustenie všetkých športov naraz
    ulohy = [
        ziskaj_futbal_tipy(),
        ziskaj_hokej_tipy(),
        ziskaj_tenis_tipy()
    ]
    
    vysledky = await asyncio.gather(*ulohy)
    
    # Spojenie výsledkov do jednej tabuľky
    vsetky_tipy = [tip for podzoznam in vysledky for tip in podzoznam]

    if not vsetky_tipy:
        print("📭 Dnes žiadne výhodné stávky nenájdené.")
        return

    df = pd.DataFrame(vsetky_tipy)
    
    # Zoradenie podľa sily Edge (vytvoríme pomocný stĺpec na zoradenie)
    df['Edge_Val'] = df['Edge'].str.replace('%', '').astype(float)
    df = df.sort_values(by='Edge_Val', ascending=False).drop(columns=['Edge_Val'])

    posli_email(df)

def posli_email(df):
    msg = MIMEMultipart()
    pocet = len(df)
    msg['Subject'] = f"🏆 AI REPORT ({pocet} tipov) - {datetime.now().strftime('%d.%m')}"
    msg['From'] = os.getenv('GMAIL_USER')
    msg['To'] = os.getenv('GMAIL_RECEIVER')

    html = f"""
    <html>
    <head>
        <style>
            table {{ border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 14px; }}
            th {{ background-color: #2c3e50; color: white; padding: 12px; text-align: center; }}
            td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .edge-win {{ color: #27ae60; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h2 style="color: #2c3e50;">🎯 AI Analýza: Najlepšie dnešné príležitosti</h2>
        <p>Na základe štatistických modelov Poisson (futbal/hokej) a Form-Indexu (tenis) boli nájdené nasledujúce zápasy:</p>
        {df.to_html(index=False, escape=False)}
        <br>
        <hr>
        <p style="font-size: 12px; color: grey;">Tento email je generovaný automaticky systémom GitHub Actions.</p>
    </body>
    </html>
    """
    msg.attach(MIMEText(html, 'html'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(os.getenv('GMAIL_USER'), os.getenv('GMAIL_PASSWORD'))
            server.send_message(msg)
            print(f"📧 Report s {pocet} tipmi bol úspešne odoslaný na email.")
    except Exception as e:
        print(f"❌ Chyba pri odosielaní emailu: {e}")

if __name__ == "__main__":
    asyncio.run(hlavny_proces())

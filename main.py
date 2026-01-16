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
    print("🚀 Štartujem kombinovanú analýzu (všetky ligy)...")
    ulohy = [ziskaj_futbal_tipy(), ziskaj_hokej_tipy(), ziskaj_tenis_tipy()]
    vysledky = await asyncio.gather(*ulohy)
    vsetky_tipy = [tip for podzoznam in vysledky for tip in podzoznam]

    if not vsetky_tipy:
        print("📭 Dnes žiadne výhodné stávky.")
        return

    df = pd.DataFrame(vsetky_tipy)
    df['Edge_Num'] = df['Edge'].str.replace('%', '').astype(float)
    df = df.sort_values(by='Edge_Num', ascending=False).drop(columns=['Edge_Num'])

    msg = MIMEMultipart()
    msg['Subject'] = f"🏆 AI REPORT: {len(df)} tipov ({datetime.now().strftime('%d.%m')})"
    msg['From'] = os.getenv('GMAIL_USER')
    msg['To'] = os.getenv('GMAIL_RECEIVER')

    html = f"<html><body><h2>Dnešné AI tipy</h2>{df.to_html(index=False)}</body></html>"
    msg.attach(MIMEText(html, 'html'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(os.getenv('GMAIL_USER'), os.getenv('GMAIL_PASSWORD'))
            server.send_message(msg)
            print("📧 Report úspešne odoslaný!")
    except Exception as e:
        print(f"❌ Chyba: {e}")

if __name__ == "__main__":
    asyncio.run(hlavny_proces())

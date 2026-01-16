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
    print(f"🚀 Štart: {datetime.now().strftime('%d.%m %H:%M')}")
    ulohy = [ziskaj_futbal_tipy(), ziskaj_hokej_tipy(), ziskaj_tenis_tipy()]
    vysledky = await asyncio.gather(*ulohy)
    vsetky_tipy = [tip for podzoznam in vysledky for tip in podzoznam]
    if not vsetky_tipy: return print("📭 Žiadne tipy.")
    
    df = pd.DataFrame(vsetky_tipy)
    df['Edge_N'] = df['Edge'].str.replace('%','').astype(float)
    df = df.sort_values(by='Edge_N', ascending=False).drop(columns=['Edge_N'])
    
    msg = MIMEMultipart()
    msg['Subject'] = f"🏆 AI VALUE BETS ({len(df)}) - {datetime.now().strftime('%d.%m')}"
    msg['From'], msg['To'] = os.getenv('GMAIL_USER'), os.getenv('GMAIL_RECEIVER')
    html = f"<html><body><h2>Dnešné AI tipy</h2>{df.to_html(index=False)}</body></html>"
    msg.attach(MIMEText(html, 'html'))
    
    with smtplib.SMTP('smtp.gmail.com', 587) as s:
        s.starttls()
        s.login(os.getenv('GMAIL_USER'), os.getenv('GMAIL_PASSWORD'))
        s.send_message(msg)
    print("📧 Report odoslaný!")

if __name__ == "__main__": asyncio.run(hlavny_proces())

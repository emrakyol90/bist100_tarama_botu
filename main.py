import os
import time
import sqlite3
import threading
import logging
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import numpy as np
import pandas as pd
import yfinance as yf
import holidays
from flask import Flask, jsonify


# ============================================================
# LOGGING (ANINDA RENDER EKRANINA BASILMASI İÇİN UNBUFFERED)
# ============================================================

sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

log = logging.getLogger("BIST_BOT")

# ============================================================
# AYARLAR
# ============================================================

TZ = ZoneInfo("Europe/Istanbul")

PORT = int(os.getenv("PORT", "10000"))
DB_FILE = os.getenv("DB_FILE", "bist_bot.sqlite3")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

SCAN_HOUR = 18
SCAN_MINUTE = 30

MIN_TARGET_PCT = 10.0
MIN_RR = 2.0


# Türkiye resmi ve dini tatilleri
tr_holidays = holidays.TR()

def is_bist_holiday(dt_obj):
    if dt_obj.weekday() >= 5:
        return True
    if dt_obj in tr_holidays:
        return True
    if dt_obj.month == 10 and dt_obj.day == 28:
        return True
    return False

def is_trade_day(dt_obj):
    return not is_bist_holiday(dt_obj)

def is_last_trade_day_of_month(dt_obj):
    if not is_trade_day(dt_obj):
        return False
    curr = dt_obj + timedelta(days=1)
    while curr.month == dt_obj.month:
        if is_trade_day(curr):
            return False
        curr += timedelta(days=1)
    return True


# ============================================================
# FLASK & DATABASE
# ============================================================

app = Flask(__name__)
scan_lock = threading.Lock()

MONTHS = {
    1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan",
    5: "Mayıs", 6: "Haziran", 7: "Temmuz", 8: "Ağustos",
    9: "Eylul", 10: "Ekim", 11: "Kasım", 12: "Aralık"
}


def get_db():
    c = sqlite3.connect(DB_FILE, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL;")
    return c


def init_db():
    c = get_db()
    c.execute("""
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        signal_time TEXT NOT NULL,
        month_key TEXT NOT NULL,
        score INTEGER NOT NULL,
        technical_score INTEGER NOT NULL,
        fundamental_score INTEGER NOT NULL,
        entry REAL NOT NULL,
        target REAL NOT NULL,
        stop REAL NOT NULL,
        target_pct REAL NOT NULL,
        rr REAL NOT NULL,
        status TEXT NOT NULL DEFAULT 'OPEN',
        exit_price REAL,
        exit_time TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS scans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_time TEXT NOT NULL,
        month_key TEXT NOT NULL,
        total_symbols INTEGER NOT NULL,
        scanned_ok INTEGER NOT NULL,
        candidates INTEGER NOT NULL,
        market_regime TEXT NOT NULL
    )
    """)

    if not c.execute("SELECT 1 FROM meta WHERE key='start_time'").fetchone():
        c.execute("INSERT INTO meta(key,value) VALUES(?,?)", ("start_time", datetime.now(TZ).isoformat()))

    if not c.execute("SELECT 1 FROM meta WHERE key='last_monthly_report'").fetchone():
        c.execute("INSERT INTO meta(key,value) VALUES(?,?)", ("last_monthly_report", ""))

    c.commit()
    c.close()


init_db()


# ============================================================
# TELEGRAM SERVICE (DETAYLI LOGLAMALI)
# ============================================================

def telegram_send(text, chat_id=None):
    target_chat = chat_id or TELEGRAM_CHAT_ID
    log.info(f"Telegram mesajı gönderilmeye çalışılıyor. Chat ID: {target_chat}")
    
    if not TELEGRAM_BOT_TOKEN or not target_chat:
        log.error("❌ HATA: TELEGRAM_BOT_TOKEN veya TELEGRAM_CHAT_ID Environment Variable olarak tanımlı değil!")
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": target_chat,
            "text": text,
            "disable_web_page_preview": True
        }
        r = requests.post(url, json=payload, timeout=20)
        
        if r.ok:
            log.info("✅ Telegram mesajı başarıyla iletildi.")
            return True
        else:
            log.error(f"❌ Telegram API Hatası [{r.status_code}]: {r.text}")
            return False
    except Exception as e:
        log.exception(f"❌ Telegram istek atarken istisna oluştu: {e}")
        return False


def telegram_long(text, chat_id=None):
    for i in range(0, len(text), 3800):
        telegram_send(text[i:i + 3800], chat_id)


# ============================================================
# BIST EVRENİ & YAHOO
# ============================================================

FALLBACK_SYMBOLS = ["THYAO", "GARAN", "ISCTR", "AKBNK", "EREGL", "ASELS", "BIMAS", "SISE", "KCHOL", "TUPRS"]

def normalize_symbol(s):
    s = str(s).strip().upper()
    if s.endswith(".IS"):
        s = s[:-3]
    return s if s else None

def get_all_symbols():
    env = os.getenv("BIST_SYMBOLS", "").strip()
    if env:
        return sorted(set(normalize_symbol(x) for x in env.split(",") if normalize_symbol(x)))

    try:
        url = "https://www.isyatirim.com.tr/_layouts/15/IsYatirim.MusteriKanalı/CommonData.aspx/GetHisseList"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if r.ok:
            data = r.json()
            symbols = [item["code"] for item in data if "code" in item and item["code"]]
            if len(symbols) >= 300:
                log.info(f"BIST evreni çekildi: {len(symbols)} hisse.")
                return sorted(set(symbols))
    except Exception as e:
        log.warning(f"İş Yatırım sembol listesi alınamadı, yedeğe geçiliyor: {e}")

    return sorted(set(FALLBACK_SYMBOLS))


def yahoo_symbol(symbol):
    if symbol in ("^XU100", "^GSPC", "^IXIC"):
        return symbol
    return symbol if symbol.endswith(".IS") else symbol + ".IS"


def get_history(symbol, period="2y", interval="1d"):
    try:
        df = yf.download(
            yahoo_symbol(symbol),
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False
        )
        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(x).lower() for x in df.columns]
        cols = ["open", "high", "low", "close", "volume"]

        if not all(x in df.columns for x in cols):
            return pd.DataFrame()

        return df[cols].dropna()
    except Exception:
        return pd.DataFrame()


# ============================================================
# İNDİKATÖRLER & TEKNİK / TEMEL
# ============================================================

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def atr(df, n=14):
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def wavetrend(df, n1=10, n2=21):
    ap = (df["high"] + df["low"] + df["close"]) / 3
    esa = ap.ewm(span=n1, adjust=False).mean()
    de = (ap - esa).abs().ewm(span=n1, adjust=False).mean()
    ci = (ap - esa) / (0.015 * de.replace(0, np.nan))
    wt1 = ci.ewm(span=n2, adjust=False).mean()
    wt2 = wt1.rolling(4).mean()
    return wt1, wt2

def to_4h(df):
    return df.resample("4h", offset="2h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()

def technical_analysis(daily, four_hour):
    if len(daily) < 200 or len(four_hour) < 40: return None
    d, h = daily.copy(), four_hour.copy()
    d["ema20"], d["ema50"], d["ema200"], d["atr"] = ema(d["close"], 20), ema(d["close"], 50), ema(d["close"], 200), atr(d)
    h["wt1"], h["wt2"] = wavetrend(h)

    a, old, x, p = d.iloc[-1], d.iloc[-11], h.iloc[-1], h.iloc[-2]
    wt_signal = (x["wt1"] < 0 and x["wt2"] < 0) and (p["wt1"] <= p["wt2"] and x["wt1"] > x["wt2"]) and (x["wt1"] > p["wt1"]) and (x["close"] >= x["open"])
    
    score = ((20 if a["close"] > a["ema200"] else 0) + (10 if a["ema200"] > old["ema200"] else 0) +
             (10 if a["ema20"] > a["ema50"] else 0) + (15 if wt_signal else 0) + 
             (5 if a["volume"] >= d["volume"].rolling(20).mean().iloc[-1] * 0.80 else 0))

    return {"technical_score": int(score), "close": float(a["close"]), "ema200": float(a["ema200"]), "atr": float(a["atr"]), "wt_signal": bool(wt_signal)}

def fundamental_score(symbol):
    try:
        url = f"https://www.isyatirim.com.tr/_layouts/15/IsYatirim.MaliTablolar/MaliTablo.aspx/GetOranlar?hisse={symbol}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"}, timeout=10)
        if not r.ok or not r.json().get("d"): return 0
        metrics = {item["KOD"]: float(str(item["DEGER"]).replace(",", ".")) for item in r.json()["d"] if "KOD" in item and item.get("DEGER") is not None}
        
        roe, p_growth, r_growth, d_eq, op_m, pe, pb = metrics.get("ROE"), metrics.get("NET_KAR_BUYUME"), metrics.get("SATIS_BUYUME"), metrics.get("BORC_OZSERMAYE"), metrics.get("FAALIYET_MARJI"), metrics.get("FK"), metrics.get("PDDD")
        score = 0
        if roe is not None: score += (8 if roe >= 20 else 5 if roe >= 10 else 0)
        if p_growth is not None: score += (8 if p_growth >= 20 else 5 if p_growth > 0 else 0)
        if r_growth is not None: score += (6 if r_growth >= 15 else 3 if r_growth > 0 else 0)
        if d_eq is not None: score += (6 if d_eq <= 100 else 3 if d_eq <= 200 else 0)
        if op_m is not None: score += (5 if op_m >= 15 else 3 if op_m > 0 else 0)
        if pe is not None: score += (4 if 0 < pe <= 20 else 2 if pe <= 35 else 0)
        if pb is not None: score += (3 if 0 < pb <= 3 else 1 if pb <= 6 else 0)
        return score
    except Exception: return 0

def final_score(tech, fund):
    tot = tech + fund
    if tech < 45: return 0
    if tot >= 90 and tech >= 50 and fund >= 32: return 100
    if tot >= 82 and tech >= 48 and fund >= 28: return 90
    if tot >= 72 and tech >= 45 and fund >= 24: return 80
    if tot >= 60 and tech >= 45 and fund >= 18: return 60
    return 0

def trade_plan(t):
    entry = t["close"]
    r_dist = max(t["atr"] * 1.5, entry * 0.01)
    stop = entry - r_dist
    if stop <= 0: return None
    target = max(entry * (1 + MIN_TARGET_PCT / 100.0), entry + r_dist * MIN_RR)
    target_pct = (target / entry - 1) * 100
    rr = (target - entry) / r_dist
    if target_pct < MIN_TARGET_PCT or rr < MIN_RR: return None
    return {"entry": entry, "target": target, "stop": stop, "target_pct": target_pct, "rr": rr}

def market_regime():
    df = get_history("^XU100", "2y", "1d")
    if df.empty or len(df) < 200: return {"value": None, "ema200": None, "regime": "BİLİNMİYOR"}
    e200 = ema(df["close"], 200)
    v, ev = float(df["close"].iloc[-1]), float(e200.iloc[-1])
    return {"value": v, "ema200": ev, "regime": "YÜKSELİŞ" if v > ev else "DÜŞÜŞ"}

def month_key(dt=None):
    if dt is None: dt = datetime.now(TZ)
    return f"{dt.year:04d}-{dt.month:02d}"

def analyze_symbol(symbol):
    daily = get_history(symbol, "2y", "1d")
    if daily.empty: return None, False
    hourly = get_history(symbol, "60d", "1h")
    if hourly.empty: return None, False
    tech = technical_analysis(daily, to_4h(hourly))
    if not tech or tech["close"] <= tech["ema200"] or not tech["wt_signal"]: return None, True
    fund = fundamental_score(symbol)
    score = final_score(tech["technical_score"], fund)
    if score == 0: return None, True
    plan = trade_plan(tech)
    if not plan: return None, True
    return {"symbol": symbol, "score": score, "technical_score": tech["technical_score"], "fundamental_score": fund, **plan}, True

def save_signal(x):
    now = datetime.now(TZ)
    c = get_db()
    exists = c.execute("SELECT id FROM signals WHERE symbol=? AND signal_time LIKE ?", (x["symbol"], now.strftime("%Y-%m-%d") + "%")).fetchone()
    if not exists:
        c.execute("""
        INSERT INTO signals(symbol, signal_time, month_key, score, technical_score, fundamental_score, entry, target, stop, target_pct, rr)
        VALUES(?,?,?,?,?,?,?,?,?,?,?)
        """, (x["symbol"], now.isoformat(), month_key(now), x["score"], x["technical_score"], x["fundamental_score"], x["entry"], x["target"], x["stop"], x["target_pct"], x["rr"]))
        c.commit()
        c.close()
        return True
    c.close()
    return False

def generate_monthly_report(current_m_key=None):
    if not current_m_key: current_m_key = month_key()
    c = get_db()
    signals = c.execute("SELECT * FROM signals WHERE month_key=? ORDER BY signal_time ASC", (current_m_key,)).fetchall()
    c.close()
    now = datetime.now(TZ)
    m_name = MONTHS.get(now.month, "")

    if not signals:
        return f"📊 BIST BOTU - {m_name.upper()} {now.year} GÜNCEL RAPORU\n━━━━━━━━━━━━━━━━━━\n\nBu ay için henüz kayıtlı sinyal bulunmuyor."

    total = len(signals)
    target_cnt = sum(1 for s in signals if s["status"] == "TARGET")
    stop_cnt = sum(1 for s in signals if s["status"] == "STOP")
    open_cnt = sum(1 for s in signals if s["status"] == "OPEN")
    completed = target_cnt + stop_cnt
    win_rate = (target_cnt / completed * 100) if completed > 0 else 0

    lines = [
        f"📊 BIST BOTU - {m_name.upper()} {now.year} GÜNCEL RAPORU",
        "━━━━━━━━━━━━━━━━━━",
        f"🎯 Toplam Sinyal: {total} | ✅ Hedef: {target_cnt} | 🛑 Stop: {stop_cnt} | ⏳ Açık: {open_cnt}",
        f"🏆 Başarı Oranı: %{win_rate:.1f}\n"
    ]
    return "\n".join(lines)


def scan_market(chat_id=None):
    if not scan_lock.acquire(blocking=False):
        if chat_id: telegram_send("⚠️ Taramada zaten aktif bir işlem yürütülüyor.", chat_id)
        return

    try:
        log.info("🔍 Tarama süreci başlatıldı...")
        if chat_id: telegram_send("🔍 Tarama başlatıldı, piyasa taranıyor...", chat_id)
        
        symbols = get_all_symbols()
        market = market_regime()
        candidates, scanned_ok = [], 0

        for symbol in symbols:
            try:
                res, ok = analyze_symbol(symbol)
                if ok: scanned_ok += 1
                if res: candidates.append(res)
            except Exception as e:
                log.error(f"Analiz hatası ({symbol}): {e}")
            time.sleep(0.02)

        candidates.sort(key=lambda x: (x["score"], x["target_pct"], x["rr"]), reverse=True)
        for c in candidates: save_signal(c)

        lines = [f"🔍 BIST TARAMA RAPORU - {datetime.now(TZ).strftime('%d.%m.%Y %H:%M')}", "━━━━━━━━━━━━━━━━━━"]
        if not candidates:
            lines.append("❌ Stratejiye uygun hisse bulunamadı.")
        else:
            for x in candidates:
                lines.append(f"📌 {x['symbol']} ({x['score']} Puan)\nGiriş: {x['entry']:.2f} | Hedef: {x['target']:.2f} | Stop: {x['stop']:.2f}\n")

        telegram_long("\n".join(lines), chat_id or TELEGRAM_CHAT_ID)
        log.info("✅ Tarama başarıyla bitti.")
    finally:
        scan_lock.release()


# ============================================================
# TELEGRAM SOHBET DİNLEYİCİ (POLLING LOOP)
# ============================================================

def telegram_poll_loop():
    log.info("🔄 Telegram sohbet dinleyici (Polling Thread) başlatılıyor...")
    if not TELEGRAM_BOT_TOKEN:
        log.error("❌ TELEGRAM_BOT_TOKEN yok, dinleyici başlatılamadı!")
        return

    offset = 0
    log.info("✅ Telegram dinleyici yayında, komutlar bekleniyor.")

    while True:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates?offset={offset}&timeout=10"
            r = requests.get(url, timeout=15)
            if r.ok:
                updates = r.json().get("result", [])
                for update in updates:
                    offset = update["update_id"] + 1
                    msg = update.get("message", {})
                    text = msg.get("text", "").strip().lower()
                    chat_id = msg.get("chat", {}).get("id")

                    if not text or not chat_id: continue

                    log.info(f"📩 Telegram'dan Komut Geldi: '{text}' (Chat ID: {chat_id})")

                    if "rapor" in text:
                        rep = generate_monthly_report()
                        telegram_send(rep, chat_id)
                    elif "tarama" in text or "tara" in text or text == "/scan":
                        threading.Thread(target=scan_market, args=(chat_id,), daemon=True).start()
                    elif text in ("/start", "merhaba", "sa", "selam"):
                        telegram_send("👋 BIST Botu aktif! Komutlar:\n• 'rapor'\n• 'tarama'", chat_id)
            else:
                log.error(f"❌ Telegram getUpdates hatası: {r.text}")
        except Exception as e:
            log.error(f"❌ Polling döngü hatası: {e}")
            time.sleep(5)
        time.sleep(1)


# ============================================================
# UYUMAMA (KEEP-ALIVE) THREAD'İ
# ============================================================

def keep_alive():
    """Render'ın ücretsiz planının sunucuyu uyutmasını engellemek için kendine istek atar."""
    log.info("😴 Keep-Alive (Uyumama) servisi aktif edildi.")
    while True:
        time.sleep(600)  # 10 dakikada bir çalış
        try:
            log.info("⏰ Sunucu aktif tutuluyor (Self-Ping)...")
            requests.get(f"http://127.0.0.1:{PORT}/", timeout=5)
        except Exception as e:
            log.warning(f"Self-ping uyarısı: {e}")


def scheduler():
    last_scan_date = ""
    while True:
        try:
            now = datetime.now(TZ)
            today_str = now.strftime("%Y-%m-%d")
            if (is_trade_day(now.date()) and now.hour == SCAN_HOUR and now.minute == SCAN_MINUTE and last_scan_date != today_str):
                last_scan_date = today_str
                log.info(f"⏰ Otomatik tarama tetiklendi: {today_str}")
                threading.Thread(target=scan_market, daemon=True).start()
            time.sleep(20)
        except Exception as e:
            log.error(f"Scheduler hatası: {e}")
            time.sleep(30)


# ============================================================
# BAŞLANGIÇ SERVİSLERİ
# ============================================================

def start_bot_background_services():
    log.info("🚀 BIST Botu Servisleri Başlatılıyor...")
    
    # Telegram Başlangıç Bildirimi
    sent = telegram_send("🚀 BIST Tarama Botu başarıyla başlatıldı ve servise girdi!")
    if not sent:
        log.error("⚠️ Başlangıç mesajı Telegram'a GÖNDERİLEMEDİ! Token veya Chat ID dizilimlerini kontrol edin.")

    # Thread'ler
    threading.Thread(target=scheduler, daemon=True).start()
    threading.Thread(target=telegram_poll_loop, daemon=True).start()
    threading.Thread(target=keep_alive, daemon=True).start()

# Sunucu başlangıcında arka plan servislerini bağımsız çalıştır
def init_app():
    log.info("⚙️ Arka plan servisleri başlatılıyor...")
    threading.Thread(target=start_bot_background_services, daemon=True).start()

# Flask uygulaması yüklendiğinde servisleri tetikle
init_app()

# ============================================================
# FLASK
# ============================================================

@app.get("/")
def home():
    return "BIST Tarama Botu Aktif."

@app.get("/report")
def get_report():
    return generate_monthly_report()

@app.post("/scan")
def manual_scan():
    threading.Thread(target=scan_market, daemon=True).start()
    return jsonify({"ok": True, "message": "Tarama başlatıldı."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)

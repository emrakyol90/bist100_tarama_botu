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

MIN_TARGET_PCT = 6.0
MIN_RR = 1.5


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
services_started = False
scan_lock = threading.Lock()

MONTHS = {
    1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan",
    5: "Mayıs", 6: "Haziran", 7: "Temmuz", 8: "Ağustos",
    9: "Eylul", 10: "Ekim", 11: "Kasım", 12: "Aralık"
}


def get_db():
    c = sqlite3.connect(DB_FILE, check_same_thread=False, timeout=30.0)
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
# TELEGRAM SERVICE
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

    # 1. DENEME: İş Yatırım
    try:
        url = "https://www.isyatirim.com.tr/_layouts/15/IsYatirim.MusteriKanali/CommonData.aspx/GetHisseList"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if r.ok:
            fetched = [item["code"] for item in r.json() if "code" in item and item["code"]]
            if len(fetched) >= 300:
                log.info(f"✅ İş Yatırım üzerinden {len(fetched)} hisse otomatik çekildi.")
                return sorted(set(fetched))
    except Exception as e:
        log.warning(f"⚠️ İş Yatırım dinamik liste çekilemedi: {e}")

    # 2. DENEME: TradingView BIST Taraması
    try:
        tv_url = "https://scanner.tradingview.com/turkey/scan"
        payload = {
            "filter": [{"left": "type", "operation": "in_range", "right": ["stock", "dr", "fund"]}],
            "symbols": {"query": {"types": []}},
            "columns": ["name"],
            "range": [0, 1000]
        }
        r = requests.post(tv_url, json=payload, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if r.ok:
            data = r.json().get("data", [])
            fetched = [item["s"].split(":")[-1] for item in data if "s" in item]
            if len(fetched) >= 300:
                log.info(f"✅ TradingView üzerinden {len(fetched)} hisse otomatik çekildi.")
                return sorted(set(fetched))
    except Exception as e:
        log.warning(f"⚠️ TradingView üzerinden liste çekilemedi: {e}")

    # 3. DENEME: Güncel Açık Kaynak JSON
    try:
        gh_url = "https://raw.githubusercontent.com/fawazahmed0/currency-api/1/bist.json"
        r = requests.get(gh_url, timeout=10)
        if r.ok:
            fetched = r.json()
            if isinstance(fetched, list) and len(fetched) >= 300:
                log.info(f"✅ Açık kaynak yedek listeden {len(fetched)} hisse çekildi.")
                return sorted(set(fetched))
    except Exception as e:
        log.warning(f"⚠️ Açık kaynak yedek listeden çekilemedi: {e}")

    log.error("❌ Otomatik listelerin hiçbiri çekilemedi, acil durum yedeği kullanılıyor!")
    return sorted(set(FALLBACK_SYMBOLS))

def yahoo_symbol(symbol):
    if symbol in ("^XU100", "XU100.IS", "^GSPC", "^IXIC"):
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

    a, old = d.iloc[-1], d.iloc[-11]

    # Son 6 mum (gün içindeki 24 saat) kontrolü
    wt_signal = False
    lookback = min(6, len(h) - 1)
    for i in range(1, lookback + 1):
        x_candle = h.iloc[-i]
        p_candle = h.iloc[-i-1]
        if (x_candle["wt1"] < 25 and x_candle["wt2"] < 25) and (p_candle["wt1"] <= p_candle["wt2"] and x_candle["wt1"] > x_candle["wt2"]) and (x_candle["close"] >= x_candle["open"]):
            wt_signal = True
            break

    score = ((20 if a["close"] > a["ema200"] else 0) + 
             (10 if a["ema200"] > old["ema200"] else 0) +
             (10 if a["ema20"] > a["ema50"] else 0) + 
             (25 if wt_signal else 0) + 
             (10 if a["volume"] >= d["volume"].rolling(20).mean().iloc[-1] * 0.80 else 0))

    return {"technical_score": int(score), "close": float(a["close"]), "ema200": float(a["ema200"]), "atr": float(a["atr"]), "wt_signal": bool(wt_signal)}

def fundamental_score(symbol):
    try:
        url = f"https://www.isyatirim.com.tr/_layouts/15/IsYatirim.MaliTablolar/MaliTablo.aspx/GetOranlar?hisse={symbol}"
        r = requests.get(
            url, 
            headers={"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest", "Connection": "close"}, 
            timeout=4
        )
        if not r.ok or not r.json().get("d"): return 0
        
        metrics = {}
        for item in r.json()["d"]:
            if "KOD" in item and item.get("DEGER") is not None:
                try:
                    metrics[item["KOD"]] = float(str(item["DEGER"]).replace(",", "."))
                except ValueError:
                    continue
        
        roe, p_growth, r_growth, d_eq, op_m, pe, pb = (
            metrics.get("ROE"), metrics.get("NET_KAR_BUYUME"), metrics.get("SATIS_BUYUME"), 
            metrics.get("BORC_OZSERMAYE"), metrics.get("FAALIYET_MARJI"), metrics.get("FK"), metrics.get("PDDD")
        )
        score = 0
        if roe is not None: score += (8 if roe >= 20 else 5 if roe >= 10 else 0)
        if p_growth is not None: score += (8 if p_growth >= 20 else 5 if p_growth > 0 else 0)
        if r_growth is not None: score += (6 if r_growth >= 15 else 3 if r_growth > 0 else 0)
        if d_eq is not None: score += (6 if d_eq <= 100 else 3 if d_eq <= 200 else 0)
        if op_m is not None: score += (5 if op_m >= 15 else 3 if op_m > 0 else 0)
        if pe is not None: score += (4 if 0 < pe <= 20 else 2 if pe <= 35 else 0)
        if pb is not None: score += (3 if 0 < pb <= 3 else 1 if pb <= 6 else 0)
        return score
    except Exception: 
        return 0

def trade_plan(t):
    entry = t["close"]
    r_dist = max(t["atr"] * 1.5, entry * 0.01)
    stop = entry - r_dist
    if stop <= 0: return None
    target = max(entry * (1 + MIN_TARGET_PCT / 100.0), entry + r_dist * MIN_RR)
    target_pct = (target / entry - 1) * 100
    rr = (target - entry) / r_dist

    if target_pct < MIN_TARGET_PCT or rr < MIN_RR: 
        return None

    return {"entry": entry, "target": target, "stop": stop, "target_pct": target_pct, "rr": rr}

def market_regime():
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        url = "https://query1.finance.yahoo.com/v8/finance/chart/XU100.IS?range=1y&interval=1d"
        
        res = requests.get(url, headers=headers, timeout=10)
        data = res.json()
        
        prices = data['chart']['result'][0]['indicators']['quote'][0]['close']
        clean_prices = [p for p in prices if p is not None]
        
        if not clean_prices:
            return {"value": None, "ema200": None, "regime": "BİLİNMİYOR"}
            
        v = round(clean_prices[-1], 2)
        
        if len(clean_prices) >= 200:
            s = pd.Series(clean_prices)
            ev = round(s.ewm(span=200, adjust=False).mean().iloc[-1], 2)
        else:
            ev = v
            
        regime_str = "YÜKSELİŞ 📈 (EMA200 Üstünde)" if v > ev else "DÜŞÜŞ 📉 (EMA200 Altında)"
        return {"value": v, "ema200": ev, "regime": regime_str}
        
    except Exception as e:
        log.error(f"Market rejim hatası: {e}")
        return {"value": None, "ema200": None, "regime": "BİLİNMİYOR"}


def month_key(dt=None):
    if dt is None:
        dt = datetime.now(TZ)
    return f"{dt.year:04d}-{dt.month:02d}"

def save_signal(x):
    c = get_db()

    today = datetime.now(TZ).strftime("%Y-%m-%d")

    existing = c.execute("""
        SELECT 1 FROM signals
        WHERE symbol=?
        AND date(signal_time)=?
    """, (x["symbol"], today)).fetchone()

    if existing:
        c.close()
        return False

    c.execute("""
    INSERT INTO signals (
        symbol, signal_time, month_key,
        score, technical_score, fundamental_score,
        entry, target, stop, target_pct, rr, status
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
    """, (
        x["symbol"],
        datetime.now(TZ).isoformat(),
        month_key(),
        x["score"],
        x["technical_score"],
        x["fundamental_score"],
        x["entry"],
        x["target"],
        x["stop"],
        x["target_pct"],
        x["rr"]
    ))

    c.commit()
    c.close()
    return True

    

def analyze_symbol(symbol):
    # ============================================================
    # AÇIK SİNYAL KONTROLÜ
    # ============================================================

    c = get_db()
    open_sig = c.execute(
        "SELECT id FROM signals WHERE symbol=? AND status='OPEN'",
        (symbol,)
    ).fetchone()
    c.close()

    if open_sig:
        return None, False, "OPEN"

    # ============================================================
    # VERİLERİ AL
    # ============================================================

    daily = get_history(symbol, "2y", "1d")

    if daily.empty:
        return None, False, "DATA_DAILY"

    hourly = get_history(symbol, "60d", "1h")

    if hourly.empty:
        return None, False, "DATA_HOURLY"

    four_hour = to_4h(hourly)

    # ============================================================
    # NORMAL TEKNİK ANALİZ
    # ============================================================

    tech = technical_analysis(daily, four_hour)

    # ============================================================
    # ÖZEL TARAMA
    #
    # Teknik analiz için 200 günlük / 40 adet 4H veri yetmiyorsa
    # hisseyi çöpe atmıyoruz.
    #
    # Eldeki günlük veriden:
    # - EMA20
    # - EMA50
    # - ATR
    # - Hacim
    #
    # hesaplanıyor.
    #
    # Ayrıca temel analiz yapılıyor.
    #
    # Böylece bu hisselere de:
    # GİRİŞ / TP / STOP / RR
    # çıkarıyoruz.
    # ============================================================

    if not tech:
        log.info(
            f"🔎 ÖZEL TARAMA: {symbol} | "
            f"Günlük veri: {len(daily)} | "
            f"4H veri: {len(four_hour)}"
        )

    if len(daily) < 50:
        return None, True, "SPECIAL_DAILY_DATA"

    d = daily.copy()
    d["ema20"] = ema(d["close"], 20)
    d["ema50"] = ema(d["close"], 50)
    d["atr"] = atr(d)
    last = d.iloc[-1]

    entry = float(last["close"])
    atr_value = float(last["atr"])

    if entry <= 0 or atr_value <= 0:
        return None, True, "SPECIAL_PLAN"

    special_technical = 0

    if entry > float(last["ema20"]):
        special_technical += 10

    if float(last["ema20"]) > float(last["ema50"]):
        special_technical += 10

    avg_volume = d["volume"].rolling(20).mean().iloc[-1]

    if pd.notna(avg_volume) and avg_volume > 0:
        if float(last["volume"]) >= float(avg_volume) * 0.80:
            special_technical += 10

    if float(last["close"]) >= float(last["open"]):
        special_technical += 10

    fund = fundamental_score(symbol)

    if fund < 5:
        return None, True, "SPECIAL_FUND"

    score = special_technical + fund

    if score < 35:
        return None, True, "SPECIAL_SCORE"

    risk_distance = max(
        atr_value * 1.5,
        entry * 0.01
    )

    stop = entry - risk_distance

    if stop <= 0:
        return None, True, "SPECIAL_PLAN"

    target = max(
        entry * (1 + MIN_TARGET_PCT / 100.0),
        entry + risk_distance * MIN_RR
    )

    target_pct = (target / entry - 1) * 100
    rr = (target - entry) / risk_distance

    if target_pct < MIN_TARGET_PCT or rr < MIN_RR:
        return None, True, "SPECIAL_PLAN"

    return {
        
        "symbol": symbol,
        "score": int(score),
        "technical_score": int(special_technical),
        "fundamental_score": int(fund),
        "entry": entry,
        "target": target,
        "stop": stop,
        "target_pct": target_pct,
        "rr": rr,
        "daily_data": len(daily),
        "four_hour_data": len(four_hour)
    }, True, "SPECIAL_SIGNAL"
    # ============================================================
    # NORMAL STRATEJİ
    # ============================================================

    # EMA200 filtresi
    if tech["close"] <= tech["ema200"]:
        return None, True, "EMA200"

    # WT filtresi
    if not tech["wt_signal"]:
        return None, True, "WT"

    # Teknik skor filtresi
    if tech["technical_score"] < 45:
        return None, True, "TECH_SCORE"

    # Temel analiz
    fund = fundamental_score(symbol)

    # Temel skor filtresi
    if fund < 5:
        return None, True, "FUND"

    score = tech["technical_score"] + fund

    # İşlem planı
    plan = trade_plan(tech)

    if not plan:
        return None, True, "PLAN"

    return {
        "symbol": symbol,
        "score": score,
        "technical_score": tech["technical_score"],
        "fundamental_score": fund,
        **plan
    }, True, "SIGNAL"
def generate_monthly_report(mode="all", current_m_key=None):
    c = get_db()

    m_reg = market_regime()
    xu100_val = f"{m_reg['value']:.2f}" if m_reg["value"] is not None else "Alınamadı"
    regime_text = m_reg["regime"]

    if mode == "all":
        signals = c.execute("SELECT * FROM signals ORDER BY signal_time ASC").fetchall()
        title = "TÜM ZAMANLAR RAPORU"
    else:
        if current_m_key is None:
            current_m_key = month_key()

        signals = c.execute(
            "SELECT * FROM signals WHERE month_key=? ORDER BY signal_time ASC",
            (current_m_key,)
        ).fetchall()

        dt = datetime.strptime(current_m_key, "%Y-%m")
        title = f"{MONTHS[dt.month].upper()} {dt.year} AYLIK RAPORU"

    start_row = c.execute("SELECT value FROM meta WHERE key='start_time'").fetchone()
    c.close()

    if start_row and start_row["value"]:
        start_dt = datetime.fromisoformat(start_row["value"])
        uptime_days = (datetime.now(TZ) - start_dt).days
        uptime_text = f"{uptime_days} Gün"
    else:
        uptime_text = "Bilinmiyor"

    total = len(signals)
    target_cnt = sum(1 for s in signals if s["status"] == "TARGET")
    stop_cnt = sum(1 for s in signals if s["status"] == "STOP")
    open_cnt = sum(1 for s in signals if s["status"] == "OPEN")

    completed = target_cnt + stop_cnt
    win_rate = (target_cnt / completed * 100) if completed else 0

    return (
        f"📊 BIST BOTU - {title}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"⏱️ Bot Çalışma Süresi: {uptime_text}\n"
        f"📈 BIST 100 Endeks: {xu100_val}\n"
        f"🚦 Piyasa Trendi: {regime_text}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🎯 Toplam Sinyal: {total}\n"
        f"✅ Hedef: {target_cnt}\n"
        f"🛑 Stop: {stop_cnt}\n"
        f"⏳ Açık: {open_cnt}\n"
        f"🏆 Win Rate: %{win_rate:.1f}"
    )

def scan_market(chat_id=None):
    if not scan_lock.acquire(blocking=False):
        if chat_id:
            telegram_send(
                "⚠️ Taramada zaten aktif bir işlem yürütülüyor.",
                chat_id
            )
        return

    try:
        log.info("🔍 Tarama süreci başlatıldı...")

        if chat_id:
            telegram_send(
                "🔍 Tarama başlatıldı, piyasa taranıyor...",
                chat_id
            )

        symbols = get_all_symbols()
        market = market_regime()

        candidates = []
        special_candidates = []

        scanned_ok = 0

        # ============================================================
        # FİLTRE SAYAÇLARI
        # ============================================================

        count_ema200 = 0
        count_wt = 0
        count_tech = 0
        count_fund = 0
        count_plan = 0
        
        # ÖZEL TARAMA ELEME SAYAÇLARI
        count_special_daily = 0
        count_special_fund = 0
        count_special_score = 0
        count_special_plan = 0

        # VERİ ALAMAYAN HİSSELER
        count_data = 0
        # ============================================================
        # TÜM HİSSELERİ TARA
        # ============================================================

        for symbol in symbols:

            try:

                res, ok, reason = analyze_symbol(symbol)

                if ok:
                    scanned_ok += 1

                if reason == "EMA200":
                    count_ema200 += 1
                
                elif reason == "WT":
                    count_wt += 1
                
                elif reason == "TECH_SCORE":
                    count_tech += 1
                
                elif reason == "FUND":
                    count_fund += 1
                
                elif reason == "PLAN":
                    count_plan += 1
                
                elif reason == "SPECIAL_DAILY_DATA":
                    count_special_daily += 1
                
                elif reason == "SPECIAL_FUND":
                    count_special_fund += 1
                
                elif reason == "SPECIAL_SCORE":
                    count_special_score += 1
                
                elif reason == "SPECIAL_PLAN":
                    count_special_plan += 1
                
                elif reason == "SPECIAL_SIGNAL":
                    special_candidates.append(res)
                
                elif reason in ("DATA_DAILY", "DATA_HOURLY"):
                    count_data += 1
                
                elif reason == "SIGNAL":
                    candidates.append(res)

            except Exception as e:
                log.error(
                    f"Analiz hatası ({symbol}): {e}"
                )

        # ============================================================
        # NORMAL SİNYALLERİ SIRALA
        # ============================================================

        candidates.sort(
            key=lambda x: (
                x["score"],
                x["target_pct"],
                x["rr"]
            ),
            reverse=True
        )

        # ============================================================
        # ÖZEL SİNYALLERİ SIRALA
        # ============================================================

        special_candidates.sort(
            key=lambda x: (
                x["score"],
                x["target_pct"],
                x["rr"]
            ),
            reverse=True
        )

        # ============================================================
        # NORMAL + ÖZEL SİNYALLER DB'YE KAYDEDİLİR

        for candidate in candidates:
            save_signal(candidate)

        for candidate in special_candidates:
            save_signal(candidate)

        # ============================================================
        # MARKET BİLGİSİ
        # ============================================================

        xu100_val = (
            f"{market['value']:.2f}"
            if market["value"] is not None
            else "Alınamadı"
        )

        regime_text = market["regime"]

        total_symbols_count = len(symbols)

        # ============================================================
        # TELEGRAM RAPORU
        # ============================================================

        lines = [

            f"🔍 BIST TARAMA RAPORU - "
            f"{datetime.now(TZ).strftime('%d.%m.%Y %H:%M')}",

            "━━━━━━━━━━━━━━━━━━",

            f"📈 BIST 100 Endeks: {xu100_val}",

            f"🚦 Piyasa Trendi: {regime_text}",

            f"📊 Taranan Hisse Sayısı: "
            f"{total_symbols_count} "
            f"(Başarılı: {scanned_ok})",

            "━━━━━━━━━━━━━━━━━━",

            "🔎 FİLTRE ELEME RAPORU:",
            f"❌ EMA200 altında: {count_ema200}",
            f"❌ WT şartı: {count_wt}",
            f"❌ Teknik puan: {count_tech}",
            f"❌ Temel puan: {count_fund}",
            f"❌ Hedef / RR: {count_plan}",
            
            "🔎 ÖZEL TARAMA ELEME:",
            f"⚠️ Günlük veri <50: {count_special_daily}",
            f"❌ Temel puan <5: {count_special_fund}",
            f"❌ Özel toplam puan <35: {count_special_score}",
            f"❌ Özel Hedef / RR: {count_special_plan}",
            f"⚠️ Veri alınamadı: {count_data}",
            "━━━━━━━━━━━━━━━━━━",

            f"🎯 NORMAL SİNYAL: "
            f"{len(candidates)}",

            f"🔎 ÖZEL TARAMA SİNYALİ: "
            f"{len(special_candidates)}",

            "━━━━━━━━━━━━━━━━━━"
        ]

        # ============================================================
        # NORMAL SİNYALLER
        # ============================================================

        if candidates:

            lines.append(
                "🎯 NORMAL STRATEJİ SİNYALLERİ:"
            )

            lines.append("")

            for x in candidates:

                puan = x["score"]

                seviye = (
                    "🔥 MÜKEMMEL (100+)"
                    if puan >= 100
                    else "🌟 ÇOK İYİ (75+)"
                    if puan >= 75
                    else "👍 İYİ (65+)"
                    if puan >= 65
                    else "⚡ BAŞLANGIÇ (55+)"
                )

                lines.append(

                    f"📌 {x['symbol']} - "
                    f"{puan} Puan [{seviye}]\n"

                    f"📊 Teknik: "
                    f"{x['technical_score']} | "
                    f"Temel: "
                    f"{x['fundamental_score']}\n"

                    f"💰 Giriş: "
                    f"{x['entry']:.2f}\n"

                    f"🎯 TP: "
                    f"{x['target']:.2f} "
                    f"(+%{x['target_pct']:.2f})\n"

                    f"🛑 Stop: "
                    f"{x['stop']:.2f}\n"

                    f"⚖️ RR: "
                    f"{x['rr']:.2f}\n"

                    "━━━━━━━━━━━━━━━━━━"
                )

        else:

            lines.append(
                "❌ Normal stratejiye uygun "
                "hisse bulunamadı."
            )

        # ============================================================
        # ÖZEL TARAMA SİNYALLERİ
        # ============================================================

        if special_candidates:

            lines.append("")

            lines.append(
                "🔎 ÖZEL TARAMA — "
                "VERİSİ YETERSİZ HİSSELER:"
            )

            lines.append(
                "⚠️ Bu hisseler normal "
                "200G + 4H teknik filtresinden "
                "geçmedi."
            )

            lines.append("")

            for x in special_candidates:

                puan = x["score"]

                lines.append(

                    f"🔍 {x['symbol']} - "
                    f"{puan} Puan\n"

                    f"📊 Özel Teknik: "
                    f"{x['technical_score']} | "
                    f"Temel: "
                    f"{x['fundamental_score']}\n"

                    f"📚 Veri: "
                    f"{x['daily_data']} günlük | "
                    f"{x['four_hour_data']} adet 4H\n"

                    f"💰 Giriş: "
                    f"{x['entry']:.2f}\n"

                    f"🎯 TP: "
                    f"{x['target']:.2f} "
                    f"(+%{x['target_pct']:.2f})\n"

                    f"🛑 Stop: "
                    f"{x['stop']:.2f}\n"

                    f"⚖️ RR: "
                    f"{x['rr']:.2f}\n"

                    "━━━━━━━━━━━━━━━━━━"
                )

        else:

            lines.append("")

            lines.append(
                "🔎 Özel taramada da "
                "uygun aday bulunamadı."
            )

        # ============================================================
        # TARAMA KAYDI
        # ============================================================

        c = get_db()

        c.execute("""
            INSERT INTO scans (
                scan_time,
                month_key,
                total_symbols,
                scanned_ok,
                candidates,
                market_regime
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (

            datetime.now(TZ).isoformat(),

            month_key(),

            total_symbols_count,

            scanned_ok,

            len(candidates),

            regime_text
        ))

        c.commit()
        c.close()

        # ============================================================
        # TELEGRAM
        # ============================================================

        telegram_long(
            "\n".join(lines),
            chat_id or TELEGRAM_CHAT_ID
        )

        log.info(
            f"✅ Tarama bitti. "
            f"Normal: {len(candidates)} | "
            f"Özel: {len(special_candidates)}"
        )

    finally:

        scan_lock.release()

# ============================================================
# AÇIK SİNYALLERİ TAKİP ET (MUM-BAZLI TOPLU KONTROL)
# ============================================================

def track_open_signals():
    while True:
        try:
            c = get_db()
            signals = c.execute("SELECT * FROM signals WHERE status='OPEN'").fetchall()
            c.close()

            if not signals:
                time.sleep(300)
                continue

            symbols_list = [yahoo_symbol(s["symbol"]) for s in signals]
            
            # Tüm açık hisselerin verisini tek seferde indir
            data = yf.download(
                tickers=symbols_list,
                period="1d",
                interval="5m",
                group_by="ticker",
                progress=False,
                threads=True
            )

            for s in signals:
                sym = s["symbol"]
                y_sym = yahoo_symbol(sym)

                try:
                    if len(signals) == 1:
                        df = data.dropna().copy()
                    else:
                        if isinstance(data.columns, pd.MultiIndex) and y_sym in data.columns.get_level_values(0):
                            df = data[y_sym].dropna().copy()
                        else:
                            df = pd.DataFrame()

                    if df.empty:
                        continue

                    # Sütun isimlerini garanti olarak küçük harfe çevir
                    df.columns = [str(col).lower() for col in df.columns]

                    # SADECE SİNYAL OLUŞTUKTAN SONRAKİ MUMları takip et
                    signal_dt = datetime.fromisoformat(s["signal_time"])
                    if signal_dt.tzinfo is None:
                        signal_dt = signal_dt.replace(tzinfo=TZ)
                    
                    if df.index.tz is None:
                        df.index = df.index.tz_localize(TZ)
                    else:
                        df.index = df.index.tz_convert(TZ)
                    
                    df_after_signal = df[df.index > signal_dt]
                    
                    if df_after_signal.empty:
                        continue
                    
                    last_high = float(df_after_signal["high"].max())
                    last_low = float(df_after_signal["low"].min())
                    last_close = float(df_after_signal["close"].iloc[-1])
                    # 1. ÇİFT İHLAL KONTROLÜ (Aynı mumda ikisi de olduysa Stop say)
                    if last_high >= s["target"] and last_low <= s["stop"]:
                        db = get_db()
                        db.execute(
                            "UPDATE signals SET status='STOP', exit_price=?, exit_time=? WHERE id=?",
                            (s["stop"], datetime.now(TZ).isoformat(), s["id"])
                        )
                        db.commit()
                        db.close()

                        telegram_send(
                            f"⚠️ AŞIRI VOLATİLİTE / ÇİFT İHLAL!\n\n"
                            f"📌 {sym}\n"
                            f"Aynı mumda hem hedef hem stop görüldü. Kasayı korumak için STOP kabul edildi.\n"
                            f"🛑 Stop Seviyesi: {s['stop']:.2f}\n"
                            f"🏁 Kapanış: {last_close:.2f}"
                        )

                    # 2. SADECE HEDEF KONTROLÜ
                    elif last_high >= s["target"]:
                        db = get_db()
                        db.execute(
                            "UPDATE signals SET status='TARGET', exit_price=?, exit_time=? WHERE id=?",
                            (s["target"], datetime.now(TZ).isoformat(), s["id"])
                        )
                        db.commit()
                        db.close()

                        telegram_send(
                            f"🎯 HEDEF GELDİ!\n\n"
                            f"📌 {sym}\n"
                            f"🎯 Hedef Seviyesi: {s['target']:.2f}\n"
                            f"📈 Mum En Yüksek (High): {last_high:.2f}\n"
                            f"🏁 Kapanış: {last_close:.2f}"
                        )

                    # 3. SADECE STOP KONTROLÜ
                    elif last_low <= s["stop"]:
                        db = get_db()
                        db.execute(
                            "UPDATE signals SET status='STOP', exit_price=?, exit_time=? WHERE id=?",
                            (s["stop"], datetime.now(TZ).isoformat(), s["id"])
                        )
                        db.commit()
                        db.close()

                        telegram_send(
                            f"🛑 STOP GELDİ!\n\n"
                            f"📌 {sym}\n"
                            f"🛑 Stop Seviyesi: {s['stop']:.2f}\n"
                            f"📉 Mum En Düşük (Low): {last_low:.2f}\n"
                            f"🏁 Kapanış: {last_close:.2f}"
                        )

                except Exception as ex:
                    log.error(f"Sinyal takip hatası ({sym}): {ex}")

            time.sleep(300)

        except Exception as e:
            log.error(f"❌ Sinyal takip genel döngü hatası: {e}")
            time.sleep(60)

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
            r = requests.get(url, timeout=30)
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
                        rep = generate_monthly_report(mode="all")
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
# UYUMAMA (KEEP-ALIVE) THREAD'İ & SCHEDULER
# ============================================================

def keep_alive():
    log.info("😴 Keep-Alive servisi aktif.")
    while True:
        time.sleep(600)

def scheduler():
    last_scan_date = ""
    while True:
        try:
            now = datetime.now(TZ)
            today_str = now.strftime("%Y-%m-%d")
            
            if (is_trade_day(now.date()) and now.hour == SCAN_HOUR and now.minute == SCAN_MINUTE and last_scan_date != today_str):
                last_scan_date = today_str
                log.info(f"⏰ Otomatik tarama tetiklendi: {today_str}")
                
                if is_last_trade_day_of_month(now.date()):
                    log.info("📊 Ayın son işlem günü! Tarama başlatılıyor...")

                    def month_end_scan_and_report():
                        scan_market()
                        log.info("✅ Ay sonu taraması bitti. Aylık rapor oluşturuluyor...")
                        rep = generate_monthly_report(mode="monthly")
                        telegram_send(f"🚨 AY SONU KAPANIŞ RAPORU 🚨\n\n{rep}")

                    threading.Thread(
                        target=month_end_scan_and_report,
                        daemon=True
                    ).start()

                else:
                    threading.Thread(
                        target=scan_market,
                        daemon=True
                    ).start()

            time.sleep(20)
        except Exception as e:
            log.error(f"Scheduler hatası: {e}")
            time.sleep(30)

# ============================================================
# BAŞLANGIÇ SERVİSLERİ
# ============================================================

def start_bot_background_services():
    global services_started
    if services_started: 
        return
    services_started = True
    log.info("🚀 BIST Botu Servisleri Başlatılıyor...")
    
    threading.Thread(target=telegram_send, args=("🚀 BIST Tarama Botu başarıyla başlatıldı ve servise girdi!",), daemon=True).start()
    threading.Thread(target=scheduler, daemon=True).start()
    threading.Thread(target=telegram_poll_loop, daemon=True).start()
    threading.Thread(target=keep_alive, daemon=True).start()
    threading.Thread(target=track_open_signals, daemon=True).start()

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

start_bot_background_services()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, use_reloader=False)

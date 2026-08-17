import os
import time
import sqlite3
import threading
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import numpy as np
import pandas as pd
import yfinance as yf
import holidays
from flask import Flask, jsonify


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
# LOGGING & FLASK
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger("BIST_BOT")

app = Flask(__name__)
scan_lock = threading.Lock()


# ============================================================
# AY İSİMLERİ
# ============================================================

MONTHS = {
    1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan",
    5: "Mayıs", 6: "Haziran", 7: "Temmuz", 8: "Ağustos",
    9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık"
}


# ============================================================
# DATABASE
# ============================================================

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


def get_meta(key, default=""):
    c = get_db()
    row = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    c.close()
    return row["value"] if row else default


def set_meta(key, value):
    c = get_db()
    c.execute("""
    INSERT INTO meta(key,value)
    VALUES(?,?)
    ON CONFLICT(key)
    DO UPDATE SET value=excluded.value
    """, (key, str(value)))
    c.commit()
    c.close()


# ============================================================
# TELEGRAM SERVICE
# ============================================================

def telegram_send(text, chat_id=None):
    target_chat = chat_id or TELEGRAM_CHAT_ID
    if not TELEGRAM_BOT_TOKEN or not target_chat:
        log.warning("Telegram Bot Token veya Chat ID eksik.")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": target_chat,
                "text": text,
                "disable_web_page_preview": True
            },
            timeout=20
        )
        return r.ok
    except Exception:
        log.exception("Telegram mesaj gönderme hatası")
        return False


def telegram_long(text, chat_id=None):
    for i in range(0, len(text), 3800):
        telegram_send(text[i:i + 3800], chat_id)


# ============================================================
# BIST EVRENİ & YAHOO
# ============================================================

FALLBACK_SYMBOLS = [
    "THYAO", "GARAN", "ISCTR", "AKBNK", "EREGL",
    "ASELS", "BIMAS", "SISE", "KCHOL", "TUPRS"
]


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
                log.info(f"BIST evreni başarıyla çekildi: {len(symbols)} hisse taranacak.")
                return sorted(set(symbols))
    except Exception as e:
        log.warning(f"Otomatik hisse listesi çekilemedi, yedek listeye geçiliyor: {e}")

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
# İNDİKATÖRLER & TEKNİK / TEMEL ANALİZ
# ============================================================

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def atr(df, n=14):
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs()
    ], axis=1).max(axis=1)
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
    return df.resample("4h", offset="2h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()


def technical_analysis(daily, four_hour):
    if len(daily) < 200 or len(four_hour) < 40:
        return None

    d = daily.copy()
    h = four_hour.copy()

    d["ema20"] = ema(d["close"], 20)
    d["ema50"] = ema(d["close"], 50)
    d["ema200"] = ema(d["close"], 200)
    d["atr"] = atr(d)

    h["wt1"], h["wt2"] = wavetrend(h)

    a = d.iloc[-1]
    old = d.iloc[-11]
    x = h.iloc[-1]
    p = h.iloc[-2]

    below_middle = (x["wt1"] < 0 and x["wt2"] < 0)
    bullish_cross = (p["wt1"] <= p["wt2"] and x["wt1"] > x["wt2"])
    wt_rising = (x["wt1"] > p["wt1"])
    green_candle = (x["close"] >= x["open"])

    wt_signal = (below_middle and bullish_cross and wt_rising and green_candle)
    price_above_ema200 = (a["close"] > a["ema200"])
    ema200_rising = (a["ema200"] > old["ema200"])
    ema20_above_ema50 = (a["ema20"] > a["ema50"])
    vol_avg = d["volume"].rolling(20).mean().iloc[-1]
    volume_ok = (a["volume"] >= vol_avg * 0.80)

    technical_score = (
        (20 if price_above_ema200 else 0) +
        (10 if ema200_rising else 0) +
        (10 if ema20_above_ema50 else 0) +
        (15 if wt_signal else 0) +
        (5 if volume_ok else 0)
    )

    return {
        "technical_score": int(technical_score),
        "close": float(a["close"]),
        "ema200": float(a["ema200"]),
        "atr": float(a["atr"]),
        "wt_signal": bool(wt_signal)
    }


def fundamental_score(symbol):
    try:
        url = f"https://www.isyatirim.com.tr/_layouts/15/IsYatirim.MaliTablolar/MaliTablo.aspx/GetOranlar?hisse={symbol}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "X-Requested-With": "XMLHttpRequest"
        }
        r = requests.get(url, headers=headers, timeout=10)
        if not r.ok:
            return 0

        data = r.json().get("d", [])
        if not data:
            return 0

        metrics = {}
        for item in data:
            if "KOD" in item and "DEGER" in item and item["DEGER"] is not None:
                try:
                    metrics[item["KOD"]] = float(str(item["DEGER"]).replace(",", "."))
                except ValueError:
                    continue

        roe = metrics.get("ROE")
        profit_growth = metrics.get("NET_KAR_BUYUME")
        revenue_growth = metrics.get("SATIS_BUYUME")
        debt_equity = metrics.get("BORC_OZSERMAYE")
        operating_margin = metrics.get("FAALIYET_MARJI")
        pe = metrics.get("FK")
        pb = metrics.get("PDDD")

        score = 0
        if roe is not None: score += (8 if roe >= 20 else 5 if roe >= 10 else 0)
        if profit_growth is not None: score += (8 if profit_growth >= 20 else 5 if profit_growth > 0 else 0)
        if revenue_growth is not None: score += (6 if revenue_growth >= 15 else 3 if revenue_growth > 0 else 0)
        if debt_equity is not None: score += (6 if debt_equity <= 100 else 3 if debt_equity <= 200 else 0)
        if operating_margin is not None: score += (5 if operating_margin >= 15 else 3 if operating_margin > 0 else 0)
        if pe is not None: score += (4 if 0 < pe <= 20 else 2 if pe <= 35 else 0)
        if pb is not None: score += (3 if 0 < pb <= 3 else 1 if pb <= 6 else 0)

        return score
    except Exception:
        return 0


def final_score(technical, fundamental):
    total = technical + fundamental
    if technical < 45:
        return 0

    if total >= 90 and technical >= 50 and fundamental >= 32:
        return 100
    if total >= 82 and technical >= 48 and fundamental >= 28:
        return 90
    if total >= 72 and technical >= 45 and fundamental >= 24:
        return 80
    if total >= 60 and technical >= 45 and fundamental >= 18:
        return 60

    return 0


def trade_plan(t):
    entry = t["close"]
    risk_distance = max(t["atr"] * 1.5, entry * 0.01)
    stop = entry - risk_distance

    if stop <= 0:
        return None

    target_by_pct = entry * (1 + MIN_TARGET_PCT / 100.0)
    target_by_rr = entry + risk_distance * MIN_RR
    target = max(target_by_pct, target_by_rr)

    target_pct = (target / entry - 1) * 100
    rr = (target - entry) / risk_distance

    if target_pct < MIN_TARGET_PCT or rr < MIN_RR:
        return None

    return {
        "entry": entry,
        "target": target,
        "stop": stop,
        "target_pct": target_pct,
        "rr": rr
    }


def market_regime():
    df = get_history("^XU100", "2y", "1d")
    if df.empty or len(df) < 200:
        return {"value": None, "ema200": None, "regime": "BİLİNMİYOR"}

    e200 = ema(df["close"], 200)
    value = float(df["close"].iloc[-1])
    ema_value = float(e200.iloc[-1])

    return {
        "value": value,
        "ema200": ema_value,
        "regime": "YÜKSELİŞ" if value > ema_value else "DÜŞÜŞ"
    }


def month_key(dt=None):
    if dt is None:
        dt = datetime.now(TZ)
    return f"{dt.year:04d}-{dt.month:02d}"


def analyze_symbol(symbol):
    daily = get_history(symbol, "2y", "1d")
    if daily.empty:
        return None, False

    hourly = get_history(symbol, "60d", "1h")
    if hourly.empty:
        return None, False

    four_hour = to_4h(hourly)
    technical = technical_analysis(daily, four_hour)
    if not technical:
        return None, False

    if technical["close"] <= technical["ema200"] or not technical["wt_signal"]:
        return None, True

    fundamental = fundamental_score(symbol)
    score = final_score(technical["technical_score"], fundamental)
    if score == 0:
        return None, True

    plan = trade_plan(technical)
    if not plan:
        return None, True

    return {
        "symbol": symbol,
        "score": score,
        "technical_score": technical["technical_score"],
        "fundamental_score": fundamental,
        **plan
    }, True


def save_signal(x):
    now = datetime.now(TZ)
    c = get_db()
    exists = c.execute(
        "SELECT id FROM signals WHERE symbol=? AND signal_time LIKE ?",
        (x["symbol"], now.strftime("%Y-%m-%d") + "%")
    ).fetchone()

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


def update_open_signals():
    c = get_db()
    rows = c.execute("SELECT * FROM signals WHERE status='OPEN'").fetchall()

    for row in rows:
        try:
            signal_time = datetime.fromisoformat(row["signal_time"])
            df = get_history(row["symbol"], "10d", "1d")
            if df.empty:
                continue

            if df.index.tz is not None:
                df_index = df.index.tz_convert(TZ).tz_localize(None)
            else:
                df_index = df.index

            signal_date = signal_time.astimezone(TZ).replace(tzinfo=None).date()
            valid_mask = [x.date() > signal_date for x in df_index]
            df = df.loc[valid_mask]

            if df.empty:
                continue

            high = float(df["high"].max())
            low = float(df["low"].min())

            if low <= row["stop"]:
                status, exit_price = "STOP", row["stop"]
            elif high >= row["target"]:
                status, exit_price = "TARGET", row["target"]
            else:
                continue

            c.execute("UPDATE signals SET status=?, exit_price=?, exit_time=? WHERE id=?",
                      (status, exit_price, datetime.now(TZ).isoformat(), row["id"]))

            icon = "🎯" if status == "TARGET" else "🛑"
            telegram_send(f"{icon} SİNYAL SONUCU\n{row['symbol']} | {row['score']} PUAN\nSonuç: {status}\nÇıkış: {exit_price:.2f}")
        except Exception:
            log.exception(f"Sinyal takip hatası: {row['symbol']}")

    c.commit()
    c.close()


# ============================================================
# RAPORLAR & TARAMA
# ============================================================

def generate_monthly_report(current_m_key=None):
    if not current_m_key:
        current_m_key = month_key()

    c = get_db()
    signals = c.execute("SELECT * FROM signals WHERE month_key=? ORDER BY signal_time ASC", (current_m_key,)).fetchall()
    c.close()

    try:
        report_year, report_month = int(current_m_key[:4]), int(current_m_key[5:7])
    except Exception:
        now = datetime.now(TZ)
        report_year, report_month = now.year, now.month

    month_name = MONTHS.get(report_month, "")
    if not signals:
        return f"📊 BIST BOTU - {month_name.upper()} {report_year} AYLIK RAPOR\n━━━━━━━━━━━━━━━━━━\n\nBu ay için henüz sinyal kaydı bulunamadı."

    total = len(signals)
    target_cnt = sum(1 for s in signals if s["status"] == "TARGET")
    stop_cnt = sum(1 for s in signals if s["status"] == "STOP")
    open_cnt = sum(1 for s in signals if s["status"] == "OPEN")
    completed = target_cnt + stop_cnt
    win_rate = (target_cnt / completed * 100) if completed > 0 else 0

    score_lines = []
    for score in (100, 90, 80, 60):
        group = [s for s in signals if s["score"] == score]
        g_target = sum(1 for s in group if s["status"] == "TARGET")
        g_stop = sum(1 for s in group if s["status"] == "STOP")
        g_open = sum(1 for s in group if s["status"] == "OPEN")
        g_completed = g_target + g_stop
        g_win = (g_target / g_completed * 100) if g_completed > 0 else 0
        score_lines.append(f"{score} PUAN → {len(group)} sinyal | 🎯 {g_target} | 🛑 {g_stop} | ⏳ {g_open} | Başarı %{g_win:.1f}")

    open_lines = [f"⏳ {s['symbol']} | {s['score']} P | Giriş {s['entry']:.2f} | Hedef {s['target']:.2f} | Stop {s['stop']:.2f}" for s in signals if s["status"] == "OPEN"]
    if len(open_lines) > 20:
        open_lines = open_lines[:20] + [f"... {len(open_lines) - 20} açık sinyal daha var."]

    lines = [
        f"📊 BIST BOTU - {month_name.upper()} {report_year} GÜNCEL RAPORU",
        "━━━━━━━━━━━━━━━━━━",
        f"🗓️ Raporlanan Ay: {month_name} {report_year}\n",
        "📈 GENEL İSTATİSTİK",
        f"🎯 Toplam Sinyal: {total}",
        f"✅ Hedef: {target_cnt}",
        f"🛑 Stop: {stop_cnt}",
        f"⏳ Açık: {open_cnt}",
        f"📊 Tamamlanan: {completed}",
        f"🏆 Başarı Oranı: %{win_rate:.1f}\n",
        "🏅 PUAN GRUPLARI"
    ]
    lines.extend(score_lines)
    lines.append("\n⏳ AÇIK POZİSYONLAR")
    lines.extend(open_lines if open_lines else ["Açık pozisyon yok."])

    return "\n".join(lines)


def scan_report(candidates, market, total, scanned_ok):
    now = datetime.now(TZ)
    bist_value = f"{market['value']:.2f}" if market["value"] is not None else "N/A"

    lines = [
        "🔍 BIST TARAMA RAPORU",
        now.strftime("%d.%m.%Y %H:%M"),
        "━━━━━━━━━━━━━━━━━━",
        f"Evren: {total} şirket | Verisi Alınan: {scanned_ok}",
        f"BIST100: {bist_value} (Piyasa: {market['regime']})",
        f"Bulunan Aday: {len(candidates)}\n"
    ]

    if not candidates:
        lines.append("❌ Bugün strateji şartlarını sağlayan hisse bulunamadı.")
        return "\n".join(lines)

    for score in (100, 90, 80, 60):
        group = [x for x in candidates if x["score"] == score]
        if not group:
            continue

        lines.append(f"━━ {score} PUAN | {len(group)} HİSSE ━━")
        for x in group:
            lines.extend([
                f"📌 {x['symbol']}",
                f"Teknik: {x['technical_score']}/60 | Temel: {x['fundamental_score']}/40",
                f"Giriş: {x['entry']:.2f} | Hedef: {x['target']:.2f} (+%{x['target_pct']:.1f}) | Stop: {x['stop']:.2f}",
                f"R/R: 1:{x['rr']:.2f}\n"
            ])

    return "\n".join(lines)


def scan_market(chat_id=None):
    if not scan_lock.acquire(blocking=False):
        if chat_id:
            telegram_send("⚠️ Şu anda devam eden bir tarama var, lütfen bitmesini bekleyin.", chat_id)
        return

    try:
        if chat_id:
            telegram_send("🔍 Tarama başlatıldı, piyasa taranıyor...", chat_id)

        update_open_signals()
        symbols = get_all_symbols()
        market = market_regime()
        candidates, scanned_ok = [], 0

        log.info(f"Tarama başladı. Evren: {len(symbols)}")
        for symbol in symbols:
            try:
                result, ok = analyze_symbol(symbol)
                if ok: scanned_ok += 1
                if result: candidates.append(result)
            except Exception:
                log.exception(f"Analiz hatası: {symbol}")
            time.sleep(0.05)

        candidates.sort(key=lambda x: (x["score"], x["target_pct"], x["rr"]), reverse=True)
        new_signals = sum(1 for candidate in candidates if save_signal(candidate))

        now = datetime.now(TZ)
        c = get_db()
        c.execute("""
        INSERT INTO scans(scan_time, month_key, total_symbols, scanned_ok, candidates, market_regime)
        VALUES(?,?,?,?,?,?)
        """, (now.isoformat(), month_key(now), len(symbols), scanned_ok, len(candidates), market["regime"]))
        c.commit()
        c.close()

        rep = scan_report(candidates, market, len(symbols), scanned_ok)
        telegram_long(rep, chat_id or TELEGRAM_CHAT_ID)

        log.info(f"Tarama bitti. Aday={len(candidates)} Yeni sinyal={new_signals}")
    finally:
        scan_lock.release()


# ============================================================
# TELEGRAM SOHBET DİNLEYİCİ (POLLING)
# ============================================================

def telegram_poll_loop():
    if not TELEGRAM_BOT_TOKEN:
        log.warning("Telegram Bot Token bulunamadı, dinleme başlatılamadı.")
        return

    offset = 0
    log.info("Telegram sohbet dinleyici (polling) aktif.")

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

                    if not text or not chat_id:
                        continue

                    if "rapor" in text:
                        rep = generate_monthly_report()
                        telegram_send(rep, chat_id)
                    elif "tarama" in text or "tara" in text or text == "/scan":
                        threading.Thread(target=scan_market, args=(chat_id,), daemon=True).start()
                    elif text in ("/start", "merhaba", "sa", "selam"):
                        telegram_send(
                            "👋 Merhaba! BIST Bot Hizmetinizde.\n\n"
                            "Kullanabileceğiniz Komutlar:\n"
                            "• 'rapor' yazarsanız -> Güncel aylık performans raporunu atar.\n"
                            "• 'tarama' veya 'tara' yazarsanız -> Manuel hisse taraması başlatır.",
                            chat_id
                        )
        except Exception:
            log.exception("Telegram polling hatası")
            time.sleep(5)

        time.sleep(1)


# ============================================================
# SCHEDULER
# ============================================================

def scheduler():
    last_scan_date = ""
    while True:
        try:
            now = datetime.now(TZ)
            today_date = now.date()
            today_str = today_date.strftime("%Y-%m-%d")

            if (is_trade_day(today_date) and 
                now.hour == SCAN_HOUR and 
                now.minute == SCAN_MINUTE and 
                last_scan_date != today_str):

                last_scan_date = today_str
                log.info(f"Borsa açık gününde otomatik tarama başlatılıyor: {today_str}")
                threading.Thread(target=scan_market, daemon=True).start()

            time.sleep(20)
        except Exception:
            log.exception("Scheduler hatası")
            time.sleep(30)


# ============================================================
# UYGULAMA BAŞLANGICI (GUNICORN / RENDER UYUMLU)
# ============================================================

def start_bot_background_services():
    log.info("BIST Tarama Botu servisleri başlatılıyor...")
    telegram_send("🚀 BIST Tarama Botu başarıyla başlatıldı ve servise girdi!")
    
    # Arka plan thread'leri
    threading.Thread(target=scheduler, daemon=True).start()
    threading.Thread(target=telegram_poll_loop, daemon=True).start()

# Server ister WSGI (Render) ister python app.py olarak başlasın servisleri çalıştır
start_bot_background_services()


# ============================================================
# FLASK ENDPOINTS
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

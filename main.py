import holidays
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
from flask import Flask, jsonify


# ============================================================
# AYARLAR
# ============================================================

TZ = ZoneInfo("Europe/Istanbul")

PORT = int(os.getenv("PORT", "10000"))
DB_FILE = os.getenv("DB_FILE", "bist_bot.sqlite3")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

SCAN_HOUR = 18
SCAN_MINUTE = 30

MIN_TARGET_PCT = 10.0
MIN_RR = 2.0


# Türkiye resmi ve dini tatilleri (Her yıl otomatik güncellenir)
tr_holidays = holidays.TR()

def is_bist_holiday(dt_obj):
    # Hafta sonu kontrolü (Cumartesi: 5, Pazar: 6)
    if dt_obj.weekday() >= 5:
        return True
    
    # Türkiye resmi/dini tatil kontrolü
    if dt_obj.date() in tr_holidays:
        return True
        
    # Arife günleri yarım gün (13:00'da kapanır), 18:30 taraması için kapalı sayılır
    # 29 Ekim Cumhuriyet Bayramı arife günü (28 Ekim)
    if dt_obj.month == 10 and dt_obj.day == 28:
        return True
        
    return False

def is_trade_day(dt_obj):
    return not is_bist_holiday(dt_obj)


# ============================================================
# LOGGING
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
    1: "Ocak",
    2: "Şubat",
    3: "Mart",
    4: "Nisan",
    5: "Mayıs",
    6: "Haziran",
    7: "Temmuz",
    8: "Ağustos",
    9: "Eylül",
    10: "Ekim",
    11: "Kasım",
    12: "Aralık"
}


# ============================================================
# DATABASE
# ============================================================

def get_db():
    c = sqlite3.connect(
        DB_FILE,
        check_same_thread=False
    )

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

    if not c.execute(
        "SELECT 1 FROM meta WHERE key='start_time'"
    ).fetchone():

        c.execute(
            "INSERT INTO meta(key,value) VALUES(?,?)",
            (
                "start_time",
                datetime.now(TZ).isoformat()
            )
        )

    if not c.execute(
        "SELECT 1 FROM meta WHERE key='last_monthly_report'"
    ).fetchone():

        c.execute(
            "INSERT INTO meta(key,value) VALUES(?,?)",
            (
                "last_monthly_report",
                ""
            )
        )

    c.commit()
    c.close()


init_db()


def get_meta(key, default=""):

    c = get_db()

    row = c.execute(
        "SELECT value FROM meta WHERE key=?",
        (key,)
    ).fetchone()

    c.close()

    return row["value"] if row else default


def set_meta(key, value):

    c = get_db()

    c.execute("""
    INSERT INTO meta(key,value)
    VALUES(?,?)
    ON CONFLICT(key)
    DO UPDATE SET value=excluded.value
    """, (
        key,
        str(value)
    ))

    c.commit()
    c.close()


# ============================================================
# TELEGRAM
# ============================================================

def telegram_send(text):

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:

        log.warning("Telegram ENV eksik.")

        return False

    try:

        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",

            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "disable_web_page_preview": True
            },

            timeout=20
        )

        return r.ok

    except Exception:

        log.exception("Telegram hatası")

        return False


def telegram_long(text):

    for i in range(0, len(text), 3800):

        telegram_send(
            text[i:i + 3800]
        )


# ============================================================
# İŞ GÜNÜ / TATİL KONTROLÜ
# ============================================================

# ============================================================
# BIST EVRENİ
# ============================================================

FALLBACK_SYMBOLS = [
    "THYAO",
    "GARAN",
    "ISCTR",
    "AKBNK",
    "EREGL",
    "ASELS",
    "BIMAS",
    "SISE",
    "KCHOL",
    "TUPRS"
]


def normalize_symbol(s):

    s = str(s).strip().upper()

    if s.endswith(".IS"):
        s = s[:-3]

    return s if s else None


def get_all_symbols():

    env = os.getenv(
        "BIST_SYMBOLS",
        ""
    ).strip()

    if env:

        return sorted(
            set(
                normalize_symbol(x)
                for x in env.split(",")
                if normalize_symbol(x)
            )
        )

    try:

        url = (
            "https://www.isyatirim.com.tr/"
            "_layouts/15/"
            "IsYatirim.MusteriKanalı/"
            "CommonData.aspx/GetHisseList"
        )

        r = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0"
            },
            timeout=15
        )

        if r.ok:

            data = r.json()

            symbols = [
                item["code"]
                for item in data
                if "code" in item
                and item["code"]
            ]

            if len(symbols) >= 300:

                log.info(
                    f"BIST evreni başarıyla çekildi: "
                    f"{len(symbols)} hisse taranacak."
                )

                return sorted(
                    set(symbols)
                )

    except Exception as e:

        log.warning(
            "Otomatik hisse listesi çekilemedi, "
            f"yedek listeye geçiliyor: {e}"
        )

    return sorted(
        set(FALLBACK_SYMBOLS)
    )


# ============================================================
# YAHOO
# ============================================================

def yahoo_symbol(symbol):

    if symbol in (
        "^XU100",
        "^GSPC",
        "^IXIC"
    ):
        return symbol

    return (
        symbol
        if symbol.endswith(".IS")
        else symbol + ".IS"
    )


def get_history(
    symbol,
    period="2y",
    interval="1d"
):

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

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            df.columns = (
                df.columns
                .get_level_values(0)
            )

        df.columns = [
            str(x).lower()
            for x in df.columns
        ]

        cols = [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]

        if not all(
            x in df.columns
            for x in cols
        ):
            return pd.DataFrame()

        return (
            df[cols]
            .dropna()
        )

    except Exception:

        return pd.DataFrame()


# ============================================================
# İNDİKATÖRLER
# ============================================================

def ema(s, n):

    return s.ewm(
        span=n,
        adjust=False
    ).mean()


def atr(df, n=14):

    prev = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],

            (
                df["high"] - prev
            ).abs(),

            (
                df["low"] - prev
            ).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / n,
        adjust=False
    ).mean()


def wavetrend(
    df,
    n1=10,
    n2=21
):

    ap = (
        df["high"]
        + df["low"]
        + df["close"]
    ) / 3

    esa = ap.ewm(
        span=n1,
        adjust=False
    ).mean()

    de = (
        ap - esa
    ).abs().ewm(
        span=n1,
        adjust=False
    ).mean()

    ci = (
        ap - esa
    ) / (
        0.015
        * de.replace(
            0,
            np.nan
        )
    )

    wt1 = ci.ewm(
        span=n2,
        adjust=False
    ).mean()

    wt2 = wt1.rolling(4).mean()

    return wt1, wt2


def to_4h(df):

    return (
        df.resample(
            "4h",
            offset="2h"
        )
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })
        .dropna()
    )


# ============================================================
# TEKNİK ANALİZ
# ============================================================

def technical_analysis(
    daily,
    four_hour
):

    if (
        len(daily) < 200
        or len(four_hour) < 40
    ):
        return None

    d = daily.copy()
    h = four_hour.copy()

    d["ema20"] = ema(
        d["close"],
        20
    )

    d["ema50"] = ema(
        d["close"],
        50
    )

    d["ema200"] = ema(
        d["close"],
        200
    )

    d["atr"] = atr(d)

    h["wt1"], h["wt2"] = wavetrend(h)

    a = d.iloc[-1]
    old = d.iloc[-11]

    x = h.iloc[-1]
    p = h.iloc[-2]

    below_middle = (
        x["wt1"] < 0
        and x["wt2"] < 0
    )

    bullish_cross = (
        p["wt1"] <= p["wt2"]
        and x["wt1"] > x["wt2"]
    )

    wt_rising = (
        x["wt1"] > p["wt1"]
    )

    green_candle = (
        x["close"] >= x["open"]
    )

    wt_signal = (
        below_middle
        and bullish_cross
        and wt_rising
        and green_candle
    )

    price_above_ema200 = (
        a["close"] > a["ema200"]
    )

    ema200_rising = (
        a["ema200"] > old["ema200"]
    )

    ema20_above_ema50 = (
        a["ema20"] > a["ema50"]
    )

    vol_avg = (
        d["volume"]
        .rolling(20)
        .mean()
        .iloc[-1]
    )

    volume_ok = (
        a["volume"]
        >= vol_avg * 0.80
    )

    # ========================================================
    # TEKNİK PUAN / 60
    # ========================================================

    technical_score = (

        (20 if price_above_ema200 else 0)

        +

        (10 if ema200_rising else 0)

        +

        (10 if ema20_above_ema50 else 0)

        +

        (15 if wt_signal else 0)

        +

        (5 if volume_ok else 0)
    )

    return {

        "technical_score":
            int(technical_score),

        "close":
            float(a["close"]),

        "ema200":
            float(a["ema200"]),

        "atr":
            float(a["atr"]),

        "wt_signal":
            bool(wt_signal)
    }


# ============================================================
# TEMEL ANALİZ / 40 PUAN
# ============================================================

def fundamental_score(symbol):

    try:

        url = (
            "https://www.isyatirim.com.tr/"
            "_layouts/15/"
            "IsYatirim.MaliTablolar/"
            f"MaliTablo.aspx/GetOranlar?hisse={symbol}"
        )

        headers = {
            "User-Agent":
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36",

            "X-Requested-With":
                "XMLHttpRequest"
        }

        r = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        if not r.ok:
            return 0

        data = r.json().get(
            "d",
            []
        )

        if not data:
            return 0

        metrics = {}

        for item in data:

            if (
                "KOD" in item
                and "DEGER" in item
                and item["DEGER"] is not None
            ):

                try:

                    val_str = (
                        str(item["DEGER"])
                        .replace(",", ".")
                    )

                    metrics[
                        item["KOD"]
                    ] = float(val_str)

                except ValueError:

                    continue

        roe = metrics.get("ROE")
        profit_growth = metrics.get(
            "NET_KAR_BUYUME"
        )
        revenue_growth = metrics.get(
            "SATIS_BUYUME"
        )
        debt_equity = metrics.get(
            "BORC_OZSERMAYE"
        )
        operating_margin = metrics.get(
            "FAALIYET_MARJI"
        )
        pe = metrics.get("FK")
        pb = metrics.get("PDDD")

        score = 0

        # ROE / 8
        if roe is not None:
            score += (
                8 if roe >= 20
                else 5 if roe >= 10
                else 0
            )

        # Net kar büyümesi / 8
        if profit_growth is not None:
            score += (
                8 if profit_growth >= 20
                else 5 if profit_growth > 0
                else 0
            )

        # Satış büyümesi / 6
        if revenue_growth is not None:
            score += (
                6 if revenue_growth >= 15
                else 3 if revenue_growth > 0
                else 0
            )

        # Borç / özsermaye / 6
        if debt_equity is not None:
            score += (
                6 if debt_equity <= 100
                else 3 if debt_equity <= 200
                else 0
            )

        # Faaliyet marjı / 5
        if operating_margin is not None:
            score += (
                5 if operating_margin >= 15
                else 3 if operating_margin > 0
                else 0
            )

        # F/K / 4
        if pe is not None:
            score += (
                4 if 0 < pe <= 20
                else 2 if pe <= 35
                else 0
            )

        # PD/DD / 3
        if pb is not None:
            score += (
                3 if 0 < pb <= 3
                else 1 if pb <= 6
                else 0
            )

        return score

    except Exception:

        return 0


# ============================================================
# NİHAİ PUAN
# ============================================================

def final_score(
    technical,
    fundamental
):

    total = (
        technical
        + fundamental
    )

    if technical < 45:
        return 0

    if (
        total >= 90
        and technical >= 50
        and fundamental >= 32
    ):
        return 100

    if (
        total >= 82
        and technical >= 48
        and fundamental >= 28
    ):
        return 90

    if (
        total >= 72
        and technical >= 45
        and fundamental >= 24
    ):
        return 80

    if (
        total >= 60
        and technical >= 45
        and fundamental >= 18
    ):
        return 60

    return 0


# ============================================================
# TRADE PLAN
# ============================================================

def trade_plan(t):

    entry = t["close"]

    risk_distance = max(
        t["atr"] * 1.5,
        entry * 0.01
    )

    stop = (
        entry
        - risk_distance
    )

    if stop <= 0:
        return None

    target_by_pct = (
        entry
        * (
            1
            + MIN_TARGET_PCT / 100.0
        )
    )

    target_by_rr = (
        entry
        + risk_distance * MIN_RR
    )

    target = max(
        target_by_pct,
        target_by_rr
    )

    target_pct = (
        target / entry - 1
    ) * 100

    rr = (
        target - entry
    ) / risk_distance

    if (
        target_pct < MIN_TARGET_PCT
        or rr < MIN_RR
    ):
        return None

    return {

        "entry": entry,

        "target": target,

        "stop": stop,

        "target_pct": target_pct,

        "rr": rr
    }


# ============================================================
# PİYASA REJİMİ
# ============================================================

def market_regime():

    df = get_history(
        "^XU100",
        "2y",
        "1d"
    )

    if (
        df.empty
        or len(df) < 200
    ):

        return {
            "value": None,
            "ema200": None,
            "regime": "BİLİNMİYOR"
        }

    e200 = ema(
        df["close"],
        200
    )

    value = float(
        df["close"].iloc[-1]
    )

    ema_value = float(
        e200.iloc[-1]
    )

    return {

        "value": value,

        "ema200": ema_value,

        "regime":
            "YÜKSELİŞ"
            if value > ema_value
            else "DÜŞÜŞ"
    }


# ============================================================
# AY KEY
# ============================================================

def month_key(dt=None):

    if dt is None:
        dt = datetime.now(TZ)

    return (
        f"{dt.year:04d}-"
        f"{dt.month:02d}"
    )


# ============================================================
# SEMBOL ANALİZİ
# ============================================================

def analyze_symbol(symbol):

    daily = get_history(
        symbol,
        "2y",
        "1d"
    )

    if daily.empty:
        return None, False

    hourly = get_history(
        symbol,
        "60d",
        "1h"
    )

    if hourly.empty:
        return None, False

    four_hour = to_4h(hourly)

    technical = technical_analysis(
        daily,
        four_hour
    )

    if not technical:
        return None, False

    # EMA200 altında ise tamamen elenir
    if (
        technical["close"]
        <= technical["ema200"]
    ):
        return None, True

    # WaveTrend sinyali yoksa tamamen elenir
    if not technical["wt_signal"]:
        return None, True

    fundamental = fundamental_score(
        symbol
    )

    score = final_score(
        technical["technical_score"],
        fundamental
    )

    if score == 0:
        return None, True

    plan = trade_plan(
        technical
    )

    if not plan:
        return None, True

    return {

        "symbol": symbol,

        "score": score,

        "technical_score":
            technical["technical_score"],

        "fundamental_score":
            fundamental,

        **plan
    }, True


# ============================================================
# SİNYAL KAYDET
# ============================================================

def save_signal(x):

    now = datetime.now(TZ)

    c = get_db()

    exists = c.execute(
        """
        SELECT id
        FROM signals
        WHERE symbol=?
        AND signal_time LIKE ?
        """,

        (
            x["symbol"],
            now.strftime(
                "%Y-%m-%d"
            ) + "%"
        )
    ).fetchone()

    if not exists:

        c.execute(
            """
            INSERT INTO signals(
                symbol,
                signal_time,
                month_key,
                score,
                technical_score,
                fundamental_score,
                entry,
                target,
                stop,
                target_pct,
                rr
            )
            VALUES(
                ?,?,?,?,?,?,?,?,?,?,?
            )
            """,

            (
                x["symbol"],
                now.isoformat(),
                month_key(now),

                x["score"],

                x["technical_score"],

                x["fundamental_score"],

                x["entry"],

                x["target"],

                x["stop"],

                x["target_pct"],

                x["rr"]
            )
        )

        c.commit()
        c.close()

        return True

    c.close()

    return False


# ============================================================
# AÇIK SİNYALLERİ TAKİP ET
# ============================================================

def update_open_signals():

    c = get_db()

    rows = c.execute(
        """
        SELECT *
        FROM signals
        WHERE status='OPEN'
        """
    ).fetchall()

    for row in rows:

        try:

            signal_time = datetime.fromisoformat(
                row["signal_time"]
            )

            # Sinyalin üretildiği günün günlük mumunu
            # sonuç hesabına dahil etmiyoruz.
            #
            # Böylece 18:30'da üretilen sinyal,
            # o gün daha önce gerçekleşmiş TP/SL'yi
            # yanlışlıkla kendi sonucu olarak almaz.

            df = get_history(
                row["symbol"],
                "10d",
                "1d"
            )

            if df.empty:
                continue

            # Tarih indexini timezone'suz hale getir
            # karşılaştırmayı güvenli yap.
            try:

                if df.index.tz is not None:

                    df_index = df.index.tz_convert(
                        TZ
                    ).tz_localize(None)

                else:

                    df_index = df.index

                signal_date = (
                    signal_time
                    .astimezone(TZ)
                    .replace(
                        tzinfo=None
                    )
                    .date()
                )

                valid_mask = [
                    x.date() > signal_date
                    for x in df_index
                ]

                df = df.loc[valid_mask]

            except Exception:

                # Tarih filtrelemesi başarısız olursa
                # yanlış sonuç üretmemek için geç.
                continue

            if df.empty:
                continue

            high = float(
                df["high"].max()
            )

            low = float(
                df["low"].min()
            )

            hit_target = (
                high >= row["target"]
            )

            hit_stop = (
                low <= row["stop"]
            )

            if hit_stop:

                status = "STOP"

                exit_price = row["stop"]

            elif hit_target:

                status = "TARGET"

                exit_price = row["target"]

            else:

                continue

            c.execute(
                """
                UPDATE signals
                SET status=?,
                    exit_price=?,
                    exit_time=?
                WHERE id=?
                """,

                (
                    status,
                    exit_price,
                    datetime.now(TZ).isoformat(),
                    row["id"]
                )
            )

            icon = (
                "🎯"
                if status == "TARGET"
                else "🛑"
            )

            telegram_send(
                f"{icon} SİNYAL SONUCU\n"
                f"{row['symbol']} | "
                f"{row['score']} PUAN\n"
                f"Sonuç: {status}\n"
                f"Çıkış: {exit_price:.2f}"
            )

        except Exception:

            log.exception(
                "Sinyal takip hatası: "
                f"{row['symbol']}"
            )

    c.commit()
    c.close()


# ============================================================
# AYLIK RAPOR
# ============================================================

def generate_monthly_report(
    current_m_key
):

    c = get_db()

    signals = c.execute(
        """
        SELECT *
        FROM signals
        WHERE month_key=?
        ORDER BY signal_time ASC
        """,

        (current_m_key,)
    ).fetchall()

    c.close()

    try:

        report_year = int(
            current_m_key[:4]
        )

        report_month = int(
            current_m_key[5:7]
        )

    except Exception:

        now = datetime.now(TZ)

        report_year = now.year
        report_month = now.month

    month_name = MONTHS.get(
        report_month,
        ""
    )

    # --------------------------------------------------------
    # Hiç sinyal yok
    # --------------------------------------------------------

    if not signals:

        return (
            f"📊 BIST BOTU - "
            f"{month_name.upper()} "
            f"{report_year} AYLIK RAPOR\n"
            f"━━━━━━━━━━━━━━━━━━\n\n"
            f"Bu ay için sinyal kaydı bulunamadı."
        )

    # --------------------------------------------------------
    # Genel sayılar
    # --------------------------------------------------------

    total = len(signals)

    target_cnt = sum(
        1
        for s in signals
        if s["status"] == "TARGET"
    )

    stop_cnt = sum(
        1
        for s in signals
        if s["status"] == "STOP"
    )

    open_cnt = sum(
        1
        for s in signals
        if s["status"] == "OPEN"
    )

    completed = (
        target_cnt
        + stop_cnt
    )

    win_rate = (
        target_cnt
        / completed
        * 100
        if completed > 0
        else 0
    )

    # --------------------------------------------------------
    # Puan grupları
    # --------------------------------------------------------

    score_lines = []

    for score in (
        100,
        90,
        80,
        60
    ):

        group = [
            s
            for s in signals
            if s["score"] == score
        ]

        group_total = len(group)

        group_target = sum(
            1
            for s in group
            if s["status"] == "TARGET"
        )

        group_stop = sum(
            1
            for s in group
            if s["status"] == "STOP"
        )

        group_open = sum(
            1
            for s in group
            if s["status"] == "OPEN"
        )

        group_completed = (
            group_target
            + group_stop
        )

        group_win = (
            group_target
            / group_completed
            * 100
            if group_completed > 0
            else 0
        )

        score_lines.append(
            f"{score} PUAN → "
            f"{group_total} sinyal | "
            f"🎯 {group_target} | "
            f"🛑 {group_stop} | "
            f"⏳ {group_open} | "
            f"Başarı %{group_win:.1f}"
        )

    # --------------------------------------------------------
    # Açık pozisyon listesi
    # --------------------------------------------------------

    open_lines = []

    for s in signals:

        if s["status"] != "OPEN":
            continue

        open_lines.append(
            f"⏳ {s['symbol']} "
            f"| {s['score']} P "
            f"| Giriş {s['entry']:.2f} "
            f"| Hedef {s['target']:.2f} "
            f"| Stop {s['stop']:.2f}"
        )

    # Telegram mesajını çok şişirmemek için
    # açık pozisyonların ilk 20 tanesini göster.
    if len(open_lines) > 20:

        open_lines = (
            open_lines[:20]
            + [
                f"... "
                f"{len(open_lines) - 20} "
                f"açık sinyal daha var."
            ]
        )

    # --------------------------------------------------------
    # RAPOR
    # --------------------------------------------------------

    lines = [

        f"📊 BIST BOTU - "
        f"{month_name.upper()} "
        f"{report_year} AYLIK KANIT KAPANIŞ RAPORU",

        "━━━━━━━━━━━━━━━━━━",

        f"🗓️ Raporlanan Ay: "
        f"{month_name} {report_year}",

        "📌 Ayın son işlem günü raporu",

        "",

        "📈 GENEL İSTATİSTİK",

        f"🎯 Toplam Sinyal: {total}",

        f"✅ Hedef: {target_cnt}",

        f"🛑 Stop: {stop_cnt}",

        f"⏳ Açık: {open_cnt}",

        f"📊 Tamamlanan: {completed}",

        f"🏆 Başarı Oranı: %{win_rate:.1f}",

        "",

        "🏅 PUAN GRUPLARI",

    ]

    lines.extend(score_lines)

    lines.append("")

    lines.append(
        "⏳ AY SONUNDA AÇIK POZİSYONLAR"
    )

    if open_lines:

        lines.extend(open_lines)

    else:

        lines.append(
            "Yok."
        )

    lines.append("")

    lines.append(
        "ℹ️ Yeni ayda istatistikler "
        "month_key üzerinden otomatik "
        "olarak sıfırdan başlar."
    )

    return "\n".join(lines)


# ============================================================
# AYLIK RAPOR KONTROLÜ
# ============================================================

def check_monthly_report():

    now_dt = datetime.now(TZ)

    current_month = month_key(
        now_dt
    )

    last_reported = get_meta(
        "last_monthly_report",
        ""
    )

    # --------------------------------------------------------
    # Sadece ayın son işlem günü
    # --------------------------------------------------------

    if not is_last_trade_day_of_month(
        now_dt.date()
    ):
        return

    # --------------------------------------------------------
    # Aynı ay raporu ikinci kez gönderilmesin
    # --------------------------------------------------------

    if last_reported == current_month:
        return

    report_text = generate_monthly_report(
        current_month
    )

    success = telegram_send(
        report_text
    )

    # Telegram gönderimi başarısızsa
    # rapor gönderilmiş kabul edilmiyor.
    #
    # Böylece sonraki kontrol döngüsünde
    # tekrar denenebilir.

    if success:

        set_meta(
            "last_monthly_report",
            current_month
        )

        log.info(
            "Ayın son iş günü aylık raporu "
            f"gönderildi: {current_month}"
        )

    else:

        log.warning(
            "Aylık rapor gönderilemedi. "
            "Tekrar denenecek."
        )


# ============================================================
# PİYASA TARAMASI
# ============================================================

def scan_market():

    if not scan_lock.acquire(
        blocking=False
    ):
        return

    try:

        # Önce açık sinyalleri kontrol et
        update_open_signals()

        # BIST evreni
        symbols = get_all_symbols()

        # Piyasa rejimi
        market = market_regime()

        candidates = []

        scanned_ok = 0

        log.info(
            f"Tarama başladı. "
            f"Evren: {len(symbols)}"
        )

        # ----------------------------------------------------
        # HİSSELERİ TARA
        # ----------------------------------------------------

        for symbol in symbols:

            try:

                result, ok = analyze_symbol(
                    symbol
                )

                if ok:
                    scanned_ok += 1

                if result:
                    candidates.append(
                        result
                    )

            except Exception:

                log.exception(
                    f"Analiz hatası: {symbol}"
                )

            time.sleep(0.05)

        # ----------------------------------------------------
        # SIRALA
        # ----------------------------------------------------

        candidates.sort(
            key=lambda x: (
                x["score"],
                x["target_pct"],
                x["rr"]
            ),
            reverse=True
        )

        # ----------------------------------------------------
        # SİNYALLERİ KAYDET
        # ----------------------------------------------------

        new_signals = sum(
            1
            for candidate in candidates
            if save_signal(candidate)
        )

        # ----------------------------------------------------
        # SCAN KAYDI
        # ----------------------------------------------------

        now = datetime.now(TZ)

        c = get_db()

        c.execute(
            """
            INSERT INTO scans(
                scan_time,
                month_key,
                total_symbols,
                scanned_ok,
                candidates,
                market_regime
            )
            VALUES(?,?,?,?,?,?)
            """,

            (
                now.isoformat(),
                month_key(now),
                len(symbols),
                scanned_ok,
                len(candidates),
                market["regime"]
            )
        )

        c.commit()
        c.close()

        # ----------------------------------------------------
        # GÜNLÜK TARAMA RAPORU
        # ----------------------------------------------------

        telegram_long(
            scan_report(
                candidates,
                market,
                len(symbols),
                scanned_ok
            )
        )

        # ----------------------------------------------------
        # AY SONU RAPORU
        # ----------------------------------------------------

        check_monthly_report()

        log.info(
            f"Tarama bitti. "
            f"Aday={len(candidates)} "
            f"Yeni sinyal={new_signals}"
        )

    finally:

        scan_lock.release()


# ============================================================
# GÜNLÜK TARAMA RAPORU
# ============================================================

def scan_report(
    candidates,
    market,
    total,
    scanned_ok
):

    now = datetime.now(TZ)

    if market["value"] is not None:

        bist_value = (
            f"{market['value']:.2f}"
        )

    else:

        bist_value = "N/A"

    lines = [

        "🔍 BIST TARAMA RAPORU",

        now.strftime(
            "%d.%m.%Y %H:%M"
        ),

        "━━━━━━━━━━━━━━━━━━",

        f"Evren: {total} şirket "
        f"| Verisi Alınan: {scanned_ok}",

        f"BIST100: {bist_value} "
        f"(Piyasa: {market['regime']})",

        f"Bulunan Aday: "
        f"{len(candidates)}",

        ""
    ]

    if not candidates:

        lines.append(
            "❌ Bugün strateji şartlarını "
            "sağlayan hisse bulunamadı."
        )

        return "\n".join(lines)

    for score in (
        100,
        90,
        80,
        60
    ):

        group = [
            x
            for x in candidates
            if x["score"] == score
        ]

        if not group:
            continue

        lines.append(
            f"━━ {score} PUAN "
            f"| {len(group)} HİSSE ━━"
        )

        for x in group:

            lines.extend(
                [

                    f"📌 {x['symbol']}",

                    f"Teknik: "
                    f"{x['technical_score']}/60 "
                    f"| Temel: "
                    f"{x['fundamental_score']}/40",

                    f"Giriş: "
                    f"{x['entry']:.2f} "
                    f"| Hedef: "
                    f"{x['target']:.2f} "
                    f"(+%{x['target_pct']:.1f}) "
                    f"| Stop: "
                    f"{x['stop']:.2f}",

                    f"R/R: "
                    f"1:{x['rr']:.2f}\n"
                ]
            )

    return "\n".join(lines)


# ============================================================
# FLASK
# ============================================================

@app.get("/")
def home():

    return (
        "BIST Tarama Botu Aktif."
    )


# ============================================================
# GENEL RAPOR
# ============================================================

@app.get("/report")
def get_report():

    c = get_db()

    total = c.execute(
        "SELECT COUNT(*) FROM signals"
    ).fetchone()[0]

    target_cnt = c.execute(
        """
        SELECT COUNT(*)
        FROM signals
        WHERE status='TARGET'
        """
    ).fetchone()[0]

    stop_cnt = c.execute(
        """
        SELECT COUNT(*)
        FROM signals
        WHERE status='STOP'
        """
    ).fetchone()[0]

    open_cnt = c.execute(
        """
        SELECT COUNT(*)
        FROM signals
        WHERE status='OPEN'
        """
    ).fetchone()[0]

    c.close()

    completed = (
        target_cnt
        + stop_cnt
    )

    win_rate = (
        target_cnt
        / completed
        * 100
        if completed > 0
        else 0
    )

    return jsonify({

        "total_signals":
            total,

        "target_count":
            target_cnt,

        "stop_count":
            stop_cnt,

        "open_count":
            open_cnt,

        "completed_count":
            completed,

        "win_rate_pct":
            round(
                win_rate,
                2
            )
    })


# ============================================================
# MANUEL TARAMA
# ============================================================

@app.post("/scan")
def manual_scan():

    threading.Thread(
        target=scan_market,
        daemon=True
    ).start()

    return jsonify(
        {
            "ok": True,
            "message":
                "Tarama başlatıldı."
        }
    )


# ============================================================
# SCHEDULER
# ============================================================

def scheduler():

    last_scan_date = ""

    while True:

        try:

            now = datetime.now(TZ)

            today_date = now.date()

            today_str = (
                today_date.strftime(
                    "%Y-%m-%d"
                )
            )

            # ------------------------------------------------
            # İŞ GÜNÜ + 18:30
            # ------------------------------------------------

            if (
                is_trade_day(today_date)

                and now.hour == SCAN_HOUR

                and now.minute == SCAN_MINUTE

                and last_scan_date != today_str
            ):

                last_scan_date = today_str

                log.info(
                    "Borsa açık gününde "
                    f"tarama başlatılıyor: "
                    f"{today_str}"
                )

                threading.Thread(
                    target=scan_market,
                    daemon=True
                ).start()

            time.sleep(20)

        except Exception:

            log.exception(
                "Scheduler hatası"
            )

            time.sleep(30)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    log.info(
        "BIST Tarama Botu başlatılıyor..."
    )

    threading.Thread(
        target=scheduler,
        daemon=True
    ).start()

    app.run(
        host="0.0.0.0",
        port=PORT,
        threaded=True
    )

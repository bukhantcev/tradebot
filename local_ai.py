# local_ai.py — FULL REWRITE (stateful, fast flat detection, anti-chase, spike guard, pullback→resume)

import os
import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable, Awaitable, Tuple

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

# ============================ ENV / CONFIG ============================
load_dotenv()

POLL_TICK_MS = int(os.getenv("POLL_TICK_MS", "1000"))
BOOTSTRAP_HOURS = int(os.getenv("BOOTSTRAP_HOURS", "5"))
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
CATEGORY = "linear"

# базовые
SL_TICKS_DEFAULT = int(os.getenv("SL_TICKS", "6000"))       # дефолт если решалка не дала иного
MIN_SL_TICKS     = int(os.getenv("MIN_SL_TICKS", "3000"))   # кламп минимальной дистанции SL
MIN_TP_TICKS     = int(os.getenv("MIN_TP_TICKS", "2500"))   # кламп минимальной дистанции TP (по требованию)
HOLD_BODY_TICKS  = int(os.getenv("HOLD_BODY_TICKS", "0"))   # если прошлое тело слишком мало — шум → HOLD

# EMA/ADX/VWAP
EMA_FAST = int(os.getenv("EMA_FAST", "20"))
EMA_MID  = int(os.getenv("EMA_MID",  "50"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "200"))

ADX_MIN     = float(os.getenv("ADX_MIN", "25"))   # минимальный ADX для «строгого» тренда
ADX_STRONG  = float(os.getenv("ADX_STRONG", "38"))# сильный ADX — можно раньше разрешить тренд
DMI_PERIOD  = int(os.getenv("DMI_PERIOD", "14"))

# моментум (по телам свечей)
MOM_WIN         = int(os.getenv("MOM_WINDOW", "10"))
MOM_GREEN_MIN   = int(os.getenv("MOM_GREEN_MIN", "6"))
MOM_RED_MIN     = int(os.getenv("MOM_RED_MIN",   "6"))

# быстрый каскадный тренд (когда рынок «течёт»)
CASCADE_TREND_ADX          = float(os.getenv("CASCADE_TREND_ADX", "40"))
CASCADE_WIN                = int(os.getenv("CASCADE_WIN", "8"))
CASCADE_MIN_CONSEC         = int(os.getenv("CASCADE_MIN_CONSEC", "3"))
CASCADE_MIN_SLOPE20_TICKS  = int(os.getenv("CASCADE_MIN_SLOPE20_TICKS", "1200"))

# real-trend валидатор (чтобы не ловить одиночные импульсы)
REAL_TREND_MIN_SLOPE50_TICKS  = int(os.getenv("REAL_TREND_MIN_SLOPE50_TICKS", "1200"))
REAL_TREND_MIN_EMA_GAP_TICKS  = int(os.getenv("REAL_TREND_MIN_EMA_GAP_TICKS", "2500"))
REAL_TREND_MIN_BODY_SUM_TICKS = int(os.getenv("REAL_TREND_MIN_BODY_SUM_TICKS", "6000"))
REAL_TREND_MIN_CONSEC         = int(os.getenv("REAL_TREND_MIN_CONSEC", "3"))

# анти-спайк (пережидаем «ракеты»)
POST_SPIKE_HOLD_BODY_TICKS = int(os.getenv("POST_SPIKE_HOLD_BODY_TICKS", "20000"))

# FLAT (классика)
FLAT_ADX_MAX           = float(os.getenv("FLAT_ADX_MAX", "22"))
FLAT_ATR_N             = int(os.getenv("FLAT_ATR_N", "14"))
FLAT_ATR_RATIO_MAX     = float(os.getenv("FLAT_ATR_RATIO_MAX", "0.6"))
FLAT_EMA_COMP_TICKS    = int(os.getenv("FLAT_EMA_COMP_TICKS", "1200"))
FLAT_VWAP_BAND_TICKS   = int(os.getenv("FLAT_VWAP_BAND_TICKS", "800"))

# EARLY-FLAT (ускоренный — чтобы бот быстрее говорил «это флэт»)
EARLY_FLAT_ADX_MAX         = float(os.getenv("EARLY_FLAT_ADX_MAX", "28"))
EARLY_FLAT_ATR_RATIO_MAX   = float(os.getenv("EARLY_FLAT_ATR_RATIO_MAX", "0.9"))
EARLY_FLAT_EMA_COMP_TICKS  = int(os.getenv("EARLY_FLAT_EMA_COMP_TICKS", "2000"))
EARLY_FLAT_VWAP_BAND_TICKS = int(os.getenv("EARLY_FLAT_VWAP_BAND_TICKS", "1500"))
EARLY_FLAT_MIN_BARS        = int(os.getenv("EARLY_FLAT_MIN_BARS", "2"))
EARLY_FLAT_BODY_TICKS_MAX  = int(os.getenv("EARLY_FLAT_BODY_TICKS_MAX", "4000"))

# анти-погоня (не покупаем прямо на экстремумах)
REQUIRE_PULLBACK              = os.getenv("REQUIRE_PULLBACK", "true").lower() == "true"
PULLBACK_MIN_TO_EMA_TICKS     = int(os.getenv("PULLBACK_MIN_TO_EMA_TICKS", "1200"))
PULLBACK_CONFIRM_BREAK        = os.getenv("PULLBACK_CONFIRM_BREAK", "true").lower() == "true"
CHASE_MAX_EMA20_DIST_TICKS    = int(os.getenv("CHASE_MAX_EMA20_DIST_TICKS", "3500"))
CHASE_MAX_VWAP_DIST_TICKS     = int(os.getenv("CHASE_MAX_VWAP_DIST_TICKS", "30000"))
CHASE_MAX_CONSEC_BARS         = int(os.getenv("CHASE_MAX_CONSEC_BARS", "5"))
CHASE_RSI_HIGH                = float(os.getenv("CHASE_RSI_HIGH", "74"))
CHASE_RSI_LOW                 = float(os.getenv("CHASE_RSI_LOW", "26"))
CHASE_SWING_LOOKBACK          = int(os.getenv("CHASE_SWING_LOOKBACK", "20"))

# guard toggles (to let bot actually trade; can turn back on later)
DISABLE_OVEREXTENSION_GUARD = os.getenv("DISABLE_OVEREXTENSION_GUARD", "true").lower() == "true"
DISABLE_PULLBACK_GUARD      = os.getenv("DISABLE_PULLBACK_GUARD", "true").lower() == "true"
DISABLE_SWING_RSI_GUARDS    = os.getenv("DISABLE_SWING_RSI_GUARDS", "true").lower() == "true"

# baseline-trend (облегчённый вход, когда тренд "и так виден")
TREND_BASELINE_ENABLE          = os.getenv("TREND_BASELINE_ENABLE", "true").lower() == "true"
TREND_BASELINE_ADX_MIN         = float(os.getenv("TREND_BASELINE_ADX_MIN", "22"))
TREND_BASELINE_MIN_SLOPE20_TCK = int(os.getenv("TREND_BASELINE_MIN_SLOPE20_TCK", "200"))
TREND_BASELINE_MIN_BODY_SUM_TK = int(os.getenv("TREND_BASELINE_MIN_BODY_SUM_TK", "1200"))
TREND_BASELINE_REQUIRE_VWAP    = os.getenv("TREND_BASELINE_REQUIRE_VWAP", "false").lower() == "true"
TREND_BASELINE_REQUIRE_DI      = os.getenv("TREND_BASELINE_REQUIRE_DI", "false").lower() == "true"

# ============================ TYPES ============================
class Regime(str):
    TREND = "trend"
    FLAT  = "flat"
    HOLD  = "hold"

class Side(str):
    BUY  = "Buy"
    SELL = "Sell"
    NONE = "None"

@dataclass
class Filters:
    tick_size: float
    qty_step: float
    min_qty: float

@dataclass
class Position:
    size: float
    side: Side
    avg_price: float

@dataclass
class MarketData:
    last_price: float
    filters: Filters
    kline_1m: List[List[Any]]
    position: Position
    balance_usdt: float

@dataclass
class Decision:
    regime: Regime
    side: Side
    sl_ticks: Optional[int]
    comment: str
    # клампы (на выход — чтобы мейн мог применить)
    tp_min_ticks: int = MIN_TP_TICKS
    sl_min_ticks: int = MIN_SL_TICKS

# ============================ BYBIT HELPERS ============================
def make_bybit() -> HTTP:
    testnet = os.getenv("BYBIT_TESTNET", "true").lower() == "true"
    api_key = os.getenv("BYBIT_API_KEY_TEST" if testnet else "BYBIT_API_KEY_MAIN")
    api_secret = os.getenv("BYBIT_SECRET_KEY_TEST" if testnet else "BYBIT_SECRET_KEY_MAIN")
    return HTTP(testnet=testnet, api_key=api_key, api_secret=api_secret)

def bybit_filters(http: HTTP, symbol: str) -> Filters:
    r = http.get_instruments_info(category=CATEGORY, symbol=symbol)
    it = r["result"]["list"][0]
    return Filters(
        tick_size=float(it["priceFilter"]["tickSize"]),
        qty_step=float(it["lotSizeFilter"]["qtyStep"]),
        min_qty=float(it["lotSizeFilter"]["minOrderQty"]),
    )

def bybit_last_price(http: HTTP, symbol: str) -> float:
    r = http.get_tickers(category=CATEGORY, symbol=symbol)
    return float(r["result"]["list"][0]["lastPrice"])

def bybit_kline_1m(http: HTTP, symbol: str, limit: int) -> List[List[Any]]:
    # Bybit v5 отдаёт последние первой — аккумулируем с разворотом
    out: List[List[Any]] = []
    need = limit
    while need > 0:
        batch = 200 if need > 200 else need
        rr = http.get_kline(category=CATEGORY, symbol=symbol, interval="1", limit=batch)
        lst = rr["result"]["list"]
        out = lst + out
        need -= batch
        if need > 0:
            time.sleep(0.15)
    return out

def bybit_wallet_usdt(http: HTTP) -> float:
    r = http.get_wallet_balance(accountType="UNIFIED")
    for coin in r["result"]["list"][0]["coin"]:
        if coin["coin"] == "USDT":
            return float(coin["walletBalance"])
    return 0.0

def bybit_position(http: HTTP, symbol: str) -> Position:
    rr = http.get_positions(category=CATEGORY, symbol=symbol)
    if not rr["result"]["list"]:
        return Position(0.0, Side.NONE, 0.0)
    p = rr["result"]["list"][0]
    size = float(p.get("size", 0) or 0)
    side = Side(p.get("side", "None")) if size > 0 else Side.NONE
    avg = float(p.get("avgPrice", 0) or 0)
    return Position(size, side, avg)

# ============================ DATA / INDICATORS ============================
def kline_to_df(kline: List[List[Any]]) -> pd.DataFrame:
    df = pd.DataFrame(kline, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
    for col in ["open","high","low","close","volume","turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.sort_values("ts")
    return df

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low  = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low  - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False, min_periods=n).mean()

def dmi_adx(df: pd.DataFrame, period: int = DMI_PERIOD) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Возвращает (+DI, -DI, ADX) такой же длины, как df (n),
    c NaN на первом баре (где нет предыдущего закрытия).
    Это устраняет рассинхрон 299 vs 300 и т.п.
    """
    high = df["high"].astype(float).values
    low  = df["low"].astype(float).values
    close = df["close"].astype(float).values
    n = len(df)
    if n < period + 2:
        s = pd.Series([np.nan] * n, index=df.index)
        return s, s, s

    # True Range (TR) длины n, но первый элемент 0 (нет prev close)
    tr = np.zeros(n)
    tr[1:] = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:]  - close[:-1]),
    ])

    # Directional Movement (DM) длины n-1 → потом паддим до n
    up   = high[1:] - high[:-1]
    down = low[:-1] - low[1:]
    plus_dm_raw  = np.where((up > 0) & (up > down), up, 0.0)
    minus_dm_raw = np.where((down > 0) & (down > up), down, 0.0)

    def _ema(arr, p):
        alpha = 1.0 / p
        out = np.zeros_like(arr, dtype=float)
        out[0] = arr[0]
        for i in range(1, arr.shape[0]):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
        return out

    # EMA по TR (длины n), по DM (длины n-1)
    tr_s = _ema(tr, period)  # n
    plus_dm_s  = _ema(plus_dm_raw,  period)  # n-1
    minus_dm_s = _ema(minus_dm_raw, period)  # n-1

    # Превращаем DM в длину n (паддим NaN в начале)
    plus_dm_s_full  = np.empty(n, dtype=float);  plus_dm_s_full[:]  = np.nan; plus_dm_s_full[1:]  = plus_dm_s
    minus_dm_s_full = np.empty(n, dtype=float); minus_dm_s_full[:] = np.nan; minus_dm_s_full[1:] = minus_dm_s

    # DI длины n
    denom = tr_s + 1e-12
    plus_di  = 100.0 * (plus_dm_s_full  / denom)
    minus_di = 100.0 * (minus_dm_s_full / denom)

    # DX считаем с 1-го бара, затем EMA → длина n-1, паддим до n
    dx_raw = 100.0 * np.abs(plus_di[1:] - minus_di[1:]) / (plus_di[1:] + minus_di[1:] + 1e-12)
    adx_s  = _ema(dx_raw, period)  # n-1
    adx_full = np.empty(n, dtype=float); adx_full[:] = np.nan; adx_full[1:] = adx_s

    # Возвращаем Series длины n с тем же индексом, что и df
    idx = df.index
    return (
        pd.Series(plus_di,  index=idx),
        pd.Series(minus_di, index=idx),
        pd.Series(adx_full, index=idx),
    )

def vwap_series(df: pd.DataFrame) -> pd.Series:
    vol = df["volume"].astype(float)
    turn= df["turnover"].astype(float)
    cum_vol = vol.cumsum().replace(0, np.nan)
    cum_turn= turn.cumsum()
    return cum_turn / cum_vol

def momentum_counts(df: pd.DataFrame, window: int) -> Tuple[int, int]:
    last = df.tail(window)
    greens = int((last["close"] > last["open"]).sum())
    reds   = int((last["close"] < last["open"]).sum())
    return greens, reds

def body_edges_prev(df: pd.DataFrame) -> Dict[str, float]:
    prev = df.iloc[-2]
    o, c = float(prev["open"]), float(prev["close"])
    return {
        "prev_open": o, "prev_close": c,
        "prev_high": float(prev["high"]),
        "prev_low" : float(prev["low"]),
        "body_low" : min(o, c),
        "body_high": max(o, c),
    }

def ticks(distance_price: float, tick_size: float) -> float:
    if tick_size <= 0:
        return 0.0
    return distance_price / tick_size

# ============================ GUARDS / HELPERS ============================
def count_consecutive(df: pd.DataFrame, side: Side, window: int) -> int:
    last = df.tail(window)
    seq = (last["close"] > last["open"]) if side == Side.BUY else (last["close"] < last["open"])
    cnt = 0
    for v in seq.values[::-1]:
        if v: cnt += 1
        else: break
    return cnt

def is_real_trend(df: pd.DataFrame, f: Filters, *, side: Side, ema_fast: pd.Series, ema_mid: pd.Series) -> bool:
    tick = f.tick_size if f.tick_size > 0 else 1.0
    if len(df) < 55:
        return False
    slope50_ticks = abs(float(ema_mid.iloc[-1] - ema_mid.iloc[-5])) / tick
    if slope50_ticks < REAL_TREND_MIN_SLOPE50_TICKS:
        return False
    gap_20_50_ticks = abs(float(ema_fast.iloc[-1] - ema_mid.iloc[-1])) / tick
    if gap_20_50_ticks < REAL_TREND_MIN_EMA_GAP_TICKS:
        return False
    lastN = df.tail(5)
    body_sum_ticks = (lastN["close"] - lastN["open"]).abs().astype(float).sum() / tick
    if body_sum_ticks < REAL_TREND_MIN_BODY_SUM_TICKS:
        return False
    consec = count_consecutive(df, side, window=5)
    if consec < REAL_TREND_MIN_CONSEC:
        return False
    return True

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    s = series.astype(float)
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_dn = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_dn + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def _swing_levels(df: pd.DataFrame, lookback: int) -> Tuple[float, float]:
    lb = min(max(2, lookback), len(df))
    last = df.tail(lb)
    return float(last["high"].max()), float(last["low"].min())

def _consec_in_dir(df: pd.DataFrame, side: Side, max_bars: int = 20) -> int:
    last = df.tail(max_bars)
    seq = (last["close"] > last["open"]) if side == Side.BUY else (last["close"] < last["open"])
    cnt = 0
    for v in seq.values[::-1]:
        if v: cnt += 1
        else: break
    return cnt

def _has_pullback_then_resume(df: pd.DataFrame, side: Side, ema20: pd.Series, tick: float) -> bool:
    if len(df) < 6:
        return False
    last6 = df.tail(6)
    e20 = float(ema20.iloc[-1])
    # было ли касание/сближение к EMA20 в последних барах
    near_ema = False
    for i in range(2, min(6, len(last6)+1)):
        px_low  = float(last6.iloc[-i]["low"])
        px_high = float(last6.iloc[-i]["high"])
        dist_ticks = abs((px_low if side == Side.BUY else px_high) - e20) / (tick if tick>0 else 1.0)
        if dist_ticks <= PULLBACK_MIN_TO_EMA_TICKS:
            near_ema = True
            break
    if not near_ema:
        return False
    # подтверждение возобновления движения
    prev_high = float(last6.iloc[-2]["high"])
    prev_low  = float(last6.iloc[-2]["low"])
    c = float(last6.iloc[-1]["close"])
    if PULLBACK_CONFIRM_BREAK:
        if side == Side.BUY and not (c > prev_high):
            return False
        if side == Side.SELL and not (c < prev_low):
            return False
    return True

def _is_overextended(df: pd.DataFrame, side: Side, tick: float, ema20: pd.Series, vwap_s: pd.Series) -> Tuple[bool, str]:
    if DISABLE_OVEREXTENSION_GUARD:
        return False, "disabled"
    c = float(df.iloc[-1]["close"])
    e20 = float(ema20.iloc[-1]) if not np.isnan(ema20.iloc[-1]) else c
    vw  = float(vwap_s.iloc[-1]) if not np.isnan(vwap_s.iloc[-1]) else c
    dist_ema20_ticks = abs(c - e20) / (tick if tick>0 else 1.0)
    dist_vwap_ticks  = abs(c - vw)  / (tick if tick>0 else 1.0)
    consec = _consec_in_dir(df, side, max_bars=20)
    r = float(rsi(df["close"], 14).iloc[-1])
    rsi_extreme = (r >= CHASE_RSI_HIGH and side == Side.BUY) or (r <= CHASE_RSI_LOW and side == Side.SELL)
    swing_h, swing_l = _swing_levels(df, CHASE_SWING_LOOKBACK)
    at_extreme = (side == Side.BUY and c >= swing_h) or (side == Side.SELL and c <= swing_l)
    if DISABLE_SWING_RSI_GUARDS:
        rsi_extreme = False
        at_extreme = False

    if dist_ema20_ticks >= CHASE_MAX_EMA20_DIST_TICKS:
        return True, f"far from EMA20: {dist_ema20_ticks:.0f}t ≥ {CHASE_MAX_EMA20_DIST_TICKS}t"
    if dist_vwap_ticks >= CHASE_MAX_VWAP_DIST_TICKS:
        return True, f"far from VWAP: {dist_vwap_ticks:.0f}t ≥ {CHASE_MAX_VWAP_DIST_TICKS}t"
    if consec >= CHASE_MAX_CONSEC_BARS:
        return True, f"consecutive bars {consec} ≥ {CHASE_MAX_CONSEC_BARS}"
    if rsi_extreme:
        return True, f"RSI={r:.1f} extreme"
    if at_extreme:
        return True, "at swing extreme"
    return False, ""

def apply_trend_entry_guards(decision: Dict[str, Any], df: pd.DataFrame, md: MarketData) -> Dict[str, Any]:
    """Не даём входить в тренд, если это погоня/экстремум; требуем pullback→resume."""
    if decision.get("regime") != Regime.TREND or decision.get("side") not in (Side.BUY, Side.SELL):
        return decision

    side = decision["side"]
    tick = md.filters.tick_size if md.filters.tick_size > 0 else 1.0

    # Если все защитные фильтры выключены — пропускаем без ограничений
    if DISABLE_OVEREXTENSION_GUARD and (DISABLE_PULLBACK_GUARD or not REQUIRE_PULLBACK):
        return decision

    ema20 = ema(df["close"].astype(float), EMA_FAST)
    vwap_s = vwap_series(df)

    # Overextension guard
    overext, reason = _is_overextended(df, side, tick, ema20, vwap_s)
    if not DISABLE_OVEREXTENSION_GUARD and overext:
        return {
            "regime": Regime.HOLD, "side": Side.NONE,
            "sl_ticks": decision.get("sl_ticks", SL_TICKS_DEFAULT),
            "comment": f"BLOCK TREND {side}: overextended ({reason})"
        }

    # Pullback guard
    if (not DISABLE_PULLBACK_GUARD) and REQUIRE_PULLBACK and (not _has_pullback_then_resume(df, side, ema20, tick)):
        return {
            "regime": Regime.HOLD, "side": Side.NONE,
            "sl_ticks": decision.get("sl_ticks", SL_TICKS_DEFAULT),
            "comment": f"BLOCK TREND {side}: need pullback→resume near EMA20 (≥{PULLBACK_MIN_TO_EMA_TICKS}t)"
        }

    return decision

# ============================ DECIDER CORE ============================
def decide(symbol: str, df: pd.DataFrame, md: MarketData) -> Dict[str, Any]:
    """
    Возвращает JSON-решение:
      {"regime": Regime.*, "side": Side.*, "sl_ticks": int|None, "comment": str}
    Внешний контракт полностью совместим со «старым».
    """
    if df is None or len(df) < 210:
        return {"regime": Regime.HOLD, "side": Side.NONE, "sl_ticks": SL_TICKS_DEFAULT, "comment": "Not enough history"}

    f = md.filters
    tick = f.tick_size if f.tick_size > 0 else 1.0
    edges = body_edges_prev(df)

    # анти-шум по телу предыдущей свечи
    prev_body_ticks = abs(edges["prev_close"] - edges["prev_open"]) / tick
    if prev_body_ticks < HOLD_BODY_TICKS:
        return {"regime": Regime.HOLD, "side": Side.NONE, "sl_ticks": SL_TICKS_DEFAULT,
                "comment": f"Prev body {prev_body_ticks:.0f} < {HOLD_BODY_TICKS} ticks"}

    # анти-спайк по текущей свече
    last_body_ticks = abs(float(df.iloc[-1]["close"]) - float(df.iloc[-1]["open"])) / tick
    prev_body_ticks_now = abs(float(df.iloc[-2]["close"]) - float(df.iloc[-2]["open"])) / tick if len(df) >= 2 else 0.0
    if last_body_ticks >= POST_SPIKE_HOLD_BODY_TICKS and prev_body_ticks_now < POST_SPIKE_HOLD_BODY_TICKS:
        return {"regime": Regime.HOLD, "side": Side.NONE, "sl_ticks": SL_TICKS_DEFAULT,
                "comment": f"Post-spike hold: {last_body_ticks:.0f}t ≥ {POST_SPIKE_HOLD_BODY_TICKS}t"}

    # индикаторы
    close = df["close"].astype(float)
    ema_fast_s = ema(close, EMA_FAST)
    ema_mid_s  = ema(close, EMA_MID)
    ema_slow_s = ema(close, EMA_SLOW)
    plus_di, minus_di, adx = dmi_adx(df, period=DMI_PERIOD)
    vwap_s = vwap_series(df)

    c  = float(close.iloc[-1])
    ef = float(ema_fast_s.iloc[-1]) if not np.isnan(ema_fast_s.iloc[-1]) else c
    em = float(ema_mid_s.iloc[-1])  if not np.isnan(ema_mid_s.iloc[-1])  else c
    es = float(ema_slow_s.iloc[-1]) if not np.isnan(ema_slow_s.iloc[-1]) else c
    a  = float(adx.iloc[-1])        if not np.isnan(adx.iloc[-1])        else 0.0
    pdi= float(plus_di.iloc[-1])    if not np.isnan(plus_di.iloc[-1])    else 0.0
    ndi= float(minus_di.iloc[-1])   if not np.isnan(minus_di.iloc[-1])   else 0.0
    vw = float(vwap_s.iloc[-1])     if not np.isnan(vwap_s.iloc[-1])     else c

    slope20 = float(ema_fast_s.iloc[-1] - ema_fast_s.iloc[-5]) if len(df) >= 5 else 0.0
    slope50 = float(ema_mid_s.iloc[-1]  - ema_mid_s.iloc[-5])  if len(df) >= 5 else 0.0

    greens, reds = momentum_counts(df, MOM_WIN)
    long_stack  = (c > es) and (ef > em > es)
    short_stack = (c < es) and (ef < em < es)
    above_vwap  = c > vw
    below_vwap  = c < vw

    ema20_200_ticks = abs(ef - es) / tick
    ema50_200_ticks = abs(em - es) / tick
    vwap_dev_ticks  = abs(c - vw)  / tick

    # =================== Быстрый каскадный тренд ===================
    slope20_ticks = abs(float(ema_fast_s.iloc[-1] - ema_fast_s.iloc[-5])) / tick if len(df) >= 5 else 0.0
    consec_reds   = count_consecutive(df, Side.SELL, CASCADE_WIN)
    consec_greens = count_consecutive(df, Side.BUY,  CASCADE_WIN)

    if a >= CASCADE_TREND_ADX and slope20 < 0 and slope20_ticks >= CASCADE_MIN_SLOPE20_TICKS:
        if reds >= MOM_RED_MIN or consec_reds >= CASCADE_MIN_CONSEC:
            dec = {"regime": Regime.TREND, "side": Side.SELL, "sl_ticks": SL_TICKS_DEFAULT,
                   "comment": f"CascadeDown: ADX={a:.1f}, slope20={slope20_ticks:.0f}t, reds={reds}/{MOM_WIN}, consec={consec_reds}/{CASCADE_WIN}"}
            return apply_trend_entry_guards(dec, df, md)

    if a >= CASCADE_TREND_ADX and slope20 > 0 and slope20_ticks >= CASCADE_MIN_SLOPE20_TICKS:
        if greens >= MOM_GREEN_MIN or consec_greens >= CASCADE_MIN_CONSEC:
            dec = {"regime": Regime.TREND, "side": Side.BUY, "sl_ticks": SL_TICKS_DEFAULT,
                   "comment": f"CascadeUp: ADX={a:.1f}, slope20={slope20_ticks:.0f}t, greens={greens}/{MOM_WIN}, consec={consec_greens}/{CASCADE_WIN}"}
            return apply_trend_entry_guards(dec, df, md)

    # =================== БЫСТРЫЙ BASELINE-ТРЕНД (без тяжёлого валидатора) ===================
    if TREND_BASELINE_ENABLE:
        # накопленная "энергия" за 5 баров, и быстрый наклон
        last5 = df.tail(5)
        body_sum_ticks5 = (last5["close"] - last5["open"]).abs().astype(float).sum() / tick
        slope20_ticks_bl = abs(float(ema_fast_s.iloc[-1] - ema_fast_s.iloc[-5])) / tick if len(df) >= 5 else 0.0

        cond_common = (
            a >= TREND_BASELINE_ADX_MIN and
            body_sum_ticks5 >= TREND_BASELINE_MIN_BODY_SUM_TK and
            slope20_ticks_bl >= TREND_BASELINE_MIN_SLOPE20_TCK
        )

        # вверх
        cond_up = cond_common and (ef > em > es) and (slope20 > 0)
        if TREND_BASELINE_REQUIRE_VWAP:
            cond_up = cond_up and (c > vw)
        if TREND_BASELINE_REQUIRE_DI:
            cond_up = cond_up and (pdi >= ndi)

        if cond_up:
            dec_bl = {"regime": Regime.TREND, "side": Side.BUY, "sl_ticks": SL_TICKS_DEFAULT,
                      "comment": f"BaselineTrendUp: ADX={a:.1f}, slope20={slope20_ticks_bl:.0f}t, bodyΣ5={body_sum_ticks5:.0f}t"}
            return apply_trend_entry_guards(dec_bl, df, md)

        # вниз
        cond_dn = cond_common and (ef < em < es) and (slope20 < 0)
        if TREND_BASELINE_REQUIRE_VWAP:
            cond_dn = cond_dn and (c < vw)
        if TREND_BASELINE_REQUIRE_DI:
            cond_dn = cond_dn and (ndi >= pdi)

        if cond_dn:
            dec_bl = {"regime": Regime.TREND, "side": Side.SELL, "sl_ticks": SL_TICKS_DEFAULT,
                      "comment": f"BaselineTrendDown: ADX={a:.1f}, slope20={slope20_ticks_bl:.0f}t, bodyΣ5={body_sum_ticks5:.0f}t"}
            return apply_trend_entry_guards(dec_bl, df, md)

    # =================== Строгий тренд EMA/VWAP + real-trend ===================
    if long_stack and slope20 > 0 and a >= ADX_MIN and pdi > ndi and above_vwap and greens >= MOM_GREEN_MIN:
        if is_real_trend(df, f, side=Side.BUY, ema_fast=ema_fast_s, ema_mid=ema_mid_s):
            dec = {"regime": Regime.TREND, "side": Side.BUY, "sl_ticks": SL_TICKS_DEFAULT,
                   "comment": f"TrendUp: EMA stack + real, ADX={a:.1f}, +DI>{ndi:.1f}, VWAP, greens={greens}/{MOM_WIN}"}
            return apply_trend_entry_guards(dec, df, md)

    if short_stack and slope20 < 0 and a >= ADX_MIN and ndi > pdi and below_vwap and reds >= MOM_RED_MIN:
        if is_real_trend(df, f, side=Side.SELL, ema_fast=ema_fast_s, ema_mid=ema_mid_s):
            dec = {"regime": Regime.TREND, "side": Side.SELL, "sl_ticks": SL_TICKS_DEFAULT,
                   "comment": f"TrendDown: EMA stack + real, ADX={a:.1f}, -DI>{pdi:.1f}, VWAP, reds={reds}/{MOM_WIN}"}
            return apply_trend_entry_guards(dec, df, md)

    # =================== Ранний тренд при сильном ADX ===================
    if a >= ADX_STRONG:
        if slope20 > 0 and slope50 > 0 and pdi > ndi and c > em and above_vwap:
            if is_real_trend(df, f, side=Side.BUY, ema_fast=ema_fast_s, ema_mid=ema_mid_s):
                dec = {"regime": Regime.TREND, "side": Side.BUY, "sl_ticks": SL_TICKS_DEFAULT,
                       "comment": f"EarlyTrendUp(real): ADX={a:.1f}, slope20>0, slope50>0, +DI>{ndi:.1f}, c>EMA50"}
                return apply_trend_entry_guards(dec, df, md)
        if slope20 < 0 and slope50 < 0 and ndi > pdi and c < em and below_vwap:
            if is_real_trend(df, f, side=Side.SELL, ema_fast=ema_fast_s, ema_mid=ema_mid_s):
                dec = {"regime": Regime.TREND, "side": Side.SELL, "sl_ticks": SL_TICKS_DEFAULT,
                       "comment": f"EarlyTrendDown(real): ADX={a:.1f}, slope20<0, slope50<0, -DI>{pdi:.1f}, c<EMA50"}
                return apply_trend_entry_guards(dec, df, md)

    # =================== EARLY-FLAT (ускоренный) ===================
    small_bodies_ok = False
    if len(df) >= EARLY_FLAT_MIN_BARS + 1:
        lastN = df.tail(EARLY_FLAT_MIN_BARS)
        bodies = (lastN["close"] - lastN["open"]).abs().astype(float) / tick
        small_bodies_ok = bool((bodies <= EARLY_FLAT_BODY_TICKS_MAX).all())

    early_flat = (
        a <= EARLY_FLAT_ADX_MAX and
        (atr(df, FLAT_ATR_N).iloc[-1] / ((df["high"].astype(float) - df["low"].astype(float)).rolling(FLAT_ATR_N).mean().iloc[-1] + 1e-12) <= EARLY_FLAT_ATR_RATIO_MAX) and
        abs(ef - es) / tick <= EARLY_FLAT_EMA_COMP_TICKS and
        abs(em - es) / tick <= EARLY_FLAT_EMA_COMP_TICKS and
        abs(c - vw) / tick <= EARLY_FLAT_VWAP_BAND_TICKS and
        small_bodies_ok
    )
    if early_flat:
        last = float(md.last_price)
        d_low_ticks  = ticks(last - edges["prev_low"],  f.tick_size)
        d_high_ticks = ticks(edges["prev_high"] - last, f.tick_size)
        if d_low_ticks <= d_high_ticks:
            side = Side.BUY
            note = (f"EARLY-FLAT: ADX={a:.1f}, EMAcomp={abs(ef-es)/tick:.0f}/{abs(em-es)/tick:.0f}t, "
                    f"VWAP±{abs(c-vw)/tick:.0f}t, smallBodies≤{EARLY_FLAT_BODY_TICKS_MAX}t | from prev_low {d_low_ticks:.0f}t")
        else:
            side = Side.SELL
            note = (f"EARLY-FLAT: ADX={a:.1f}, EMAcomp={abs(ef-es)/tick:.0f}/{abs(em-es)/tick:.0f}t, "
                    f"VWAP±{abs(c-vw)/tick:.0f}t, smallBodies≤{EARLY_FLAT_BODY_TICKS_MAX}t | from prev_high {d_high_ticks:.0f}t")
        return {"regime": Regime.FLAT, "side": side, "sl_ticks": SL_TICKS_DEFAULT, "comment": note}

    # =================== Классический FLAT ===================
    atr_s = atr(df, FLAT_ATR_N)
    atr_last = float(atr_s.iloc[-1]) if not np.isnan(atr_s.iloc[-1]) else 0.0
    tr_mean = float((df["high"].astype(float) - df["low"].astype(float)).rolling(FLAT_ATR_N).mean().iloc[-1])
    atr_ratio = atr_last / (tr_mean + 1e-12) if tr_mean > 0 else 0.0

    is_flat = (
        a <= FLAT_ADX_MAX and
        atr_ratio <= FLAT_ATR_RATIO_MAX and
        abs(ef - es) / tick <= FLAT_EMA_COMP_TICKS and
        abs(em - es) / tick <= FLAT_EMA_COMP_TICKS and
        abs(c - vw) / tick <= FLAT_VWAP_BAND_TICKS
    )
    if is_flat:
        last = float(md.last_price)
        d_low_ticks  = ticks(last - edges["prev_low"],  f.tick_size)
        d_high_ticks = ticks(edges["prev_high"] - last, f.tick_size)
        if d_low_ticks <= d_high_ticks:
            side = Side.BUY
            note = (f"FLAT: ADX={a:.1f}, ATRr={atr_ratio:.2f}, EMAcomp={abs(ef-es)/tick:.0f}/{abs(em-es)/tick:.0f}t, "
                    f"VWAP±{abs(c-vw)/tick:.0f}t | from prev_low {d_low_ticks:.0f}t")
        else:
            side = Side.SELL
            note = (f"FLAT: ADX={a:.1f}, ATRr={atr_ratio:.2f}, EMAcomp={abs(ef-es)/tick:.0f}/{abs(em-es)/tick:.0f}t, "
                    f"VWAP±{abs(c-vw)/tick:.0f}t | from prev_high {d_high_ticks:.0f}t")
        return {"regime": Regime.FLAT, "side": side, "sl_ticks": SL_TICKS_DEFAULT, "comment": note}

    # =================== HOLD ===================
    if 'atr_ratio' not in locals():
        atr_s = atr(df, FLAT_ATR_N)
        atr_last = float(atr_s.iloc[-1]) if not np.isnan(atr_s.iloc[-1]) else 0.0
        tr_mean = float((df["high"].astype(float) - df["low"].astype(float)).rolling(FLAT_ATR_N).mean().iloc[-1])
        atr_ratio = atr_last / (tr_mean + 1e-12) if tr_mean > 0 else 0.0

    return {"regime": Regime.HOLD, "side": Side.NONE, "sl_ticks": SL_TICKS_DEFAULT,
            "comment": f"Wait: ADX={a:.1f}, ATRr={atr_ratio:.2f}, EMAcomp={abs(ef-es)/tick:.0f}/{abs(em-es)/tick:.0f}t, VWAP±{abs(c-vw)/tick:.0f}t"}

# ============================ LOOP ============================
async def run_decider(
    *,
    symbol: str = SYMBOL,
    http: Optional[HTTP] = None,
    poll_tick_ms: int = POLL_TICK_MS,
    bootstrap_hours: int = BOOTSTRAP_HOURS,
    force_emit_every_tick: bool = False,
    on_decision: Optional[Callable[[Decision, MarketData], Awaitable[None]]] = None,
) -> None:
    """
    Бесконечная петля «решалки». Читает рынок → строит df → вызывает decide().
    Наружу отдаём только корректный JSON (как раньше).
    """
    cli = http or make_bybit()

    # bootstrap
    try:
        f = bybit_filters(cli, symbol)
        last = bybit_last_price(cli, symbol)
        kline = bybit_kline_1m(cli, symbol, bootstrap_hours * 60)
        pos = bybit_position(cli, symbol)
        bal = bybit_wallet_usdt(cli)
        md = MarketData(last, f, kline, pos, bal)
    except Exception as e:
        print(f"[DECIDER] bootstrap fail: {e}")
        f = bybit_filters(cli, symbol)
        last = bybit_last_price(cli, symbol)
        kline = bybit_kline_1m(cli, symbol, 60)
        pos = bybit_position(cli, symbol)
        bal = bybit_wallet_usdt(cli)
        md = MarketData(last, f, kline, pos, bal)

    df = kline_to_df(md.kline_1m)

    # первое решение
    dec = decide(symbol, df, md)
    last_regime = dec["regime"]
    last_side = dec["side"]
    if on_decision:
        await on_decision(dec, md)
    else:
        print(f"[DECIDER] {symbol} regime={dec['regime']} side={dec['side']} sl={dec['sl_ticks']} | {dec['comment']}")

    # основной цикл
    while True:
        try:
            md.last_price   = bybit_last_price(cli, symbol)
            md.kline_1m     = bybit_kline_1m(cli, symbol, 210)  # 210 чтобы EMA200 сошлась
            md.position     = bybit_position(cli, symbol)
            md.balance_usdt = bybit_wallet_usdt(cli)

            df = kline_to_df(md.kline_1m)
            dec_now = decide(symbol, df, md)

            emit = force_emit_every_tick or (dec_now["regime"] != last_regime or dec_now["side"] != last_side)
            if emit:
                if on_decision:
                    await on_decision(dec_now, md)
                else:
                    print(f"[DECIDER] {symbol} regime={dec_now['regime']} side={dec_now['side']} sl={dec_now['sl_ticks']} | {dec_now['comment']}")
                last_regime, last_side = dec_now["regime"], dec_now["side"]

            await asyncio.sleep(max(poll_tick_ms, 100) / 1000.0)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[DECIDER] loop error: {e}")
            await asyncio.sleep(1.0)

# ============================ DEMО ============================
if __name__ == "__main__":
    async def demo():
        async def on_dec(dec: Decision, md: MarketData):
            print(f"[ON_DECISION] {SYMBOL}: {dec['regime']}/{dec['side']} SL={dec['sl_ticks']} :: {dec['comment']} | last={md.last_price} | clamps: SLmin={MIN_SL_TICKS},TPmin={MIN_TP_TICKS}")

        await run_decider(
            symbol=SYMBOL,
            poll_tick_ms=POLL_TICK_MS,
            bootstrap_hours=BOOTSTRAP_HOURS,
            force_emit_every_tick=False,
            on_decision=on_dec,
        )

    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("Stopped.")
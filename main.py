
import os
import hmac
import time
import json
import math
import hashlib
import logging
import importlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timezone
#
# Runtime shared context holder so helpers (like close_position_market) can emit notifications
_RUNTIME_CTX: Dict = {}

import requests


# -----------------------
# Config from environment
# -----------------------
BASE_URL = os.getenv("BASE_URL", "https://api-testnet.bybit.com")
API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")

SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
CATEGORY = os.getenv("CATEGORY", "linear")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", os.getenv("TG_TOKEN", ""))
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

RISK_PCT = float(os.getenv("RISK_PCT", "0.01"))
MAX_RISK_USDT = float(os.getenv("MAX_RISK_USDT", "10"))

SECONDS = int(os.getenv("SECONDS", "5"))


SR_WINDOW = int(os.getenv("SR_WINDOW", "6"))           # pivots: neighbors on each side
SR_MIN_TOUCHES = int(os.getenv("SR_MIN_TOUCHES", "2")) # keep levels with >= touches
ATR_WINDOW = int(os.getenv("ATR_WINDOW", "14"))
EMA_FAST = int(os.getenv("EMA_FAST", "20"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "50"))

ENTRY_ATR_THRESH = float(os.getenv("ENTRY_ATR_THRESH", "0.25"))  # enter if distance to S/R <= X*ATR
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.0"))
TP_R_MULT = float(os.getenv("TP_R_MULT", "1.0"))

# --- AI/Strategy selection ---
AI_URL = os.getenv("AI_URL", "")
AI_AUTH = os.getenv("AI_AUTH", "")
AI_TIMEOUT = int(os.getenv("AI_TIMEOUT", "10"))
STRATEGY_DEFAULT = os.getenv("STRATEGY_DEFAULT", "breakout")  # knife|density|breakout|momentum

STRATEGY_PICK_INTERVAL_MIN = int(os.getenv("STRATEGY_PICK_INTERVAL_MIN", "30"))

# --- OpenAI selection (optional) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("OPENAI_KEY", ""))
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --- Debug/dumps toggles ---
DUMP_FEATURES = os.getenv("DUMP_FEATURES", "0") in ("1","true","True","yes","on")
DUMP_AI = os.getenv("DUMP_AI", "0") in ("1","true","True","yes","on")
DUMPS_DIR = os.getenv("DUMPS_DIR", "./ai_dumps")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# Insert VERBOSE_LOG handling
VERBOSE_LOG = os.getenv("VERBOSE_LOG", "1").strip()  # 1 enables full console log
if VERBOSE_LOG in ("1", "true", "True", "yes", "on"):
    LOG_LEVEL = "DEBUG"
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.DEBUG),
    format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
)
logger = logging.getLogger("scalper")


# -----------------------
# HTTP helpers
# -----------------------
def _ts_ms() -> int:
    return int(time.time() * 1000)


def _sign(params: Dict[str, str]) -> str:
    # Bybit v5 sign: concat sorted "key=value&" + api_secret
    sorted_items = sorted(params.items(), key=lambda x: x[0])
    qs = "&".join([f"{k}={v}" for k, v in sorted_items])
    return hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()


# -----------------------
# HTTP helpers
# -----------------------
def bybit_public_get(path: str, params: Dict[str, str]) -> Dict:
    logger.debug("[HTTP][GET] %s params=%s", path, params)
    url = f"{BASE_URL}{path}"
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    res = r.json()
    logger.debug("[HTTP][GET][RES] %s ret=%s keys=%s", path, res.get("retCode"), list(res.keys()))
    return res


def bybit_private_post(path: str, params: Dict[str, str]) -> Dict:
    # Hide sensitive keys from log
    log_params = {k: v for k, v in params.items() if k not in ("api_key", "sign")}
    logger.debug("[HTTP][POST] %s params=%s", path, log_params)
    url = f"{BASE_URL}{path}"
    ts = str(_ts_ms())
    params = {**params, "api_key": API_KEY, "timestamp": ts, "recv_window": "5000"}
    sign = _sign(params)
    params["sign"] = sign
    r = requests.post(url, data=params, timeout=15)
    r.raise_for_status()
    res = r.json()
    logger.debug("[HTTP][POST][RES] %s ret=%s keys=%s", path, res.get("retCode"), list(res.keys()))
    return res


# -----------------------
# Signed GET for Bybit v5 private endpoints
# -----------------------
def bybit_private_get(path: str, params: Dict[str, str]) -> Dict:
    """Signed GET for Bybit v5 private endpoints."""
    url = f"{BASE_URL}{path}"
    ts = str(_ts_ms())
    signed = {**params, "api_key": API_KEY, "timestamp": ts, "recv_window": "5000"}
    sign = _sign(signed)
    signed["sign"] = sign
    logger.debug("[HTTP][GET*] %s params=%s", path, {k: v for k, v in signed.items() if k != "sign"})
    r = requests.get(url, params=signed, timeout=15)
    r.raise_for_status()
    res = r.json()
    logger.debug("[HTTP][GET*][RES] %s ret=%s keys=%s", path, res.get("retCode"), list(res.keys()))
    return res


# Extra market-data helpers (orderbook, trades, tickers)

def get_orderbook_l2(category: str, symbol: str, limit: int = 50) -> Dict:
    data = bybit_public_get(
        "/v5/market/orderbook",
        {"category": category, "symbol": symbol, "limit": str(limit)},
    )
    if data.get("retCode") != 0:
        raise RuntimeError(f"orderbook error: {data}")
    logger.debug("[MD][OB] bids=%d asks=%d", len(data.get("b", [])), len(data.get("a", [])))
    return data["result"]


def get_recent_trades(category: str, symbol: str, limit: int = 200) -> List[Dict]:
    data = bybit_public_get(
        "/v5/market/recent-trade",
        {"category": category, "symbol": symbol, "limit": str(limit)},
    )
    if data.get("retCode") != 0:
        raise RuntimeError(f"recent-trade error: {data}")
    # result.list -> [{"T":ms,"s":symbol,"S":"Buy|Sell","v":"qty","p":"price"}, ...]
    items = data.get("result", {}).get("list", [])
    items.sort(key=lambda x: int(x.get("T", 0)))
    logger.debug("[MD][TRADES] n=%d t0=%s t1=%s", len(items), items[0].get("T") if items else None, items[-1].get("T") if items else None)
    return items


def get_ticker(category: str, symbol: str) -> Dict:
    data = bybit_public_get(
        "/v5/market/tickers",
        {"category": category, "symbol": symbol},
    )
    if data.get("retCode") != 0:
        raise RuntimeError(f"tickers error: {data}")
    lst = data.get("result", {}).get("list", [])
    logger.debug("[MD][TICKER] keys=%s", list((lst[0] if lst else {}).keys()))
    return lst[0] if lst else {}


# -----------------------
# Market data
# -----------------------
def get_kline(category: str, symbol: str, interval: str, limit: int) -> List[Dict]:
    """
    Returns list of bars (oldest..newest) with keys: open, high, low, close, volume, start
    """
    data = bybit_public_get(
        "/v5/market/kline",
        {"category": category, "symbol": symbol, "interval": interval, "limit": str(limit)},
    )
    if data.get("retCode") != 0:
        raise RuntimeError(f"kline error: {data}")
    items = data["result"]["list"]
    items.sort(key=lambda x: int(x[0]))  # 0=start(ms)
    bars = []
    for it in items:
        # [ start, open, high, low, close, volume, turnover ]
        bars.append(
            {
                "start": int(it[0]),
                "open": float(it[1]),
                "high": float(it[2]),
                "low": float(it[3]),
                "close": float(it[4]),
                "volume": float(it[5]),
            }
        )
    logger.debug("[MD][KLINE] %s %s x%d (%.2f..%.2f)", symbol, interval, len(bars), bars[0]["close"] if bars else 0, bars[-1]["close"] if bars else 0)
    return bars


def get_instrument_info(category: str, symbol: str) -> Dict:
    data = bybit_public_get(
        "/v5/market/instruments-info",
        {"category": category, "symbol": symbol},
    )
    if data.get("retCode") != 0:
        raise RuntimeError(f"instruments-info error: {data}")
    return data["result"]["list"][0]


def get_wallet_balance() -> float:
    data = bybit_private_get("/v5/account/wallet-balance", {"accountType": "UNIFIED"})
    if data.get("retCode") != 0:
        raise RuntimeError(f"balance error: {data}")
    # try USDT
    coins = data["result"]["list"][0]["coin"]
    balance = None
    for c in coins:
        if c["coin"] == "USDT":
            balance = float(c["equity"])
            break
    if balance is None:
        # fallback total
        balance = float(data["result"]["list"][0]["totalEquity"])
    logger.debug("[ACC][BAL] equity=%.4f USDT", balance)
    return balance


# -----------------------
# Math utils (EMA, ATR)
# -----------------------
# -----------------------
# Math utils (EMA, ATR)
# -----------------------
def ema(series: List[float], period: int) -> List[float]:
    if period <= 1 or len(series) == 0:
        return series[:]
    k = 2 / (period + 1)
    out = []
    ema_val = series[0]
    for v in series:
        ema_val = v * k + ema_val * (1 - k)
        out.append(ema_val)
    return out


def atr(bars: List[Dict], period: int) -> List[float]:
    out = []
    prev_close = bars[0]["close"]
    trs = []
    for i, b in enumerate(bars):
        tr = max(b["high"] - b["low"], abs(b["high"] - prev_close), abs(b["low"] - prev_close))
        prev_close = b["close"]
        trs.append(tr)
        if len(trs) < period:
            out.append(sum(trs) / len(trs))
        else:
            out.append(sum(trs[-period:]) / period)
    return out


# -----------------------
# Micro-trend & features for AI (M1/M5 + order flow)
# -----------------------

def detect_micro_trend(closes: List[float], fast: int = 9, slow: int = 21) -> str:
    if not closes:
        return "flat"
    efast = ema(closes, min(fast, max(2, len(closes)//4)))
    eslow = ema(closes, min(slow, max(3, len(closes)//2)))
    if efast[-1] > eslow[-1] and (len(efast) < 3 or efast[-1] >= efast[-3]):
        return "up"
    if efast[-1] < eslow[-1] and (len(efast) < 3 or efast[-1] <= efast[-3]):
        return "down"
    return "flat"


def _orderbook_features(ob: Dict) -> Dict:
    bids = ob.get("b", [])  # [[price, size], ...]
    asks = ob.get("a", [])
    best_bid = float(bids[0][0]) if bids else 0.0
    best_ask = float(asks[0][0]) if asks else 0.0
    spread = max(0.0, best_ask - best_bid)
    # Top densities: pick top 3 by size on each side
    top_bids = sorted(([{"price": float(p), "sz": float(s)} for p, s in bids[:50]]), key=lambda x: x["sz"], reverse=True)[:3]
    top_asks = sorted(([{"price": float(p), "sz": float(s)} for p, s in asks[:50]]), key=lambda x: x["sz"], reverse=True)[:3]
    sum_bid = sum(float(s) for _, s in bids[:20]) if bids else 0.0
    sum_ask = sum(float(s) for _, s in asks[:20]) if asks else 0.0
    imb = (sum_bid - sum_ask) / (sum_bid + sum_ask) if (sum_bid + sum_ask) > 0 else 0.0
    return {
        "spread": spread,
        "best_bid_sz": float(bids[0][1]) if bids else 0.0,
        "best_ask_sz": float(asks[0][1]) if asks else 0.0,
        "top_bids": top_bids,
        "top_asks": top_asks,
        "imbalance_bp": imb,
    }


def _tape_features(trades: List[Dict], last_n: int = 200) -> Dict:
    if not trades:
        return {"last_n": 0, "buy_ratio": 0.0, "tick_rate_per_min": 0.0}
    tail = trades[-last_n:]
    buys = sum(1 for t in tail if str(t.get("S", "")).lower().startswith("buy"))
    sells = len(tail) - buys
    # tick rate: trades per minute over the span of tail
    t0 = int(tail[0].get("T", 0))
    t1 = int(tail[-1].get("T", 0))
    minutes = max(1.0, (t1 - t0) / 60000.0)
    rate = len(tail) / minutes
    return {"last_n": len(tail), "buy_ratio": (buys / len(tail)) if tail else 0.0, "tick_rate_per_min": rate}


def collect_features() -> Dict:
    t0 = time.time()
    # M1 and M5 klines
    m1 = get_kline(CATEGORY, SYMBOL, interval="1", limit=180)   # last 3 hours of 1m
    m5 = get_kline(CATEGORY, SYMBOL, interval="5", limit=72)    # last 6 hours of 5m
    closes_m1 = [b["close"] for b in m1]
    closes_m5 = [b["close"] for b in m5]

    # ATR on M1/M5
    atr_m1 = atr(m1, min(ATR_WINDOW, max(2, len(m1)//6)))[-1] if m1 else 0.0
    atr_m5 = atr(m5, min(ATR_WINDOW, max(2, len(m5)//6)))[-1] if m5 else 0.0

    trend_m1 = detect_micro_trend(closes_m1, 9, 21)
    trend_m5 = detect_micro_trend(closes_m5, 20, 50)

    # S/R from M5
    sup_levels, res_levels = find_levels(m5, SR_WINDOW, SR_MIN_TOUCHES)
    last_price = closes_m1[-1] if closes_m1 else 0.0
    near_sup = min((lvl for lvl in sup_levels if lvl <= last_price), key=lambda x: abs(x - last_price), default=None)
    near_res = min((lvl for lvl in res_levels if lvl >= last_price), key=lambda x: abs(x - last_price), default=None)

    # Order book & trades
    ob = get_orderbook_l2(CATEGORY, SYMBOL, 50)
    obf = _orderbook_features(ob)
    trades = get_recent_trades(CATEGORY, SYMBOL, 200)
    tf = _tape_features(trades, 200)

    # Ticker deltas (30m/60m)
    # compute simple returns from M1 closes
    def _ret_from_minutes(n: int) -> float:
        if len(closes_m1) <= n:
            return 0.0
        prev = closes_m1[-n-1]
        curr = closes_m1[-1]
        return (curr - prev) / prev if prev else 0.0

    features = {
        "symbol": SYMBOL,
        "time_utc": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        "trend_m1": trend_m1,
        "trend_m5": trend_m5,
        "atr_m1": atr_m1,
        "atr_m5": atr_m5,
        "price": last_price,
        "sr": {
            "nearest_support": near_sup,
            "nearest_resistance": near_res,
            "dist_to_support": abs(last_price - near_sup) if near_sup is not None else None,
            "dist_to_resistance": abs(last_price - near_res) if near_res is not None else None,
        },
        "orderbook": obf,
        "tape": tf,
        "volatility": {"ret_30m": _ret_from_minutes(30), "ret_60m": _ret_from_minutes(60)},
    }
    logger.debug("[FEATS] built in %.3fs | price=%.2f t1=%s t5=%s atr1=%.2f atr5=%.2f ob.spread=%.2f buy_ratio=%.2f tick=%.0f/m",
                 time.time()-t0, last_price, trend_m1, trend_m5, atr_m1, atr_m5, obf.get("spread",0.0), tf.get("buy_ratio",0.0), tf.get("tick_rate_per_min",0.0))
    if DUMP_FEATURES:
        try:
            os.makedirs(DUMPS_DIR, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(DUMPS_DIR, f"features_{SYMBOL}_{ts}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(features, f, ensure_ascii=False, indent=2)
            print(f"[DUMP][FEATURES] -> {path}")
        except Exception as e:
            logger.warning("[DUMP][FEATURES] error: %s", e)
    return features


# -----------------------
# S/R detection (pivots)
# -----------------------
def find_levels(bars: List[Dict], window: int, min_touches: int) -> Tuple[List[float], List[float]]:
    highs = [b["high"] for b in bars]
    lows = [b["low"] for b in bars]

    res: Dict[float, int] = {}
    sup: Dict[float, int] = {}

    def is_pivot(arr: List[float], idx: int, is_high: bool) -> bool:
        left = arr[max(0, idx - window):idx]
        right = arr[idx + 1: idx + 1 + window]
        if not left or not right:
            return False
        if is_high:
            return arr[idx] == max(left + [arr[idx]] + right)
        else:
            return arr[idx] == min(left + [arr[idx]] + right)

    for i in range(len(bars)):
        if is_pivot(highs, i, True):
            level = highs[i]
            level = round(level, 2)
            res[level] = res.get(level, 0) + 1
        if is_pivot(lows, i, False):
            level = lows[i]
            level = round(level, 2)
            sup[level] = sup.get(level, 0) + 1

    # keep levels with touches
    res_levels = sorted([lvl for lvl, t in res.items() if t >= min_touches])
    sup_levels = sorted([lvl for lvl, t in sup.items() if t >= min_touches])
    return sup_levels, res_levels


# -----------------------
# Trend detection
# -----------------------
def detect_trend_from_daily(daily_bars: List[Dict]) -> str:
    closes = [b["close"] for b in daily_bars]
    efast = ema(closes, EMA_FAST if EMA_FAST <= len(closes) else max(2, len(closes)//2))
    eslow = ema(closes, EMA_SLOW if EMA_SLOW <= len(closes) else max(3, len(closes)//2 + 1))
    if efast[-1] > eslow[-1]:
        return "up"
    elif efast[-1] < eslow[-1]:
        return "down"
    else:
        return "flat"


def confirm_trend_hourly(hourly_bars: List[Dict]) -> str:
    closes = [b["close"] for b in hourly_bars]
    efast = ema(closes, min(EMA_FAST, max(2, len(closes)//5)))
    eslow = ema(closes, min(EMA_SLOW, max(3, len(closes)//3)))
    # slope via last two values
    slope_fast = efast[-1] - efast[-3] if len(efast) >= 3 else efast[-1] - efast[0]
    if efast[-1] > eslow[-1] and slope_fast > 0:
        return "up"
    if efast[-1] < eslow[-1] and slope_fast < 0:
        return "down"
    return "flat"


# -----------------------
# Telegram
# -----------------------
def tg_send(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("[TG] skip: no token or chat id")
        return
    logger.debug("[TG] send len=%d preview=%.60s", len(text or ""), (text or "").replace("\n"," | ")[:60])
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        logger.warning("[TG] send error: %s", e)

# -----------------------
# Trade lifecycle helpers (for strategies)
# -----------------------
def _fmt_ts(ts: Optional[float] = None) -> str:
    if ts is None:
        ts = time.time()
    return datetime.utcfromtimestamp(ts).replace(tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def _last_price() -> float:
    try:
        tk = get_ticker(CATEGORY, SYMBOL)
        # Bybit v5 ticker uses 'lastPrice' for linear
        p = tk.get("lastPrice") or tk.get("last_price") or tk.get("markPrice") or tk.get("indexPrice")
        return float(p) if p is not None else 0.0
    except Exception:
        return 0.0

def on_strategy_entry(ctx: Dict, strategy: str, side: str, indicator: str, qty: float, price: Optional[float] = None) -> None:
    """
    Strategies should call this when they OPEN a position.
    Records the trade in ctx["current_trade"] and sends a Telegram message.
    """
    if price is None or price <= 0:
        price = _last_price()
    now_ts = time.time()
    ctx["current_trade"] = {
        "strategy": strategy,
        "side": side,            # 'long' or 'short'
        "indicator": indicator,  # e.g. 'SR-breakout', 'EMA-pullback'
        "qty": float(qty),
        "entry_price": float(price),
        "entry_ts": now_ts,
    }
    msg = (
        f"🚀 <b>ENTRY</b> {SYMBOL}\n"
        f"strat: <b>{strategy}</b>\n"
        f"side: <b>{side.upper()}</b> qty: <code>{qty:.6f}</code>\n"
        f"price: <code>{price:.2f}</code>\n"
        f"by: <i>{indicator}</i>\n"
        f"time: <code>{_fmt_ts(now_ts)}</code>"
    )
    tg_send(msg)
    logger.info("[TRADE][ENTRY] strat=%s side=%s qty=%.6f price=%.2f by=%s", strategy, side, qty, price, indicator)

def on_strategy_exit(ctx: Dict, price: Optional[float] = None, reason: str = "exit") -> None:
    """
    Strategies should call this when they CLOSE a position (TP/SL/manual).
    Computes naive PnL in USDT for USDT-margined linear contracts.
    """
    trade = ctx.get("current_trade")
    if not trade:
        logger.debug("[TRADE][EXIT] skip: no current_trade in ctx")
        return
    if price is None or price <= 0:
        price = _last_price()
    now_ts = time.time()

    entry_price = float(trade.get("entry_price", 0.0) or 0.0)
    qty = float(trade.get("qty", 0.0) or 0.0)
    side = trade.get("side", "long")
    strategy = trade.get("strategy", "?")
    indicator = trade.get("indicator", "?")

    # PnL ≈ (exit - entry) * qty for long; inverse for short (USDT linear)
    direction = 1.0 if side == "long" else -1.0
    pnl_usdt = (float(price) - entry_price) * qty * direction
    pnl_pct = ((float(price) / entry_price - 1.0) * (1 if side == "long" else -1)) * 100.0 if entry_price > 0 else 0.0

    msg = (
        f"✅ <b>EXIT</b> {SYMBOL} ({reason})\n"
        f"strat: <b>{strategy}</b> · by: <i>{indicator}</i>\n"
        f"side: <b>{side.upper()}</b> qty: <code>{qty:.6f}</code>\n"
        f"entry: <code>{entry_price:.2f}</code> @ {_fmt_ts(trade.get('entry_ts'))}\n"
        f"exit:  <code>{float(price):.2f}</code> @ {_fmt_ts(now_ts)}\n"
        f"PnL: <b>{pnl_usdt:+.2f} USDT</b> (<code>{pnl_pct:+.2f}%</code>)"
    )
    tg_send(msg)
    logger.info("[TRADE][EXIT] strat=%s side=%s qty=%.6f entry=%.2f exit=%.2f pnl=%.4f USDT (%+.2f%%) reason=%s",
                strategy, side, qty, entry_price, float(price), pnl_usdt, pnl_pct, reason)
    # clear current trade
    ctx.pop("current_trade", None)


# -----------------------
# Trading helpers
# -----------------------
@dataclass
class MarketFilters:
    qty_step: float
    min_qty: float


def load_filters() -> MarketFilters:
    info = get_instrument_info(CATEGORY, SYMBOL)
    lot_sz = float(info["lotSizeFilter"]["qtyStep"])
    min_trd = float(info["lotSizeFilter"]["minOrderQty"])
    return MarketFilters(qty_step=lot_sz, min_qty=min_trd)


def place_market_order(side: str, qty: float, stop_loss: Optional[float], take_profit: Optional[float], reduce_only: bool = False) -> Dict:
    """
    Simple market order with SL/TP (server-side).
    """
    params = {
        "category": CATEGORY,
        "symbol":   SYMBOL,
        "side":     "Buy" if side == "long" else "Sell",
        "orderType":"Market",
        "qty":      f"{qty}",
        "timeInForce":"IOC",
        "reduceOnly": "true" if reduce_only else "false",
        "positionIdx":"0",  # one-way
    }
    if stop_loss:
        params["stopLoss"] = f"{stop_loss}"
        params["slTriggerBy"] = "LastPrice"
    if take_profit:
        params["takeProfit"] = f"{take_profit}"
        params["tpTriggerBy"] = "LastPrice"

    logger.debug("[ORD][NEW] side=%s qty=%s SL=%s TP=%s", side, params.get("qty"), params.get("stopLoss"), params.get("takeProfit"))
    res = bybit_private_post("/v5/order/create", params)
    if res.get("retCode") != 0:
        raise RuntimeError(f"order error: {res}")
    logger.debug("[ORD][OK] id=%s ret=%s", res.get("result", {}).get("orderId"), res.get("retCode"))
    try:
        ct = _RUNTIME_CTX.get("current_trade")
        if ct and not ct.get("entry_price"):
            ct["entry_price"] = _last_price()
    except Exception:
        pass
    return res


def get_open_position() -> Optional[Tuple[str, float]]:
    """
    Returns (side, qty) for current open position in one-way mode, or None if no position.
    side: "long" or "short"
    qty: float size in contract units
    """
    try:
        res = bybit_private_get("/v5/position/list", {"category": CATEGORY, "symbol": SYMBOL})
        if res.get("retCode") != 0:
            logger.warning("[POS] list error: %s", res)
            return None
        lst = res.get("result", {}).get("list", [])
        if not lst:
            return None
        # Bybit returns separate entries per side in some modes; find the one with size>0
        for it in lst:
            try:
                sz = float(it.get("size", 0) or 0)
            except Exception:
                sz = float(it.get("positionValue", 0) or 0)
            if sz and sz > 0:
                sd = it.get("side", "").lower()  # "Buy"/"Sell"
                side = "long" if sd.startswith("buy") else "short"
                # Prefer exact size field if present
                qty = float(it.get("size", sz))
                logger.debug("[POS] side=%s qty=%.6f", side, qty)
                return side, qty
        return None
    except Exception as e:
        logger.warning("[POS] list exception: %s", e)
        return None


def close_position_market(side: str, qty: float) -> Optional[str]:
    """
    Close existing position by placing a reduce-only market order in the opposite direction.
    side: current position side ("long"/"short").
    qty: current absolute size.
    Returns orderId or None on failure.
    """
    try:
        logger.debug("[CLOSE] side=%s qty=%.6f", side, qty)
        opp = "short" if side == "long" else "long"
        res = place_market_order(opp, qty, stop_loss=None, take_profit=None, reduce_only=True)
        if res.get("retCode") != 0:
            logger.error("[CLOSE] error: %s", res)
            return None
        logger.debug("[CLOSE][OK] id=%s", res.get("result", {}).get("orderId"))
        try:
            if _RUNTIME_CTX.get("current_trade"):
                on_strategy_exit(_RUNTIME_CTX, price=None, reason="manual/close")
        except Exception as _e:
            logger.debug("[CLOSE][NOTICE] on_strategy_exit failed: %s", _e)
        return res.get("result", {}).get("orderId")
    except Exception as e:
        logger.error("[CLOSE] exception: %s", e)
        return None


# -----------------------
# AI strategy picker & registry
# -----------------------

STRATEGY_MODULES = {
    "knife": "strategies.knife",
    "density": "strategies.density",
    "breakout": "strategies.breakout",
    "momentum": "strategies.momentum",
}

_strategy_cache: Dict[str, callable] = {}


def _load_strategy_runner(name: str):
    if name in _strategy_cache:
        return _strategy_cache[name]
    mod_name = STRATEGY_MODULES.get(name)
    if not mod_name:
        raise ValueError(f"unknown strategy: {name}")
    mod = importlib.import_module(mod_name)
    runner = getattr(mod, "run_once")
    _strategy_cache[name] = runner
    return runner


# --- OpenAI strategy picker ---
def choose_strategy_via_openai(features: Dict) -> Dict:
    """Use OpenAI Chat Completions to pick strategy.
    Requires env OPENAI_API_KEY. Returns same dict as choose_strategy_via_ai.
    """
    if not OPENAI_API_KEY:
        logger.debug("[AI:OpenAI] skip: no OPENAI_API_KEY")
        return {}
    try:
        logger.debug("[AI:OpenAI] request model=%s", OPENAI_MODEL)
        url = f"{OPENAI_API_BASE}/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        system_prompt = (
            "You are an expert high-frequency scalping strategist. "
            "Given market microstructure features (M1/M5 candles, order book, tape), "
            "choose ONE strategy from: knife, density, breakout, momentum. "
            "Respond ONLY with a compact JSON object with fields: strategy, reason, confidence. "
            "strategy must be one of [knife,density,breakout,momentum]; confidence in [0,1]."
        )
        user_prompt = json.dumps(features, ensure_ascii=False)
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 200,
        }
        # Verbose console logs for full transparency
        headers_log = dict(headers)
        if "Authorization" in headers_log:
            headers_log["Authorization"] = "***redacted***"
        logger.debug("[AI:OpenAI][HEADERS] %s", headers_log)
        logger.debug("[AI:OpenAI][FEATURES] %s", user_prompt)
        logger.debug("[AI:OpenAI][REQ] url=%s model=%s payload=%s", url, OPENAI_MODEL, json.dumps(payload, ensure_ascii=False))
        if DUMP_AI:
            try:
                os.makedirs(DUMPS_DIR, exist_ok=True)
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                req_path = os.path.join(DUMPS_DIR, f"openai_req_{SYMBOL}_{ts}.json")
                with open(req_path, "w", encoding="utf-8") as f:
                    to_dump = {"url": url, "model": OPENAI_MODEL, "headers": {k:("***redacted***" if k.lower()=="authorization" else v) for k,v in headers.items()}, "payload": payload}
                    json.dump(to_dump, f, ensure_ascii=False, indent=2)
                print(f"[DUMP][OPENAI][REQ] -> {req_path}")
            except Exception as e:
                logger.warning("[DUMP][OPENAI][REQ] error: %s", e)
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=AI_TIMEOUT)
        resp.raise_for_status()
        logger.debug("[AI:OpenAI][HTTP] status=%s", resp.status_code)
        logger.debug("[AI:OpenAI][RESP] %s", resp.text)
        if DUMP_AI:
            try:
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                resp_path = os.path.join(DUMPS_DIR, f"openai_resp_{SYMBOL}_{ts}.json")
                with open(resp_path, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                print(f"[DUMP][OPENAI][RESP] -> {resp_path}")
            except Exception as e:
                logger.warning("[DUMP][OPENAI][RESP] error: %s", e)
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        logger.debug("[AI:OpenAI][CHOICE_RAW] %s", content)
        # Try to extract JSON (strip code fences)
        txt = content.strip()
        if txt.startswith("```"):
            # remove fence lines
            lines = [ln for ln in txt.splitlines() if not ln.strip().startswith("```")]
            txt = "\n".join(lines).strip()
        try:
            parsed = json.loads(txt)
        except Exception:
            # crude fallback: find first { ... }
            start = txt.find("{")
            end = txt.rfind("}")
            parsed = json.loads(txt[start:end+1]) if start != -1 and end != -1 else {}
        st = (parsed.get("strategy") or "").strip().lower()
        if st not in STRATEGY_MODULES:
            raise ValueError(f"OpenAI returned unsupported strategy: {st}")
        logger.debug("[AI:OpenAI][RES] %s conf=%.2f", st, float(parsed.get("confidence", 0.0)))
        return {
            "strategy": st,
            "reason": parsed.get("reason", ""),
            "confidence": float(parsed.get("confidence", 0.0)),
        }
    except Exception as e:
        logger.exception("[AI:OpenAI] choose error")
        return {"strategy": STRATEGY_DEFAULT, "reason": f"OpenAI error: {e}", "confidence": 0.0}


def choose_strategy_via_ai(features: Dict) -> Dict:
    """Send features to external AI to get strategy choice.
    Expected response: {"strategy": "knife|density|breakout|momentum", "reason": str, "confidence": 0..1}
    """
    if not AI_URL:
        if OPENAI_API_KEY:
            pick = choose_strategy_via_openai(features)
            if pick:
                return pick
        logger.debug("[AI:PICK] fallback heuristic: no AI_URL and OpenAI unavailable or failed")
        # Fallback simple heuristic when neither AI_URL nor OpenAI is configured
        choice = "breakout"
        if features.get("orderbook", {}).get("imbalance_bp", 0) > 0.4:
            choice = "momentum"
        return {"strategy": choice, "reason": "fallback heuristic (no AI_URL)", "confidence": 0.5}
    try:
        logger.debug("[AI:HTTP] POST %s", AI_URL)
        headers = {"Content-Type": "application/json"}
        if AI_AUTH:
            headers["Authorization"] = AI_AUTH
        headers_log = dict(headers)
        if "Authorization" in headers_log:
            headers_log["Authorization"] = "***redacted***"
        logger.debug("[AI:HTTP][HEADERS] %s", headers_log)
        if DUMP_AI:
            try:
                os.makedirs(DUMPS_DIR, exist_ok=True)
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                req_path = os.path.join(DUMPS_DIR, f"httpai_req_{SYMBOL}_{ts}.json")
                with open(req_path, "w", encoding="utf-8") as f:
                    to_dump = {"url": AI_URL, "headers": {k:("***redacted***" if k.lower()=="authorization" else v) for k,v in headers.items()}, "payload": features}
                    json.dump(to_dump, f, ensure_ascii=False, indent=2)
                print(f"[DUMP][HTTPAI][REQ] -> {req_path}")
            except Exception as e:
                logger.warning("[DUMP][HTTPAI][REQ] error: %s", e)
        resp = requests.post(AI_URL, headers=headers, data=json.dumps(features), timeout=AI_TIMEOUT)
        resp.raise_for_status()
        logger.debug("[AI:HTTP][STATUS] %s", resp.status_code)
        logger.debug("[AI:HTTP][RESP] %s", resp.text)
        if DUMP_AI:
            try:
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                resp_path = os.path.join(DUMPS_DIR, f"httpai_resp_{SYMBOL}_{ts}.json")
                with open(resp_path, "w", encoding="utf-8") as f:
                    f.write(resp.text)
                print(f"[DUMP][HTTPAI][RESP] -> {resp_path}")
            except Exception as e:
                logger.warning("[DUMP][HTTPAI][RESP] error: %s", e)
        data = resp.json()
        st = data.get("strategy")
        if st not in STRATEGY_MODULES:
            raise ValueError(f"AI returned unsupported strategy: {st}")
        logger.debug("[AI:HTTP][RES] %s conf=%s", st, data.get("confidence"))
        return {"strategy": st, "reason": data.get("reason", ""), "confidence": float(data.get("confidence", 0.0))}
    except Exception as e:
        logger.warning("[AI] choose error: %s", e)
        return {"strategy": STRATEGY_DEFAULT, "reason": f"AI error: {e}", "confidence": 0.0}


# -----------------------
# Core strategy loop (delegates to selected strategy)
# -----------------------

def run_once_active_strategy(filters: MarketFilters, active_strategy: str, shared_context: Dict) -> None:
    # NOTE for strategy authors:
    # Call shared_context["on_entry"](strategy="name", side="long|short", indicator="...", qty=..., price=optional_fill_price)
    # and shared_context["on_exit"](price=optional_exit_price, reason="tp|sl|exit") to emit Telegram trade reports.
    logger.debug("[STRAT] tick using %s", active_strategy)
    try:
        runner = _load_strategy_runner(active_strategy)
    except Exception as e:
        logger.error("[STRAT] load failed for %s: %s", active_strategy, e)
        tg_send(f"❌ <b>Strategy load failed</b> <code>{active_strategy}</code>\n<code>{e}</code>")
        return
    try:
        runner(filters, shared_context)
    except Exception as e:
        logger.exception("[STRAT] run_once exception: %s", e)
        tg_send(f"⚠️ <b>Strategy error</b> <code>{active_strategy}</code>\n<code>{e}</code>")


def build_shared_context() -> Dict:
    feats = collect_features()
    logger.debug("[CTX] features ready")
    # Keep context minimal but useful to strategies
    ctx = {
        "features": feats,
        "symbol": SYMBOL,
        "category": CATEGORY,
        # will be filled below
    }
    # bind entry/exit callbacks for strategies
    def _on_entry(*, strategy: str, side: str, indicator: str, qty: float, price: Optional[float] = None):
        return on_strategy_entry(ctx, strategy=strategy, side=side, indicator=indicator, qty=qty, price=price)

    def _on_exit(*, price: Optional[float] = None, reason: str = "exit"):
        return on_strategy_exit(ctx, price=price, reason=reason)

    ctx["on_entry"] = _on_entry
    ctx["on_exit"] = _on_exit

    # expose filters & math shortcuts if strategies want them
    ctx["ema"] = ema
    ctx["atr"] = atr

    # publish to runtime holder so other helpers (like close_position_market) can notify
    _RUNTIME_CTX.clear()
    _RUNTIME_CTX.update(ctx)
    return ctx


def main():
    logger.info("[BOOT] Scalper starting | symbol=%s category=%s", SYMBOL, CATEGORY)
    if not API_KEY or not API_SECRET:
        logger.warning("[CFG] API keys are empty — trading will fail.")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("[CFG] Telegram not configured; notifications disabled.")

    filters = load_filters()
    logger.info("[INFO] qty_step=%.6f min_qty=%.6f", filters.qty_step, filters.min_qty)

    active_strategy = STRATEGY_DEFAULT
    shared_context = build_shared_context()

    # Initial AI pick at boot
    pick = choose_strategy_via_ai(shared_context["features"])
    if pick:
        active_strategy = pick.get("strategy", STRATEGY_DEFAULT)
        tg_send(
            f"🤖 <b>Strategy pick</b> {SYMBOL}\n"
            f"choice: <b>{active_strategy.upper()}</b> (conf {pick.get('confidence', 0.0):.2f})\n"
            f"reason: {pick.get('reason', '')}"
        )
        logger.info("[PICK] %s | conf=%.2f | %s", active_strategy, pick.get("confidence", 0.0), pick.get("reason", ""))

    last_min = -1
    last_pick_key = None  # (day, hour, halfhour)

    while True:
        try:
            now = time.time()
            now_min = int(now // 60)

            # Re-pick every STRATEGY_PICK_INTERVAL_MIN on aligned boundary
            t = time.gmtime(now)
            half_block = (t.tm_min // STRATEGY_PICK_INTERVAL_MIN)
            pick_key = (t.tm_mday, t.tm_hour, half_block)
            logger.debug("[LOOP] minute=%d key=%s", t.tm_min, pick_key)
            if pick_key != last_pick_key:
                shared_context = build_shared_context()
                pick = choose_strategy_via_ai(shared_context["features"])
                new_strategy = pick.get("strategy", active_strategy)
                if new_strategy != active_strategy:
                    active_strategy = new_strategy
                tg_send(
                    f"🤖 <b>Strategy pick</b> {SYMBOL}\n"
                    f"choice: <b>{active_strategy.upper()}</b> (conf {pick.get('confidence', 0.0):.2f})\n"
                    f"reason: {pick.get('reason', '')}"
                )
                logger.info("[PICK] %s | conf=%.2f | %s", active_strategy, pick.get("confidence", 0.0), pick.get("reason", ""))
                last_pick_key = pick_key

            # Call active strategy once per minute boundary
            if now_min != last_min:
                run_once_active_strategy(filters, active_strategy, shared_context)
                logger.debug("[LOOP] run_once done")
                last_min = now_min

        except Exception as e:
            logger.exception("[LOOP] exception: %s", e)
            tg_send(f"⚠️ <b>Loop error</b>\n<code>{e}</code>")
        time.sleep(SECONDS)


if __name__ == "__main__":
    main()

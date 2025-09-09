# main.py — логика рынка, поток стратегий, ИИ-переключатель
import os
import time
import hmac
import hashlib
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional, List

import requests
import asyncio
from dotenv import load_dotenv
load_dotenv()

from bot import get_bot            # весь ТГ-бот — внутри bot.py
from ai import pick_strategy       # ИИ-пикер

import momentum, breakout, density, knife  # стратегии

# ===== Instrument cache (to avoid spamming instruments-info each tick) =====
_INST_INFO = {"ts": 0.0, "data": None}

def get_instrument_filters_cached(ttl: int = 60) -> tuple[float, float, float]:
    """Return (qty_step, min_qty, price_step) with simple TTL caching to cut requests.

    Bybit source:
      - lotSizeFilter.qtyStep  -> step for quantity
      - lotSizeFilter.minOrderQty -> minimum order qty
      - priceFilter.tickSize  -> price tick size (round prices to this)
    """
    global _INST_INFO
    now = time.time()
    need_refresh = (_INST_INFO["data"] is None) or (now - _INST_INFO["ts"] > ttl)
    if need_refresh:
        try:
            info = get_instrument_info()
            insts = (info.get("result", {}) or {}).get("list", [])
            _INST_INFO = {"ts": now, "data": (insts[0] if insts else {})}
        except Exception:
            # keep previous if any
            if _INST_INFO["data"] is None:
                _INST_INFO = {"ts": now, "data": {}}
    inst = _INST_INFO["data"] or {}
    lot = float((inst.get("lotSizeFilter", {}) or {}).get("qtyStep", 0.001) or 0.001)
    min_qty = float((inst.get("lotSizeFilter", {}) or {}).get("minOrderQty", 0.001) or 0.001)
    price_step = float((inst.get("priceFilter", {}) or {}).get("tickSize", 0.1) or 0.1)
    return lot, min_qty, price_step

# one more guard to rate-limit noisy warnings
_last_no_data_warn_ts = 0.0

# ===== Logging =====
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"
)
log = logging.getLogger("main")

# ===== ENV =====
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
CATEGORY = os.getenv("BYBIT_CATEGORY", "linear")
BASE_URL = os.getenv("BYBIT_BASE_URL", "https://api-testnet.bybit.com")
API_KEY = os.getenv("BYBIT_API_KEY", "")
API_SECRET = os.getenv("BYBIT_API_SECRET", "")

RISK_PCT = float(os.getenv("RISK_PCT", "0.02"))
MAX_RISK_USDT = float(os.getenv("MAX_RISK_USDT", "25"))
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.5"))
TP_R_MULT = float(os.getenv("TP_R_MULT", "1.8"))
ENTRY_ATR_THRESH = float(os.getenv("ENTRY_ATR_THRESH", "0.6"))
ALLOW_MIN_QTY_ENTRY = os.getenv("ALLOW_MIN_QTY_ENTRY", "1") == "1"
MIN_NOTIONAL_USDT = float(os.getenv("MIN_NOTIONAL_USDT", "5"))

PICK_INTERVAL_MIN = int(os.getenv("PICK_INTERVAL_MIN", "15"))
TICK_INTERVAL_SEC = int(os.getenv("TICK_INTERVAL_SEC", "3"))

# ===== Shared state =====
RUN_FLAG = threading.Event()
CURRENT_STRATEGY = {"name": None}
STATS: Dict[str, Dict[str, float | int]] = {}

# ===== Helpers =====
def _ts_ms() -> str:
    return str(int(time.time() * 1000))

def _sign(params: Dict[str, str]) -> str:
    qs = "&".join([f"{k}={params[k]}" for k in sorted(params)])
    return hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()

async def tg(text: str):
    await get_bot().send_message(text)

# ===== Bybit Public =====
def get_klines(symbol=SYMBOL, interval="1", limit=180) -> List[List[str]]:
    r = requests.get(f"{BASE_URL}/v5/market/kline",
                     params={"category": CATEGORY, "symbol": symbol, "interval": interval, "limit": str(limit)},
                     timeout=10)
    return (r.json().get("result", {}) or {}).get("list", []) or []

def get_orderbook(symbol=SYMBOL, limit=50) -> Dict[str, Any]:
    r = requests.get(f"{BASE_URL}/v5/market/orderbook",
                     params={"category": CATEGORY, "symbol": symbol, "limit": str(limit)}, timeout=10)
    return r.json().get("result", {}) or {}

def get_recent_trades(symbol=SYMBOL, limit=200) -> List[Dict[str, Any]]:
    r = requests.get(f"{BASE_URL}/v5/market/recent-trade",
                     params={"category": CATEGORY, "symbol": symbol, "limit": str(limit)}, timeout=10)
    return (r.json().get("result", {}) or {}).get("list", []) or []

def get_instrument_info(symbol=SYMBOL) -> Dict[str, Any]:
    r = requests.get(f"{BASE_URL}/v5/market/instruments-info",
                     params={"category": CATEGORY, "symbol": symbol}, timeout=10)
    return r.json()

# ===== Bybit Private =====
def _private_get(path: str, params: Dict[str, str]) -> Dict[str, Any]:
    params = dict(params)
    params.update({"api_key": API_KEY, "timestamp": _ts_ms(), "recv_window": "5000"})
    params["sign"] = _sign(params)
    r = requests.get(f"{BASE_URL}{path}", params=params, timeout=10)
    return r.json()

def _private_post(path: str, params: Dict[str, str]) -> Dict[str, Any]:
    params = dict(params)
    params.update({"api_key": API_KEY, "timestamp": _ts_ms(), "recv_window": "5000"})
    params["sign"] = _sign(params)
    r = requests.post(f"{BASE_URL}{path}", data=params, timeout=10)
    return r.json()

def get_wallet_balance() -> float:
    j = _private_get("/v5/account/wallet-balance", {"accountType": "UNIFIED"})
    try:
        for acct in j.get("result", {}).get("list", []):
            for coin in acct.get("coin", []):
                if coin.get("coin") == "USDT":
                    return float(coin.get("walletBalance", 0.0))
    except Exception:
        pass
    return 0.0

def get_open_position() -> Optional[Tuple[str, float]]:
    j = _private_get("/v5/position/list", {"category": CATEGORY, "symbol": SYMBOL})
    try:
        lst = j.get("result", {}).get("list", [])
        for p in lst:
            sz = float(p.get("size", 0.0))
            if sz != 0.0:
                side = "long" if p.get("side", "Buy") == "Buy" else "short"
                return side, abs(sz)
    except Exception:
        return None
    return None

def place_market_order(side: str, qty: float, stop_loss: float, take_profit: float, reduce_only: bool = False) -> Dict[str, Any]:
    side_bybit = "Buy" if side == "long" else "Sell"
    params = {
        "category": CATEGORY,
        "symbol": SYMBOL,
        "side": side_bybit,
        "orderType": "Market",
        "qty": f"{qty:.6f}",          # строкой с 6 знаками ок
        "timeInForce": "IOC",         # для Market лучше IOC
        "reduceOnly": "true" if reduce_only else "false",
        "positionIdx": "0",           # 0 — one-way; 1/2 — hedge
    }
    # v5: takeProfit/stopLoss вместо tpPrice/slPrice
    if take_profit and take_profit > 0:
        params["takeProfit"] = f"{take_profit:.2f}"
        params["tpTriggerBy"] = "MarkPrice"
    if stop_loss and stop_loss > 0:
        params["stopLoss"] = f"{stop_loss:.2f}"
        params["slTriggerBy"] = "MarkPrice"

    resp = _private_post("/v5/order/create", params)
    try:
        log.debug("[ORDER][req] %s", params)
        log.debug("[ORDER][resp] %s", resp)
        if (resp or {}).get("retCode", 0) != 0:
            log.warning("[ORDER][error] retCode=%s retMsg=%s",
                        (resp or {}).get("retCode"), (resp or {}).get("retMsg"))
    except Exception:
        pass
    return resp

# ===== Features =====
def calc_atr(kl: List[List[str]], period=14) -> float:
    if not kl or len(kl) < period + 1:
        return 0.0
    data = list(reversed(kl))
    trs = []
    prev_close = float(data[0][4])
    for i in range(1, len(data)):
        h = float(data[i][2]); l = float(data[i][3]); c_prev = prev_close
        tr = max(h-l, abs(h-c_prev), abs(l-c_prev))
        trs.append(tr)
        prev_close = float(data[i][4])
    if len(trs) < period:
        return 0.0
    return sum(trs[-period:]) / period

def trend_from_ema(kl: List[List[str]], fast=12, slow=26) -> str:
    if len(kl) < slow + 1:
        return "flat"
    data = list(reversed(kl))
    closes = [float(x[4]) for x in data]
    kf = 2 / (fast + 1)
    ks = 2 / (slow + 1)
    ef = closes[0]; es = closes[0]
    for v in closes[1:]:
        ef = v * kf + ef * (1 - kf)
        es = v * ks + es * (1 - ks)
    return "up" if ef > es else ("down" if ef < es else "flat")

def build_features() -> Dict[str, Any]:
    kl1 = get_klines(SYMBOL, "1", 180)
    kl5 = get_klines(SYMBOL, "5", 72)
    ob = get_orderbook(SYMBOL, 50) or {}
    trades = get_recent_trades(SYMBOL, 200)

    # Early guard: if candles are missing, skip this tick entirely
    if not kl1 or not kl5:
        return {
            "data_ok": False,
            "price": 0.0,
            "atr_m1": 0.0,
            "atr_m5": 0.0,
            "trend_m1": "flat",
            "trend_m5": "flat",
            "tape": {"buy_ratio": 0.0},
            "tick_rate": 0.0,
            "sr": {},
            "orderbook": {"bids": [], "asks": []},
            "klines_m1": kl1 or [],
            "klines_m5": kl5 or [],
        }

    # Core metrics
    price = float(kl1[0][4]) if kl1 else 0.0
    # ATR(14) on 1m and 5m using classical TR definition
    atr1 = calc_atr(kl1, 14)
    atr5 = calc_atr(kl5, 14)
    # EMA(12/26) slope/trend classification
    t1 = trend_from_ema(kl1)
    t5 = trend_from_ema(kl5)

    # Tape metrics from recent trades:
    # buy_ratio = share of Buy prints in the recent trades window
    buys = sum(1 for t in (trades or [])
               if (t.get("S") or t.get("side")) == "Buy")
    total_trades = len(trades or [])
    buy_ratio = buys / max(1, total_trades)
    # tick_rate ~ trades per minute (Bybit recent-trade typically ~last ~3s)
    tick_rate = (total_trades / 3.0) * 60.0

    # Simple S/R from the 5m window:
    # nearest_resistance = max high over the lookback (excluding the most recent incomplete bar if possible)
    # nearest_support    = min low  over the same lookback
    sr = {}
    try:
        highs = [float(x[2]) for x in kl5[1:]] if len(kl5) > 1 else [float(x[2]) for x in kl5]
        lows  = [float(x[3]) for x in kl5[1:]] if len(kl5) > 1 else [float(x[3]) for x in kl5]
        if highs and lows:
            nearest_resistance = max(highs)
            nearest_support = min(lows)
            sr["nearest_resistance"] = nearest_resistance
            sr["nearest_support"] = nearest_support
            sr["dist_to_resistance"] = abs(nearest_resistance - price)
            sr["dist_to_support"] = abs(price - nearest_support)
    except Exception:
        pass

    ob_bids = ob.get("b", []) or ob.get("bids", []) or []
    ob_asks = ob.get("a", []) or ob.get("asks", []) or []

    return {
        "data_ok": True,
        "price": price,
        "atr_m1": atr1,
        "atr_m5": atr5,
        "trend_m1": t1,
        "trend_m5": t5,
        "tape": {"buy_ratio": buy_ratio},
        "tick_rate": tick_rate,
        "sr": sr,
        "orderbook": {"bids": ob_bids, "asks": ob_asks},
        "klines_m1": kl1, "klines_m5": kl5,
    }

# ===== Strategy registry =====
def make_ctx(loop: asyncio.AbstractEventLoop):
    async def on_entry(**kw):
        s = kw.get("strategy", "?"); side = kw.get("side"); qty = kw.get("qty"); price = kw.get("price")
        await tg(f"✅ <b>ENTRY</b> <code>{s}</code> {SYMBOL}\nSide: <b>{side.upper()}</b> Qty: <code>{qty}</code> @ <code>{price:.2f}</code>\n{kw.get('indicator','')}")
        STATS.setdefault(s, {"pnl": 0.0, "wins": 0, "losses": 0, "trades": 0})

    async def on_exit(**kw):
        s = kw.get("strategy", "?"); side = kw.get("side", "?")
        entry_p = kw.get("entry_price"); exit_p = kw.get("price")
        qty = float(kw.get("qty", 0.0))
        pnl = 0.0
        if entry_p and exit_p and qty:
            pnl = (exit_p - entry_p) * qty * (1 if side == "long" else -1)
        st = STATS.setdefault(s, {"pnl": 0.0, "wins": 0, "losses": 0, "trades": 0})
        st["pnl"] += pnl
        st["trades"] += 1
        if pnl >= 0: st["wins"] += 1
        else: st["losses"] += 1
        await tg(f"🧾 <b>EXIT</b> <code>{s}</code> {SYMBOL}\nside={side} pnl=<code>{pnl:.2f}</code> reason={kw.get('reason','?')}")

    # обёртки для вызова из потока
    def _on_entry_sync(**kw):
        return asyncio.run_coroutine_threadsafe(on_entry(**kw), loop).result()

    def _on_exit_sync(**kw):
        return asyncio.run_coroutine_threadsafe(on_exit(**kw), loop).result()

    def _notify_sync(msg: str):
        return asyncio.run_coroutine_threadsafe(tg(msg), loop).result()

    return {"on_entry": _on_entry_sync, "on_exit": _on_exit_sync, "notify": _notify_sync}

STRATS = {
    "momentum": momentum.run_once,
    "breakout": breakout.run_once,
    "density": density.run_once,
    "knife": knife.run_once,
}

_last_pick_ts = 0

async def pick_and_maybe_switch(features: Dict[str, Any]):
    global _last_pick_ts, CURRENT_STRATEGY
    now = time.time()
    if not features.get("data_ok"):
        return
    if CURRENT_STRATEGY["name"] is None or (now - _last_pick_ts) >= PICK_INTERVAL_MIN * 60:
        payload = {
            "symbol": SYMBOL,
            "equity": get_wallet_balance(),
            "risk_pct": RISK_PCT,
            "features": features,
            "klines_m1": features.get("klines_m1"),
            "klines_m5": features.get("klines_m5"),
            "orderbook": features.get("orderbook"),
            "time_utc": datetime.now(timezone.utc).isoformat(),
        }
        await tg("🤖 <b>AI request</b> sending fresh market snapshot…")
        choice = pick_strategy(payload)
        await tg(f"🤖 <b>AI choice</b>: <code>{choice['strategy']}</code> (conf {choice.get('confidence',0):.2f})\n<i>{choice.get('reason','')}</i>")
        _last_pick_ts = now
        new_name = choice["strategy"]
        old = CURRENT_STRATEGY.get("name")
        if old != new_name:
            CURRENT_STRATEGY["name"] = new_name
            log.info("[PICK] switch %s -> %s", old, new_name)

# ===== Background strategy loop (thread) =====
def _strategy_thread(loop: asyncio.AbstractEventLoop):
    ctx = make_ctx(loop)

    while RUN_FLAG.is_set():
        try:
            feats = build_features()
            # Early return if data is missing, with rate-limited warning
            if not feats.get("data_ok"):
                global _last_no_data_warn_ts
                if time.time() - _last_no_data_warn_ts > 15:
                    log.debug("[DATA] candles not ready — skipping tick")
                    _last_no_data_warn_ts = time.time()
                time.sleep(TICK_INTERVAL_SEC)
                continue
            fut = asyncio.run_coroutine_threadsafe(pick_and_maybe_switch(feats), loop)
            try:
                fut.result(timeout=25)
            except Exception as e:
                log.warning("AI pick error: %s", e)

            name = CURRENT_STRATEGY.get("name")
            if name and name in STRATS:
                # Bybit instrument filters — used for qty rounding and price ticks
                lot, min_qty, price_step = get_instrument_filters_cached(ttl=60)

                class Filters:
                    pass
                Filters.qty_step = lot
                Filters.min_qty = min_qty
                Filters.price_step = price_step

                params = {
                    "risk_pct": RISK_PCT,
                    "max_risk_usdt": MAX_RISK_USDT,
                    "sl_atr_mult": SL_ATR_MULT,
                    "tp_r_mult": TP_R_MULT,
                    "entry_atr_thresh": ENTRY_ATR_THRESH,
                    "allow_min_qty_entry": ALLOW_MIN_QTY_ENTRY,
                    "min_notional_usdt": MIN_NOTIONAL_USDT,
                }
                ctx2 = {
                    "features": feats,
                    "params": params,
                    "on_entry": ctx["on_entry"],
                    "on_exit":  ctx["on_exit"],
                    "notify":   ctx["notify"],
                    "get_open_position": get_open_position,
                    "get_wallet_balance": get_wallet_balance,
                    "place_order": place_market_order,
                }
                STRATS[name](Filters, ctx2)
        except Exception as e:
            log.warning("[STRAT] error: %s", e)
        time.sleep(TICK_INTERVAL_SEC)

# ===== UI callbacks (async, для aiogram) =====
_worker_thread: Optional[threading.Thread] = None

async def ui_start():
    if RUN_FLAG.is_set():
        await tg("⚠️ Уже запущено.")
        return
    RUN_FLAG.set()
    bal = get_wallet_balance()
    await tg(f"🚀 <b>Старт</b> {SYMBOL}\nБаланс: <code>{bal:.2f} USDT</code>\nРиск: <code>{RISK_PCT*100:.1f}%</code>")
    loop = asyncio.get_running_loop()
    global _worker_thread
    _worker_thread = threading.Thread(target=_strategy_thread, args=(loop,), daemon=True)
    _worker_thread.start()

async def ui_stop():
    if not RUN_FLAG.is_set():
        await tg("ℹ️ Уже остановлено.")
        return
    RUN_FLAG.clear()
    await tg("🛑 Остановлено.")

async def ui_balance():
    bal = get_wallet_balance()
    await tg(f"💰 Баланс: <code>{bal:.2f} USDT</code>")

async def ui_positions():
    pos = get_open_position()
    if not pos:
        await tg("📜 Открытых позиций нет.")
    else:
        side, qty = pos
        await tg(f"📜 Позиция: <b>{side.upper()}</b> qty=<code>{qty}</code>")

async def ui_stats():
    if not STATS:
        await tg("Пока сделок нет.")
        return
    lines = ["📊 <b>Статистика за сессию</b>"]
    for k, v in STATS.items():
        wr = (v["wins"] / v["trades"]) if v["trades"] else 0.0
        lines.append(f"• {k}: pnl=<code>{v['pnl']:.2f}</code> | W/L {v['wins']}/{v['losses']} | WR {wr:.2%}")
    await tg("\n".join(lines))

# ===== Entrypoint =====
async def main():
    bot = get_bot()
    bot.set_handlers(
        on_start=ui_start,
        on_stop=ui_stop,
        on_balance=ui_balance,
        on_positions=ui_positions,
        on_stats=ui_stats,
    )
    await bot.send_menu()
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
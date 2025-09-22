# main.py
import os
import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Dict, Any, Tuple, List

import pandas as pd
import time
import re

# ---- Optional SSL relax (env BYBIT_VERIFY_SSL=false) ----
if os.getenv("BYBIT_VERIFY_SSL", "true").lower() == "false":
    os.environ["PYTHONHTTPSVERIFY"] = "0"

from dotenv import load_dotenv
from pybit.unified_trading import HTTP
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from aiogram.enums import ParseMode
from aiogram.utils.chat_action import ChatActionSender
from aiogram.exceptions import TelegramBadRequest

import openai

# ------------------ .env & logging ------------------
load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("tradebot.log", encoding="utf-8")]
)
log = logging.getLogger("MAIN")

# ------------------ Enums/Dataclasses ------------------
class Regime(str, Enum):
    TREND = "trend"
    FLAT = "flat"
    HOLD = "hold"

class Side(str, Enum):
    BUY = "Buy"
    SELL = "Sell"
    NONE = "None"

@dataclass
class AIDecision:
    regime: Regime
    side: Side          # For HOLD, may be NONE
    sl_ticks: Optional[int] = None   # optional, can be from .env instead
    comment: str = ""

@dataclass
class MdFilters:
    tick_size: float
    qty_step: float
    min_qty: float

@dataclass
class PositionInfo:
    size: float
    side: Side
    avg_price: float

@dataclass
class MarketData:
    last_price: float
    filters: MdFilters
    kline_1m: List[List[Any]]  # raw bybit kline list
    position: PositionInfo
    balance_usdt: float

# ------------------ Global state ------------------
class BotState:
    def __init__(self):
        self.is_trading = False
        self.current_regime: Regime = Regime.HOLD
        self.current_side: Side = Side.NONE
        self.loop_task: Optional[asyncio.Task] = None
        self.last_ai_text: str = ""
        self.flat_entry_order_id: Optional[str] = None
        self.sl_order_id: Optional[str] = None
        self.tp_order_id: Optional[str] = None
        self.last_sl_hit_at: Optional[datetime] = None
        self.symbol = os.getenv("SYMBOL", "BTCUSDT")
        self.category = "linear"
        self.last_flat_prev_ts: Optional[int] = None

STATE = BotState()

# ------------------ Clients ------------------
def make_bybit() -> HTTP:
    testnet = os.getenv("BYBIT_TESTNET", "true").lower() == "true"
    api_key = os.getenv("BYBIT_API_KEY_TEST" if testnet else "BYBIT_API_KEY_MAIN")
    api_secret = os.getenv("BYBIT_SECRET_KEY_TEST" if testnet else "BYBIT_SECRET_KEY_MAIN")
    # pybit HTTP takes testnet=bool and uses requests under the hood.
    # No direct verify flag, SSL relaxed above via env if needed.
    return HTTP(testnet=testnet, api_key=api_key, api_secret=api_secret)

BYBIT = make_bybit()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

BOT = Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None
DP = Dispatcher()

# OpenAI
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ------------------ Config from .env ------------------
LOT_SIZE_USDT = float(os.getenv("LOT_SIZE_USDT", "50"))
LEVERAGE = int(os.getenv("LEVERAGE", "5"))

# Flat mode config
FLAT_ENTRY_TICKS = int(os.getenv("FLAT_ENTRY_TICKS", "6"))  # от экстремума внутрь
# SL in ticks for both modes (if AI doesn’t override)
SL_TICKS = int(os.getenv("SL_TICKS", "80"))                 # настраивается в .env

# Take-profit rule for flat:
# "на 2 тика ниже/выше противоположного от уровня открытия позиции края тела прошлой минуты"
TP_BODY_OFFSET_TICKS = int(os.getenv("TP_BODY_OFFSET_TICKS", "2"))  # TP offset from body edge (ticks)
MARKET_BAND_EXTRA_TICKS = int(os.getenv("MARKET_BAND_EXTRA_TICKS", "4"))  # how many ticks deeper to push IOC Limit fallback

# Interval and bootstrap hours
POLL_INTERVAL_SEC = int(os.getenv("POLL_INTERVAL_SEC", "60"))
BOOTSTRAP_HOURS = int(os.getenv("BOOTSTRAP_HOURS", "5"))
POLL_TICK_MS = int(os.getenv("POLL_TICK_MS", "1000"))  # how often to check market (ms)
AI_POLL_SEC = int(os.getenv("AI_POLL_SEC", "60"))      # how often to refresh AI decision (sec)

# Retry settings
RETRY_ATTEMPTS = 10
RETRY_DELAY_SEC = 1

# ------------------ Helpers ------------------
def q(price: float, step: float) -> float:
    return float(Decimal(str(price)).quantize(Decimal(str(step)), rounding=ROUND_DOWN))

def normalize_price(price: float, tick: float) -> float:
    return q(price, tick)

def normalize_qty(qty: float, step: float, min_qty: float) -> float:
    n = q(qty, step)
    return max(n, min_qty)

def ts_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

async def tg_send(text: str):
    if not BOT or not TELEGRAM_CHAT_ID:
        log.debug("[TG] skip (no token/chat id): %s", text)
        return
    try:
        async with ChatActionSender.typing(bot=BOT, chat_id=TELEGRAM_CHAT_ID):
            await BOT.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    except Exception as e:
        log.error("[TG] send error: %s", e)

# --- Safe message edit helper ---
async def safe_edit(message, text: str, markup=None, parse_mode=None):
    """Edit message text safely: ignore 'message is not modified' errors."""
    try:
        await message.edit_text(text, reply_markup=markup, parse_mode=parse_mode)
    except TelegramBadRequest as e:
        if "message is not modified" in str(e).lower():
            return
        raise

def _kline_to_df(kline: List[List[Any]]) -> pd.DataFrame:
    # Bybit v5 returns list entries like: [startTime, open, high, low, close, volume, turnover]
    df = pd.DataFrame(kline, columns=["ts","open","high","low","close","volume","turnover"])
    for col in ["open","high","low","close","volume","turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.sort_values("ts")
    return df

def _body_edges(prev: pd.Series) -> Tuple[float, float]:
    o, c = float(prev["open"]), float(prev["close"])
    body_low = min(o, c)
    body_high = max(o, c)
    return body_low, body_high

# ------------------ Bybit wrappers ------------------
def bybit_filters(symbol: str) -> MdFilters:
    r = BYBIT.get_instruments_info(category=STATE.category, symbol=symbol)
    it = r["result"]["list"][0]
    tick = float(it["priceFilter"]["tickSize"])
    qty_step = float(it["lotSizeFilter"]["qtyStep"])
    min_qty = float(it["lotSizeFilter"]["minOrderQty"])
    return MdFilters(tick, qty_step, min_qty)

def bybit_last_price(symbol: str) -> float:
    r = BYBIT.get_tickers(category=STATE.category, symbol=symbol)
    return float(r["result"]["list"][0]["lastPrice"])

def bybit_best_prices(symbol: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (bid1, ask1) from tickers if available."""
    r = BYBIT.get_tickers(category=STATE.category, symbol=symbol)
    it = r["result"]["list"][0]
    bid = float(it.get("bid1Price")) if it.get("bid1Price") is not None else None
    ask = float(it.get("ask1Price")) if it.get("ask1Price") is not None else None
    return bid, ask

def bybit_kline_1m(symbol: str, minutes: int) -> List[List[Any]]:
    # Bybit max limit per call is ~200; 5h=300 bars → do two calls 200 + remainder
    need = minutes
    out: List[List[Any]] = []
    while need > 0:
        lim = 200 if need > 200 else need
        rr = BYBIT.get_kline(category=STATE.category, symbol=symbol, interval="1", limit=lim)
        chunk = rr["result"]["list"]
        out = chunk + out  # API returns most-recent-first; accumulate and sort later
        need -= lim
        if need > 0:
            # tiny pause to avoid rate limits (sync, we are inside running event loop)
            time.sleep(0.15)
    return out

def bybit_wallet_usdt() -> float:
    r = BYBIT.get_wallet_balance(accountType="UNIFIED")
    for coin in r["result"]["list"][0]["coin"]:
        if coin["coin"] == "USDT":
            return float(coin["walletBalance"])
    return 0.0

def bybit_position(symbol: str) -> PositionInfo:
    rr = BYBIT.get_positions(category=STATE.category, symbol=symbol)
    if not rr["result"]["list"]:
        return PositionInfo(0.0, Side.NONE, 0.0)
    p = rr["result"]["list"][0]
    size = float(p.get("size", 0) or 0)
    side = Side(p.get("side", "None")) if size > 0 else Side.NONE
    avg_price = float(p.get("avgPrice", 0) or 0)
    return PositionInfo(size, side, avg_price)

def bybit_cancel_all(symbol: str):
    return BYBIT.cancel_all_orders(category=STATE.category, symbol=symbol)

def bybit_place_order(*, symbol: str, side: Side, order_type: str, qty: float, price: Optional[float] = None,
                      reduce_only: bool = False, time_in_force: str = "GTC",
                      trigger_price: Optional[float] = None, trigger_by: str = "LastPrice",
                      trigger_direction: Optional[int] = None) -> Dict[str, Any]:
    payload = dict(
        category=STATE.category, symbol=symbol, side=side.value, orderType=order_type,
        qty=str(qty), timeInForce=time_in_force
    )
    if price is not None:
        payload["price"] = str(price)
    if reduce_only:
        payload["reduceOnly"] = True
    if trigger_price is not None:
        payload["triggerPrice"] = str(trigger_price)
        payload["triggerBy"] = trigger_by
    if trigger_direction is not None:
        payload["triggerDirection"] = trigger_direction
    return BYBIT.place_order(**payload)

def bybit_set_leverage(symbol: str, lev: int):
    return BYBIT.set_leverage(category=STATE.category, symbol=symbol, buyLeverage=str(lev), sellLeverage=str(lev))

def bybit_trading_stop(symbol: str, *, take_profit: Optional[float]=None, stop_loss: Optional[float]=None,
                       tp_trigger: str="LastPrice", sl_trigger: str="LastPrice"):
    # /v5/position/trading-stop (Full mode for the position)
    payload = dict(category=STATE.category, symbol=symbol, tpslMode="Full")
    if take_profit is not None:
        payload["takeProfit"] = str(take_profit)
        payload["tpTriggerBy"] = tp_trigger
    if stop_loss is not None:
        payload["stopLoss"] = str(stop_loss)
        payload["slTriggerBy"] = sl_trigger
    return BYBIT.set_trading_stop(**payload)

def bybit_get_order(order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
    rr = BYBIT.get_open_orders(category=STATE.category, symbol=symbol)
    for o in rr["result"]["list"]:
        if o.get("orderId") == order_id:
            return o
    return None

def bybit_open_orders(symbol: str) -> List[Dict[str, Any]]:
    rr = BYBIT.get_open_orders(category=STATE.category, symbol=symbol)
    return rr.get("result", {}).get("list", []) or []

# ------------------ Market snapshot ------------------
def read_market(symbol: str, bootstrap_hours: int) -> MarketData:
    filters = bybit_filters(symbol)
    last = bybit_last_price(symbol)
    minutes = bootstrap_hours * 60
    kl = bybit_kline_1m(symbol, minutes)
    pos = bybit_position(symbol)
    bal = bybit_wallet_usdt()
    return MarketData(last, filters, kl, pos, bal)

# ------------------ AI ------------------
def ai_prompt(symbol: str, df: pd.DataFrame, md: MarketData) -> str:
    # Last rows: prev(−2) and curr(−1)
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    body_low, body_high = _body_edges(prev)
    p = {
        "symbol": symbol,
        "now": ts_now(),
        "last_price": md.last_price,
        "prev_high": float(prev["high"]),
        "prev_low": float(prev["low"]),
        "prev_body_low": body_low,
        "prev_body_high": body_high,
        "curr_open": float(curr["open"]),
        "curr_close": float(curr["close"]),
        "position": {"side": md.position.side.value, "size": md.position.size, "avg_price": md.position.avg_price},
        "balance_usdt": md.balance_usdt
    }
    return (
        "You are a trading decision engine. Return ONLY compact JSON, no extra text.\n"
        "Decide regime among: \"trend\", \"flat\", or \"hold\".\n"
        "If regime is trend or flat, also return trading side: \"Buy\" or \"Sell\".\n"
        "If regime is hold, side can be \"None\".\n"
        "Optionally you may include sl_ticks integer override.\n\n"
        f"Market snapshot:\n{json.dumps(p, ensure_ascii=False)}\n\n"
        "JSON schema:\n"
        "{ \"regime\": \"trend|flat|hold\", \"side\": \"Buy|Sell|None\", \"sl_ticks\": int|null, \"comment\": string }\n"
        "Если флэт и тело последней закрытой свечи меньше 2500 тиков - возвращай hold."
        "Сигнал на тренд даем только если тренд подтвержден двумя последними закрытыми свечами."
        "Если флэт и сторона buy и нижняя тень последней закрытой свечи больше 2500 тиков - возвращай hold."
        "Если флэт и сторона sell и верхняя тень последней закрытой свечи больше 2500 тиков - возвращай hold."
    )

def _extract_json_block(text: str) -> str:
    text = text.strip()
    # Try to capture the first JSON object block
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return m.group(0)
    # Fallback: strip Markdown code fences like ```json ... ```
    if text.startswith("```") and text.endswith("```"):
        inner = text.strip('`')
        parts = inner.split('\n', 1)
        if len(parts) == 2:
            return parts[1]
    return text

def parse_ai(text: str) -> AIDecision:
    data = json.loads(_extract_json_block(text))
    return AIDecision(
        regime=Regime(data["regime"]),
        side=Side(data.get("side", "None")),
        sl_ticks=data.get("sl_ticks"),
        comment=data.get("comment", "")
    )

# --- Helper: pretty format AI decision for Telegram
def pretty_ai_decision(dec: AIDecision) -> str:
    parts = [f"Режим: <b>{dec.regime.value}</b>"]
    if dec.regime != Regime.HOLD:
        parts.append(f"Направление: <b>{dec.side.value}</b>")
    else:
        parts.append("Направление: <b>None</b>")
    if dec.sl_ticks is not None:
        parts.append(f"SL ticks: <b>{dec.sl_ticks}</b>")
    if dec.comment:
        parts.append(f"Комментарий: {dec.comment}")
    return "🤖 Обновление от ИИ:\n" + "\n".join(parts)

async def ask_ai(symbol: str, df: pd.DataFrame, md: MarketData) -> AIDecision:
    prompt = ai_prompt(symbol, df, md)
    log.info("[AI] request: %s", prompt.replace("\n", " ")[:500])
    try:
        resp = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model=OPENAI_MODEL,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.choices[0].message.content
        STATE.last_ai_text = text
        log.info("[AI] raw: %s", text)
        return parse_ai(text)
    except Exception as e:
        log.error("[AI] error: %s", e)
        await tg_send(f"🤖 <b>Ошибка ИИ</b>: {e}")
        # fallback: keep current, or HOLD
        return AIDecision(regime=Regime.HOLD, side=Side.NONE, comment="fallback")

# ------------------ Trading primitives with retries ------------------
async def retry_place(func, *, descr: str) -> Dict[str, Any]:
    last_err = None
    for i in range(1, RETRY_ATTEMPTS+1):
        try:
            r = func()
            log.info("[ORDER] %s OK try=%d: %s", descr, i, r.get("result", {}))
            return r
        except Exception as e:
            last_err = e
            log.warning("[ORDER] %s fail try=%d: %s", descr, i, e)
            await asyncio.sleep(RETRY_DELAY_SEC)
    raise RuntimeError(f"ORDER FAILED after {RETRY_ATTEMPTS}: {descr} :: {last_err}")

async def market_close_all(symbol: str):
    pos = bybit_position(symbol)
    if pos.size <= 0:
        return
    side = Side.SELL if pos.side == Side.BUY else Side.BUY
    qty = pos.size
    await retry_place(lambda: bybit_place_order(
        symbol=symbol, side=side, order_type="Market", qty=qty, time_in_force="IOC", reduce_only=True
    ), descr="close_position_market")

async def ensure_sl_tp(symbol: str, sl_price: Optional[float], tp_price: Optional[float]) -> None:
    """
    Use /position/trading-stop in Full mode. Treat ErrCode 34040 (not modified) as success.
    Retry a few times on transient errors; do NOT close the position just because SL/TP already set.
    """
    if sl_price is None and tp_price is None:
        return

    def call_once():
        return bybit_trading_stop(symbol, take_profit=tp_price, stop_loss=sl_price)

    tries = 0
    while tries < 3:
        try:
            r = call_once()
            log.info("[SLTP] set_trading_stop ok: %s", r.get("result", {}))
            return
        except Exception as e:
            msg = str(e)
            # Bybit returns ErrCode 34040 when values are unchanged → treat as success
            if "34040" in msg or "not modified" in msg.lower():
                log.info("[SLTP] not modified → already set; ok")
                return
            tries += 1
            log.warning("[SLTP] set_trading_stop retry %d: %s", tries, e)
            await asyncio.sleep(0.5)

    # After retries, if still failing for another reason, log but DO NOT auto-close the position here
    log.error("[SLTP] failed to set_trading_stop after retries; keeping position. sl=%s tp=%s", sl_price, tp_price)

async def ensure_order_filled_or_cancel(symbol: str, order_id: str, timeout_sec: int = 45) -> bool:
    """
    Poll open orders; if still open after timeout, cancel all and return False.
    """
    deadline = asyncio.get_event_loop().time() + timeout_sec
    while asyncio.get_event_loop().time() < deadline:
        od = bybit_get_order(order_id, symbol)
        if od is None:
            # not in open orders → filled or cancelled; assume filled and verify position
            pos = bybit_position(symbol)
            if pos.size > 0:
                return True
            else:
                return False
        await asyncio.sleep(1)
    # timeout → cancel
    try:
        bybit_cancel_all(symbol)
    except Exception:
        pass
    return False

# ------------------ Regime executors ------------------
async def do_trend(md: MarketData, dec: AIDecision):
    symbol = STATE.symbol
    f = md.filters
    desired = dec.side
    if desired not in (Side.BUY, Side.SELL):
        return

    # If opposite position exists → close once
    if md.position.size > 0 and md.position.side != desired:
        await market_close_all(symbol)
        return  # wait next tick

    # If no position and no working orders → place entry (market with price-band fallback as before), but do NOT block waiting
    if md.position.size == 0:
        open_orders = bybit_open_orders(symbol)
        if not open_orders:
            md2 = read_market(symbol, 1)
            qty = normalize_qty(LOT_SIZE_USDT / max(md2.last_price, 1e-8), f.qty_step, f.min_qty)
            # Try market first with band handling
            try:
                await retry_place(lambda: bybit_place_order(
                    symbol=symbol, side=desired, order_type="Market", qty=qty, time_in_force="IOC"
                ), descr=f"trend_enter_{desired.value}")
                await tg_send(f"📈 TREND заявка {desired.value} qty={qty} (market IOC)")
            except Exception as e:
                msg = str(e).lower()
                if ("maximum buying price" in msg) or ("maximum selling price" in msg) or ("30208" in msg):
                    bid, ask = bybit_best_prices(symbol)
                    base_px = (ask if desired == Side.BUY else bid) or md2.last_price
                    base_px = normalize_price(base_px, f.tick_size)
                    # Escalate a couple of IOC limit attempts synchronously (non-blocking for fills)
                    for i in range(1, 3):
                        px_try = base_px + i * MARKET_BAND_EXTRA_TICKS * f.tick_size if desired == Side.BUY else base_px - i * MARKET_BAND_EXTRA_TICKS * f.tick_size
                        px_try = normalize_price(px_try, f.tick_size)
                        try:
                            await retry_place(lambda: bybit_place_order(
                                symbol=symbol, side=desired, order_type="Limit", qty=qty, price=px_try, time_in_force="IOC"
                            ), descr=f"trend_enter_limit_{desired.value}")
                            await tg_send(f"📈 TREND заявка {desired.value} qty={qty} (IOC @ {px_try})")
                            break
                        except Exception:
                            continue
                else:
                    raise
        # if order exists, we just wait next tick
        return

    # If position exists → maintain/update SL (trailing per ticks)
    sl_ticks = dec.sl_ticks if dec.sl_ticks is not None else SL_TICKS
    fresh = read_market(symbol, 1)
    if fresh.position.size > 0:
        if fresh.position.side == Side.BUY:
            sl_price = normalize_price(fresh.last_price - sl_ticks * f.tick_size, f.tick_size)
        else:
            sl_price = normalize_price(fresh.last_price + sl_ticks * f.tick_size, f.tick_size)
        await ensure_sl_tp(symbol, sl_price=sl_price, tp_price=None)
        log.info("[TREND] SL set/update at %f (ticks=%d)", sl_price, sl_ticks)

async def do_flat(md: MarketData, dec: AIDecision):
    symbol = STATE.symbol
    f = md.filters
    desired = dec.side
    if desired not in (Side.BUY, Side.SELL):
        return

    df = _kline_to_df(md.kline_1m)
    prev = df.iloc[-2]
    prev_ts = int(prev["ts"])  # closed candle start time (ms)
    prev_high = float(prev["high"])  # noqa: F841
    prev_low = float(prev["low"])   # noqa: F841
    body_low, body_high = _body_edges(prev)

    # 1) If opposite position exists → close immediately, then proceed next tick
    if md.position.size > 0 and md.position.side != desired:
        await market_close_all(symbol)
        # also cancel any resting orders
        try:
            bybit_cancel_all(symbol)
        except Exception:
            pass
        STATE.last_flat_prev_ts = prev_ts
        return

    # 2) If no position:
    open_orders = bybit_open_orders(symbol)
    need_requote = (STATE.last_flat_prev_ts is None) or (prev_ts != STATE.last_flat_prev_ts)

    if md.position.size == 0:
        # Every NEW closed minute: cancel any old unfilled order and place fresh by new extremes and current AI side
        if need_requote and open_orders:
            try:
                bybit_cancel_all(symbol)
                await tg_send("♻️ FLAT ре-котировка: отменил старую заявку")
            except Exception:
                pass
            open_orders = []

        # If no working orders now → place new
        if not open_orders:
            if desired == Side.BUY:
                # BUY entry: 6 ticks ABOVE previous minute's LOW (inside range)
                entry_price = normalize_price(prev_low + FLAT_ENTRY_TICKS * f.tick_size, f.tick_size)
            else:
                # SELL entry: 6 ticks BELOW previous minute's HIGH (inside range)
                entry_price = normalize_price(prev_high - FLAT_ENTRY_TICKS * f.tick_size, f.tick_size)
            qty = normalize_qty(LOT_SIZE_USDT / max(entry_price, 1e-8), f.qty_step, f.min_qty)
            await retry_place(lambda: bybit_place_order(
                symbol=symbol, side=desired, order_type="Limit", qty=qty, price=entry_price, time_in_force="GTC"
            ), descr=f"flat_limit_enter_{desired.value}@{entry_price}")
            await tg_send(f"🤏 FLAT заявка {desired.value} {qty} @ {entry_price}")
        # remember latest prev bar
        STATE.last_flat_prev_ts = prev_ts
        return

    # 3) If position exists and direction matches AI → do nothing, just ensure SL/TP each tick
    fresh = read_market(symbol, 1)
    sl_ticks = dec.sl_ticks if dec.sl_ticks is not None else SL_TICKS
    if desired == Side.BUY:
        sl_price = normalize_price(fresh.position.avg_price - sl_ticks * f.tick_size, f.tick_size)
        tp_edge = body_high
        tp_price = normalize_price(tp_edge - TP_BODY_OFFSET_TICKS * f.tick_size, f.tick_size)
        if tp_price <= fresh.position.avg_price:
            tp_price = normalize_price(fresh.position.avg_price + 2 * f.tick_size, f.tick_size)
    else:
        sl_price = normalize_price(fresh.position.avg_price + sl_ticks * f.tick_size, f.tick_size)
        tp_edge = body_low
        tp_price = normalize_price(tp_edge + TP_BODY_OFFSET_TICKS * f.tick_size, f.tick_size)
        if tp_price >= fresh.position.avg_price:
            tp_price = normalize_price(fresh.position.avg_price - 2 * f.tick_size, f.tick_size)

    await ensure_sl_tp(symbol, sl_price=sl_price, tp_price=tp_price)
    await tg_send(f"🛡 SL={sl_price} 🎯 TP={tp_price}")
    STATE.last_flat_prev_ts = prev_ts

async def do_hold(md: MarketData, dec: AIDecision):
    # No trading, just keep polling AI
    log.info("[HOLD] No trades")

# ------------------ Main trading loop ------------------
async def trading_loop():
    symbol = STATE.symbol
    await tg_send("🚀 Старт торговли")

    # leverage
    try:
        bybit_set_leverage(symbol, LEVERAGE)
        log.info("[LEV] set leverage=%dx", LEVERAGE)
    except Exception as e:
        log.warning("[LEV] fail set leverage: %s", e)

    # initial bootstrap (5 hours history) with fallback and clear errors
    try:
        md = read_market(symbol, BOOTSTRAP_HOURS)
        df = _kline_to_df(md.kline_1m)
        await tg_send("📥 Подгрузка данных за 5 часов завершена")
    except Exception as e:
        log.exception("[BOOTSTRAP] failed 5h load: %s", e)
        await tg_send("❌ Ошибка загрузки истории за 5 часов. Пробую загрузить 60 минут…")
        try:
            md = read_market(symbol, 1)
            df = _kline_to_df(md.kline_1m)
            await tg_send("📥 Подгружено за 60 минут (fallback)")
        except Exception as e2:
            log.exception("[BOOTSTRAP] fallback 60m failed: %s", e2)
            await tg_send("⛔️ Не удалось загрузить историю (даже 60 минут). Останавливаю торговлю.")
            STATE.is_trading = False
            return

    # first AI decision
    dec = await ask_ai(symbol, df, md)
    STATE.current_regime = dec.regime
    STATE.current_side = dec.side
    await tg_send(f"🧭 Режим: <b>{dec.regime.value}</b> | Направление: <b>{dec.side.value}</b>")

    last_ai_time = 0.0
    while STATE.is_trading:
        try:
            now_monotonic = asyncio.get_event_loop().time()
            # Refresh market snapshot cheaply (1m klines window)
            md = read_market(symbol, 1)
            df = _kline_to_df(md.kline_1m)

            # Periodic AI refresh
            if (now_monotonic - last_ai_time) >= AI_POLL_SEC:
                dec_new = await ask_ai(symbol, df, md)
                regime_changed = dec_new.regime != STATE.current_regime
                side_changed = dec_new.side != STATE.current_side and dec_new.regime != Regime.HOLD

                # Special handling for FLAT: if we have an open position and FLAT persists
                if STATE.current_regime == Regime.FLAT and dec_new.regime == Regime.FLAT:
                    pos_now = md.position
                    if pos_now.size > 0:
                        if dec_new.side == pos_now.side:
                            # Same direction → keep holding, do NOT close, do NOT cancel orders
                            STATE.current_side = dec_new.side
                            STATE.current_regime = Regime.FLAT
                            last_ai_time = now_monotonic
                            # skip generic switching logic
                            pass
                        else:
                            # Opposite direction → cancel orders and close market, then follow new direction
                            await tg_send("🔁 FLAT: смена направления при открытой позиции → закрываю по рынку")
                            try:
                                await bybit_cancel_all(symbol)
                            except Exception:
                                pass
                            await market_close_all(symbol)
                            STATE.current_regime = Regime.FLAT
                            STATE.current_side = dec_new.side
                            last_ai_time = now_monotonic
                            await tg_send(pretty_ai_decision(dec_new))
                    else:
                        # No position: if side changed under FLAT, cancel resting orders so do_flat() can place a fresh one
                        if side_changed:
                            try:
                                await bybit_cancel_all(symbol)
                            except Exception:
                                pass
                            STATE.current_side = dec_new.side
                            STATE.current_regime = Regime.FLAT
                            last_ai_time = now_monotonic
                            await tg_send(pretty_ai_decision(dec_new))
                    # Done with FLAT special flow; don't run generic switch below
                else:
                    if regime_changed or side_changed:
                        await tg_send("🔁 Переключение режима/направления → Закрываю позиции")
                        try:
                            await bybit_cancel_all(symbol)
                        except Exception:
                            pass
                        await market_close_all(symbol)
                        STATE.current_regime = dec_new.regime
                        STATE.current_side = dec_new.side
                        await tg_send(pretty_ai_decision(dec_new))
                    last_ai_time = now_monotonic

            # Per-tick execution according to current regime
            if STATE.current_regime == Regime.TREND:
                await do_trend(md, AIDecision(STATE.current_regime, STATE.current_side))
            elif STATE.current_regime == Regime.FLAT:
                await do_flat(md, AIDecision(STATE.current_regime, STATE.current_side))
            else:
                await do_hold(md, AIDecision(STATE.current_regime, STATE.current_side))

            # Immediate AI ping after exit (best effort)
            if md.position.size == 0 and STATE.last_sl_hit_at is None:
                STATE.last_sl_hit_at = datetime.now(timezone.utc)
                # Non-blocking: just schedule earlier next AI poll
                last_ai_time = 0.0
            elif md.position.size > 0:
                STATE.last_sl_hit_at = None

            await asyncio.sleep(POLL_TICK_MS / 1000.0)
        except Exception as e:
            log.exception("[LOOP] error: %s", e)
            await asyncio.sleep(1)

    await tg_send("⏹️ Торговля остановлена")

# ------------------ Telegram UI ------------------
def kb():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="▶️ Старт"), KeyboardButton(text="⏹️ Стоп")],
            [KeyboardButton(text="💰 Баланс"), KeyboardButton(text="🚫 Закрыть всё")],
        ],
        resize_keyboard=True,
        one_time_keyboard=False,
        input_field_placeholder="Выберите действие…",
        selective=False,
    )

@DP.message(F.text.regexp(r"^/start$"))
async def start_cmd(msg: Message):
    await msg.answer(
        "Бот готов. Кнопки снизу.\n\n"
        "▶️ Старт — запустить торговлю\n"
        "⏹️ Стоп — остановить и закрыть всё\n"
        "💰 Баланс — показать баланс\n"
        "🚫 Закрыть всё — закрыть позиции и отменить заявки",
        reply_markup=kb(),
    )


# --- Reply keyboard handlers ---
@DP.message(F.text == "▶️ Старт")
async def btn_start(msg: Message):
    if STATE.is_trading:
        await msg.answer("⚠️ Торговля уже запущена", reply_markup=kb())
        return
    STATE.is_trading = True
    STATE.current_regime = Regime.HOLD
    STATE.current_side = Side.NONE
    if STATE.loop_task and not STATE.loop_task.done():
        STATE.loop_task.cancel()
    STATE.loop_task = asyncio.create_task(trading_loop())
    await msg.answer("✅ Торговля запущена", reply_markup=kb())

@DP.message(F.text == "⏹️ Стоп")
async def btn_stop(msg: Message):
    STATE.is_trading = False
    try:
        bybit_cancel_all(STATE.symbol)
    except Exception:
        pass
    await market_close_all(STATE.symbol)
    await msg.answer("⏹️ Торговля остановлена и позиции закрыты", reply_markup=kb())

@DP.message(F.text == "💰 Баланс")
async def btn_balance(msg: Message):
    bal = bybit_wallet_usdt()
    await msg.answer(f"💰 Баланс: <b>{bal:.2f} USDT</b>", parse_mode=ParseMode.HTML, reply_markup=kb())

@DP.message(F.text == "🚫 Закрыть всё")
async def btn_close_all(msg: Message):
    try:
        bybit_cancel_all(STATE.symbol)
    except Exception:
        pass
    await market_close_all(STATE.symbol)
    await msg.answer("🚫 Все позиции закрыты", reply_markup=kb())

# ------------------ Entry ------------------
async def main():
    log.info("Starting Bybit AI Trading Bot…")
    if not BOT:
        log.error("No TELEGRAM_BOT_TOKEN provided")
        return
    # send banner
    await tg_send("🤖 Бот запущен. Нажмите Старт для начала торговли.\n" +
                  f"Символ: <b>{STATE.symbol}</b>, Леверидж: <b>{LEVERAGE}x</b>\n" +
                  ("ТЕСТНЕТ" if os.getenv("BYBIT_TESTNET","true").lower()=="true" else "РЕАЛ"))
    await DP.start_polling(BOT)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        log.info("Stopped.")
# main.py
import os
import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Dict, Any, Tuple, List
from prompt import prompt as pr
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
# Локальная решалка
from local_ai import decide as local_decide

# ------------------ .env & logging ------------------
load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("tradebot.log", encoding="utf-8")]
)
log = logging.getLogger("MAIN")

# ------------------ Switch: local vs AI ------------------
USE_LOCAL_DECIDER = os.getenv("USE_LOCAL_DECIDER", "true").lower() == "true"  # default: local

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
        self.last_decision_sl_ticks: Optional[int] = None  # последний sl_ticks от локалки/ИИ
        self.trail_anchor: Optional[float] = None  # максимум/минимум с момента входа
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

        # Новые поля:
        self.last_ai_prev_ts: Optional[int] = None   # последняя обработанная ЗАКРЫТАЯ 1m свеча (ts)
        self.last_pos_size: float = 0.0              # размер позиции на предыдущем тике (детект открытия)
        self.last_sl_price: Optional[float] = None   # последний установленный SL (цена)
        self.reenter_block_until: float = 0.0        # запрет входа до этого времени (monotonic сек)

STATE = BotState()

# ------------------ Clients ------------------
def make_bybit() -> HTTP:
    testnet = os.getenv("BYBIT_TESTNET", "true").lower() == "true"
    api_key = os.getenv("BYBIT_API_KEY_TEST" if testnet else "BYBIT_API_KEY_MAIN")
    api_secret = os.getenv("BYBIT_SECRET_KEY_TEST" if testnet else "BYBIT_SECRET_KEY_MAIN")
    # pybit HTTP takes testnet=bool and uses requests under the hood.
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
SL_TICKS = int(os.getenv("SL_TICKS", "6000"))                 # настраивается в .env

# Take-profit rule for flat:
# "на 2 тика ниже/выше противоположного от уровня открытия позиции края тела прошлой минуты"
TP_BODY_OFFSET_TICKS = int(os.getenv("TP_BODY_OFFSET_TICKS", "2"))  # TP offset from body edge (ticks)
MARKET_BAND_EXTRA_TICKS = int(os.getenv("MARKET_BAND_EXTRA_TICKS", "4"))  # how many ticks deeper to push IOC Limit fallback

# Interval and bootstrap hours
POLL_TICK_MS = int(os.getenv("POLL_TICK_MS", "1000"))  # per-tick loop delay (ms)
BOOTSTRAP_HOURS = int(os.getenv("BOOTSTRAP_HOURS", "5"))
AI_POLL_SEC = int(os.getenv("AI_POLL_SEC", "60"))      # запасной лимит (основной триггер — закрытие свечи)

# Retry settings
RETRY_ATTEMPTS = 10
RETRY_DELAY_SEC = 1

TREND_SL_MULT = float(os.getenv("TREND_SL_MULT", "5.0"))

MIN_TP_TICKS = int(os.getenv("MIN_TP_TICKS", "2500"))  # минимум для TP в тиках

TREND_CONFIRM_BARS = int(os.getenv("TREND_CONFIRM_BARS", "3"))   # сколько подряд минутных сигналов TREND одной стороны нужно
REVERSE_HYSTERESIS_SEC = int(os.getenv("REVERSE_HYSTERESIS_SEC", "10"))  # пауза после flip (чтоб не дёргаться)

REENTER_AFTER_SL_SEC = int(os.getenv("REENTER_AFTER_SL_SEC", "10"))

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
def bybit_position_sltp(symbol: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Возвращает (sl_price, tp_price, sl_active_price) из позиции, если есть.
    Поля в v5 могут называться по-разному у разных категорий/аккаунтов, поэтому проверяем несколько ключей.
    """
    rr = BYBIT.get_positions(category=STATE.category, symbol=symbol)
    if not rr.get("result", {}).get("list"):
        return None, None, None
    p = rr["result"]["list"][0]

    def _to_float(x):
        try:
            return float(x) if x not in (None, "", "0", 0) else None
        except Exception:
            return None

    sl = _to_float(p.get("stopLoss")) or _to_float(p.get("sl"))
    if sl is None:
        sl = _to_float(p.get("stopLossPrice")) or _to_float(p.get("slPrice"))

    tp = _to_float(p.get("takeProfit")) or _to_float(p.get("tp"))
    if tp is None:
        tp = _to_float(p.get("takeProfitPrice")) or _to_float(p.get("tpPrice"))

    sl_active = _to_float(p.get("slActivePrice")) or _to_float(p.get("trailingStop"))

    return sl, tp, sl_active

def clamp_sl_for_exchange(side: Side, avg: float, last: float, sl_price: float, tick: float) -> float:
    """
    SL должен быть на правильной стороне от базовой цены.
    BUY: SL строго НИЖЕ базовой (min(avg,last)) минимум на 1 тик.
    SELL: SL строго ВЫШЕ базовой (max(avg,last)) минимум на 1 тик.
    """
    base_buy  = min(avg, last)
    base_sell = max(avg, last)
    if side == Side.BUY:
        max_sl = base_buy - tick
        return min(sl_price, max_sl)
    elif side == Side.SELL:
        min_sl = base_sell + tick
        return max(sl_price, min_sl)
    return sl_price

def clamp_tp_min_distance(side: Side, avg: float, last: float, tp_price: float, tick: float, min_ticks: int) -> float:
    """
    Минимальная дистанция для TP от базовой цены (страховка под правила биржи).
    BUY: TP ≥ base + min_ticks * tick
    SELL: TP ≤ base - min_ticks * tick
    """
    base = max(avg, last) if side == Side.BUY else min(avg, last)
    if side == Side.BUY:
        min_tp = base + max(min_ticks, 1) * tick
        return max(tp_price, min_tp)
    elif side == Side.SELL:
        max_tp = base - max(min_ticks, 1) * tick
        return min(tp_price, max_tp)
    return tp_price

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
    # Bybit max limit per call is ~200; 5h=300 bars → do multiple calls
    need = minutes
    out: List[List[Any]] = []
    while need > 0:
        lim = 200 if need > 200 else need
        rr = BYBIT.get_kline(category=STATE.category, symbol=symbol, interval="1", limit=lim)
        chunk = rr["result"]["list"]
        out = chunk + out  # API returns most-recent-first; accumulate and sort later
        need -= lim
        if need > 0:
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

# ------------------ New Bybit helpers ------------------
def bybit_cancel_order(symbol: str, order_id: str):
    # pybit v5 unified has cancel_order; route with category/symbol/orderId
    return BYBIT.cancel_order(category=STATE.category, symbol=symbol, orderId=order_id)

def bybit_cancel_entry_orders(symbol: str):
    """
    Отменяет ТОЛЬКО входные заявки (лимит/условные), которые НЕ reduceOnly.
    НИКОГДА не трогаем стоп/тейк позиции (обычно они либо не в open_orders,
    либо помечены reduceOnly / stopOrderType, мы их фильтруем).
    """
    rr = BYBIT.get_open_orders(category=STATE.category, symbol=symbol)
    lst = rr.get("result", {}).get("list", []) or []
    cancelled = 0
    for o in lst:
        try:
            # Признаки, что это позиционный выход (не трогаем):
            if o.get("reduceOnly") is True:
                continue
            # Некоторые стоп/условки подсвечиваются этими полями:
            if o.get("stopOrderType"):          # e.g. "StopLoss","TakeProfit","TrailingStop"
                continue
            if o.get("tpSlMode"):               # у некоторых аккаунтов встречается
                continue

            oid = o.get("orderId")
            if not oid:
                continue

            BYBIT.cancel_order(category=STATE.category, symbol=symbol, orderId=oid)
            cancelled += 1
        except Exception as e:
            log.warning("[CANCEL-ENTRY] fail cancel %s: %s", o.get("orderId"), e)
    log.info("[CANCEL-ENTRY] cancelled %d entry orders", cancelled)

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

# ------------------ AI / LOCAL DECIDER ------------------
def ai_prompt(symbol: str, df: pd.DataFrame, md: MarketData, txt: str) -> str:
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
        f"{txt}"
    )

def _extract_json_block(text: str) -> str:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return m.group(0)
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

def parse_local_decision(obj) -> AIDecision:
    """
    Принимает как dict, так и объект (например, dataclass Decision из local_ai).
    Поля: regime, side, sl_ticks, comment. Поддерживает как Enum, так и строки.
    """
    def get_field(name, default=None):
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    r = get_field("regime", "hold")
    regime = r if isinstance(r, Regime) else Regime(str(r))

    s = get_field("side", "None")
    side = s if isinstance(s, Side) else Side(str(s))

    sl_ticks = get_field("sl_ticks", None)
    if sl_ticks is not None:
        try:
            sl_ticks = int(sl_ticks)
        except Exception:
            sl_ticks = None

    comment = get_field("comment", "")

    return AIDecision(regime=regime, side=side, sl_ticks=sl_ticks, comment=comment)

async def ask_ai(symbol: str, df: pd.DataFrame, md: MarketData) -> AIDecision:
    prompt = ai_prompt(symbol, df, md, txt=pr)
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
        return AIDecision(regime=Regime.HOLD, side=Side.NONE, comment="fallback")

async def get_decision(symbol: str, df: pd.DataFrame, md: MarketData) -> AIDecision:
    """
    Унифицированная точка получения решения — локально или через OpenAI.
    Вызовем ТОЛЬКО ПОСЛЕ закрытия минутной свечи (см. trading_loop).
    """
    if USE_LOCAL_DECIDER:
        try:
            obj = await asyncio.to_thread(local_decide, symbol, df, md)
            dec = parse_local_decision(obj)
            log.info("[LOCAL] %s", dec)
            return dec
        except Exception as e:
            log.error("[LOCAL] error: %s", e)
            await tg_send(f"🧠 Локальная решалка дала ошибку, переключаюсь в HOLD. {e}")
            return AIDecision(regime=Regime.HOLD, side=Side.NONE, comment="local error")
    else:
        return await ask_ai(symbol, df, md)

# ------------------ Trading primitives with retries ------------------
RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", str(RETRY_ATTEMPTS)))
RETRY_DELAY_SEC = int(os.getenv("RETRY_DELAY_SEC", str(RETRY_DELAY_SEC)))

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
    Ставит SL/TP через /position/trading-stop в режиме Full.
    Обязательно делает верификацию: читает позицию и сверяет, что SL/TP появились.
    Обрабатывает 'not modified' как успех.
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
            got_sl, got_tp, got_sl_act = bybit_position_sltp(symbol)
            log.info("[SLTP] verify on exchange: SL=%s TP=%s (slActive=%s)", got_sl, got_tp, got_sl_act)
            return
        except Exception as e:
            msg = str(e)
            if "34040" in msg or "not modified" in msg.lower():
                log.info("[SLTP] not modified → already set; ok")
                got_sl, got_tp, got_sl_act = bybit_position_sltp(symbol)
                log.info("[SLTP] verify on exchange (not modified): SL=%s TP=%s (slActive=%s)", got_sl, got_tp, got_sl_act)
                return
            tries += 1
            log.warning("[SLTP] set_trading_stop retry %d: %s", tries, e)
            await asyncio.sleep(0.5)

    log.error("[SLTP] failed to set_trading_stop after retries; keeping position. sl=%s tp=%s", sl_price, tp_price)
# --- Жёсткий энфорсер SL: 10 секунд или закрываем позицию
async def enforce_sl_must_have(symbol: str, side: Side, f: MdFilters, *, sl_ticks: int, timeout_sec: int = 10) -> None:
    """
    Требует наличие SL в течение timeout_sec. Пытается выставить SL каждую секунду.
    Если за timeout_sec не удалось — закрывает позицию по рынку.
    Стоп считается от avg_price позиции (единый подход для всех режимов).
    """
    deadline = asyncio.get_event_loop().time() + timeout_sec
    last_err: Optional[Exception] = None
    while asyncio.get_event_loop().time() < deadline:
        fresh = read_market(STATE.symbol, 1)
        if fresh.position.size <= 0:
            return  # позиции уже нет

        # Если уже есть SL на бирже — выходим с успехом
        got_sl, got_tp, _ = bybit_position_sltp(symbol)
        if got_sl is not None:
            STATE.last_sl_price = got_sl
            log.info("[ENFORCE-SL] SL уже установлен на бирже: %s → ок", got_sl)
            return

        # Пытаемся выставить SL от средней цены позиции
        if side == Side.BUY:
            sl_price_raw = fresh.position.avg_price - sl_ticks * f.tick_size
        else:
            sl_price_raw = fresh.position.avg_price + sl_ticks * f.tick_size

        # 👇 КЛАМП К ПРАВИЛАМ БИРЖИ + НОРМАЛИЗАЦИЯ
        sl_price_clamped = clamp_sl_for_exchange(
            side,
            fresh.position.avg_price,
            fresh.last_price,
            sl_price_raw,
            f.tick_size,
        )
        sl_price = normalize_price(sl_price_clamped, f.tick_size)

        try:
            bybit_trading_stop(symbol, take_profit=None, stop_loss=sl_price)
            STATE.last_sl_price = sl_price
            log.info("[ENFORCE-SL] SL установлен/подтверждён: %s", sl_price)
            got_sl2, _, _ = bybit_position_sltp(symbol)
            log.info("[ENFORCE-SL] verify: SL на бирже = %s", got_sl2)
            return
        except Exception as e:
            last_err = e
            log.warning("[ENFORCE-SL] ошибка установки SL, пробую снова: %s", e)
        await asyncio.sleep(1)

    # Если сюда дошли — SL не удалось поставить в срок
    log.error("[ENFORCE-SL] не удалось установить SL за %ds → закрываю позицию! Последняя ошибка: %s", timeout_sec, last_err)
    await tg_send("⛔️ Не удалось установить SL за 10с — закрываю позицию!")
    await market_close_all(symbol)
    STATE.reenter_block_until = asyncio.get_event_loop().time() + 3  # небольшая пауза перед ре-входом
    log.info("[COOLDOWN] блок входа до %.1f (now=%.1f) после ENFORCE-SL",
             STATE.reenter_block_until, asyncio.get_event_loop().time())

# --- Добивание после срабатывания SL: если остаток висит — закрываем

async def hold_guard_sl(symbol: str, md: MarketData):
    """
    В режиме HOLD ничего не торгуем, но если позиция есть — обязаны держать SL.
    Если SL на бирже отсутствует, выставляем его по тем же правилам, что и в тренде/флэте:
    от средней цены, на base_ticks (или SL_TICKS) * (TREND_SL_MULT если текущий режим был TREND).
    Никаких закрытий позиции здесь — только попытка восстановить стоп.
    """
    if md.position.size <= 0:
        return

    got_sl, _, _ = bybit_position_sltp(symbol)
    if got_sl is not None:
        return  # всё ок

    # Берём последний sl_ticks из решения, если он был, иначе из .env
    try:
        base_ticks = STATE.last_decision_sl_ticks if STATE.last_decision_sl_ticks is not None else SL_TICKS
        base_ticks = int(base_ticks)
    except Exception:
        base_ticks = int(SL_TICKS)

    # Если до HOLD мы были в тренде — используем множитель, чтобы логика стопа не «схлопнулась»
    mult = TREND_SL_MULT if STATE.prev_regime_was_trend else 1.0
    sl_ticks = int(max(1, round(base_ticks * mult)))

    f = md.filters
    avg = md.position.avg_price
    last = md.last_price

    if md.position.side == Side.BUY:
        sl_raw = avg - sl_ticks * f.tick_size
    else:
        sl_raw = avg + sl_ticks * f.tick_size

    # клампим к правилам биржи и нормализуем
    sl_px = clamp_sl_for_exchange(md.position.side, avg, last, sl_raw, f.tick_size)
    sl_px = normalize_price(sl_px, f.tick_size)

    try:
        bybit_trading_stop(symbol, take_profit=None, stop_loss=sl_px)
        STATE.last_sl_price = sl_px
        log.info("[HOLD-GUARD] восстановил SL: %s", sl_px)
    except Exception as e:
        log.warning("[HOLD-GUARD] не удалось восстановить SL: %s", e)


async def sweep_after_sl(symbol: str, md: MarketData, f: MdFilters):
    """
    Если SL уже привязан к позиции на бирже — биржа сама закроет. Не вмешиваемся.
    Если SL нет, но цена прошла дальше на 2 тика — добиваем остаток рыночным.
    """
    got_sl, _, _ = bybit_position_sltp(symbol)
    if got_sl is not None:
        return

    if md.position.size <= 0 or STATE.last_sl_price is None:
        return

    lp = md.last_price
    if md.position.side == Side.BUY and lp <= STATE.last_sl_price - 2 * f.tick_size:
        await market_close_all(symbol)
        STATE.reenter_block_until = asyncio.get_event_loop().time() + 3
        log.info("[COOLDOWN] блок входа до %.1f (now=%.1f) после SWEEP (BUY)",
                 STATE.reenter_block_until, asyncio.get_event_loop().time())
        log.warning("[SWEEP] SL прошит вниз, остаток позиции закрыт рыночным")
    if md.position.side == Side.SELL and lp >= STATE.last_sl_price + 2 * f.tick_size:
        await market_close_all(symbol)
        STATE.reenter_block_until = asyncio.get_event_loop().time() + 3
        log.info("[COOLDOWN] блок входа до %.1f (now=%.1f) после SWEEP (SELL)",
                 STATE.reenter_block_until, asyncio.get_event_loop().time())
        log.warning("[SWEEP] SL прошит вверх, остаток позиции закрыт рыночным")

async def ensure_order_filled_or_cancel(symbol: str, order_id: str, timeout_sec: int = 45) -> bool:
    """
    Poll open orders; if still open after timeout, cancel all and return False.
    """
    deadline = asyncio.get_event_loop().time() + timeout_sec
    while asyncio.get_event_loop().time() < deadline:
        od = bybit_get_order(order_id, symbol)
        if od is None:
            pos = bybit_position(symbol)
            return pos.size > 0
        await asyncio.sleep(1)
    try:
        bybit_cancel_all(symbol)
    except Exception:
        pass
    return False

# ------------------ Regime executors ------------------
async def do_trend(md: MarketData, dec: AIDecision):
    """
    TREND:
    - Вход только по направлению dec.side.
      1) Сначала Market IOC.
      2) Если прайс-бэнд (30208/30209) ИЛИ Market не заполнился → пробуем несколько Limit IOC (fallback).
         Каждый раз ждём реального фила через ensure_order_filled_or_cancel.
    - Трейлинг SL:
      • стартовый якорь = avg_price позиции при первом входе,
      • дальше якорь — экстремум цены в пользу позиции,
      • SL = anchor ± (sl_ticks * TREND_SL_MULT) * tick,
      • ужесточаем только (никогда не отодвигаем дальше от цены).
    """
    symbol = STATE.symbol
    f = md.filters
    desired = dec.side
    if desired not in (Side.BUY, Side.SELL):
        return

    # пауза на ре-вход после закрытия/срабатывания SL
    if asyncio.get_event_loop().time() < STATE.reenter_block_until:
        return

    # Если есть позиция, но противоположная — закрываем и выходим (перевход в следующем тике)
    if md.position.size > 0 and md.position.side != desired:
        await market_close_all(symbol)
        STATE.reenter_block_until = asyncio.get_event_loop().time() + 1
        log.info("[COOLDOWN] блок входа до %.1f (now=%.1f) после переворота TREND",
                 STATE.reenter_block_until, asyncio.get_event_loop().time())
        return

    # ===== ВХОД =====
    if md.position.size == 0:
        # нет активных ордеров? — входим
        if not bybit_open_orders(symbol):
            md2 = read_market(symbol, 1)
            qty = normalize_qty(LOT_SIZE_USDT / max(md2.last_price, 1e-8), f.qty_step, f.min_qty)

            # 1) Market IOC
            try:
                r = await retry_place(lambda: bybit_place_order(
                    symbol=symbol, side=desired, order_type="Market", qty=qty, time_in_force="IOC"
                ), descr=f"trend_enter_{desired.value}")
                await tg_send(f"📈 TREND {desired.value} qty={qty} (Market IOC)")
                # ждём появления позиции (рыночный может не исполниться из-за бэндов на тестнете)
                if not await ensure_order_filled_or_cancel(symbol, r["result"]["orderId"], timeout_sec=10):
                    # Market не дал позицию → идём в fallback IOC-Limit
                    bid, ask = bybit_best_prices(symbol)
                    base_px = (ask if desired == Side.BUY else bid) or md2.last_price
                    base_px = normalize_price(base_px, f.tick_size)

                    placed = False
                    for i in range(1, 4):
                        px_try = (base_px + i * MARKET_BAND_EXTRA_TICKS * f.tick_size) if desired == Side.BUY \
                                 else (base_px - i * MARKET_BAND_EXTRA_TICKS * f.tick_size)
                        px_try = normalize_price(px_try, f.tick_size)
                        try:
                            r2 = await retry_place(lambda: bybit_place_order(
                                symbol=symbol, side=desired, order_type="Limit", qty=qty,
                                price=px_try, time_in_force="IOC"
                            ), descr=f"trend_enter_limit_{desired.value}")
                            if await ensure_order_filled_or_cancel(symbol, r2["result"]["orderId"], timeout_sec=10):
                                await tg_send(f"📈 TREND {desired.value} qty={qty} (IOC Limit @ {px_try})")
                                placed = True
                                break
                        except Exception as ee:
                            log.warning("[TREND] IOC-Limit попытка %d не удалась: %s", i, ee)
                            continue

                    if not placed:
                        log.error("[TREND] Market не заполнился и fallback IOC-Limit не сработал")
                return
            except Exception as e:
                msg = str(e).lower()
                # 2) Прайс-бэнд → сразу пробуем Limit IOC
                if ("maximum buying price" in msg) or ("maximum selling price" in msg) or ("30208" in msg) or ("30209" in msg):
                    bid, ask = bybit_best_prices(symbol)
                    base_px = (ask if desired == Side.BUY else bid) or md2.last_price
                    base_px = normalize_price(base_px, f.tick_size)

                    placed = False
                    for i in range(1, 4):
                        px_try = (base_px + i * MARKET_BAND_EXTRA_TICKS * f.tick_size) if desired == Side.BUY \
                                 else (base_px - i * MARKET_BAND_EXTRA_TICKS * f.tick_size)
                        px_try = normalize_price(px_try, f.tick_size)
                        try:
                            r2 = await retry_place(lambda: bybit_place_order(
                                symbol=symbol, side=desired, order_type="Limit", qty=qty,
                                price=px_try, time_in_force="IOC"
                            ), descr=f"trend_enter_limit_{desired.value}")
                            if await ensure_order_filled_or_cancel(symbol, r2["result"]["orderId"], timeout_sec=10):
                                await tg_send(f"📈 TREND {desired.value} qty={qty} (IOC Limit @ {px_try})")
                                placed = True
                                break
                        except Exception as ee:
                            log.warning("[TREND] IOC-Limit попытка %d не удалась: %s", i, ee)
                            continue

                    if not placed:
                        log.error("[TREND] fallback IOC-Limit не сработал — вход не выполнен")
                    return
                else:
                    # какая-то иная ошибка — пробрасываем выше
                    raise

        # если есть открытые заявки — ничего не делаем, ждём
        return

    # ===== ТРЕЙЛИНГ SL =====
    # число тиков SL (override от ИИ/локалки + множитель тренда) с защитой от кривых типов
    try:
        base_ticks = dec.sl_ticks if dec.sl_ticks is not None else SL_TICKS
        base_ticks = int(base_ticks)
    except Exception:
        base_ticks = int(SL_TICKS)
    sl_ticks = int(base_ticks * TREND_SL_MULT)

    fresh = read_market(symbol, 1)
    if fresh.position.size <= 0:
        return

    lp = fresh.last_price

    # стартовый якорь — средняя цена входа (чтобы SL не был «впритык» с открытия)
    if STATE.trail_anchor is None:
        STATE.trail_anchor = fresh.position.avg_price

    if fresh.position.side == Side.BUY:
        # обновляем максимум-якорь только вверх
        if lp > STATE.trail_anchor:
            STATE.trail_anchor = lp
        desired_sl = normalize_price(STATE.trail_anchor - sl_ticks * f.tick_size, f.tick_size)
        # ужесточаем только если новый SL ближе к цене, чем предыдущий
        if STATE.last_sl_price is None or desired_sl > STATE.last_sl_price:
            await ensure_sl_tp(symbol, sl_price=desired_sl, tp_price=None)
            STATE.last_sl_price = desired_sl
            log.info("[TREND] TRAIL BUY sl→ %.6f (anchor=%.6f, ticks=%d)", desired_sl, STATE.trail_anchor, sl_ticks)

    else:  # SELL
        # обновляем минимум-якорь только вниз
        if lp < STATE.trail_anchor:
            STATE.trail_anchor = lp
        desired_sl = normalize_price(STATE.trail_anchor + sl_ticks * f.tick_size, f.tick_size)
        if STATE.last_sl_price is None or desired_sl < STATE.last_sl_price:
            await ensure_sl_tp(symbol, sl_price=desired_sl, tp_price=None)
            STATE.last_sl_price = desired_sl
            log.info("[TREND] TRAIL SELL sl→ %.6f (anchor=%.6f, ticks=%d)", desired_sl, STATE.trail_anchor, sl_ticks)

async def do_flat(md: MarketData, dec: AIDecision):
    """
    FLAT:
    - Если позиции нет: на каждой новой закрытой минуте отменяем старую лимитку и ставим новую:
      BUY  → entry = prev_low  + 6*tick (внутрь диапазона)
      SELL → entry = prev_high - 6*tick (внутрь диапазона)
    - Если позиция есть: ставим SL от avg_price, TP от «края тела» предыдущей свечи
      (плюс клампы: TP минимум 2500 тиков от базы).
    - Если пришёл новый сигнал:
       * FLAT остаётся и направление совпадает с позицией → ничего не делаем.
       * Направление не совпадает → отменяем ордера, закрываем позицию и продолжаем.
      (Логика переключения — в основном цикле; тут только постановка заявок/SLTP.)
    """
    symbol = STATE.symbol
    f = md.filters
    desired = dec.side
    if desired not in (Side.BUY, Side.SELL):
        return

    if asyncio.get_event_loop().time() < STATE.reenter_block_until:
        return

    df = _kline_to_df(md.kline_1m)
    prev = df.iloc[-2]
    prev_ts = int(prev["ts"])
    prev_high = float(prev["high"])
    prev_low = float(prev["low"])
    body_low, body_high = _body_edges(prev)

    # Противоположная позиция → закрыть
    if md.position.size > 0 and md.position.side != desired:
        await market_close_all(symbol)
        try:
            bybit_cancel_all(symbol)
        except Exception:
            pass
        STATE.reenter_block_until = asyncio.get_event_loop().time() + 1
        log.info("[COOLDOWN] блок входа до %.1f (now=%.1f) после смены стороны FLAT",
                 STATE.reenter_block_until, asyncio.get_event_loop().time())
        return

    open_orders = bybit_open_orders(symbol)
    need_requote = (STATE.last_flat_prev_ts is None) or (prev_ts != STATE.last_flat_prev_ts)

    # Позиции нет → лимитки
    if md.position.size == 0:
        if need_requote and open_orders:
            try:
                bybit_cancel_all(symbol)
                await tg_send("♻️ FLAT ре-котировка: отменил старую заявку")
            except Exception:
                pass
            open_orders = []

        if not open_orders:
            if desired == Side.BUY:
                entry_price = normalize_price(prev_low + FLAT_ENTRY_TICKS * f.tick_size, f.tick_size)
            else:
                entry_price = normalize_price(prev_high - FLAT_ENTRY_TICKS * f.tick_size, f.tick_size)
            qty = normalize_qty(LOT_SIZE_USDT / max(entry_price, 1e-8), f.qty_step, f.min_qty)
            await retry_place(lambda: bybit_place_order(
                symbol=symbol, side=desired, order_type="Limit", qty=qty,
                price=entry_price, time_in_force="GTC"
            ), descr=f"flat_limit_enter_{desired.value}@{entry_price}")
            await tg_send(f"🤏 FLAT заявка {desired.value} {qty} @ {entry_price}")
        STATE.last_flat_prev_ts = prev_ts
        return

    # Позиция есть → SL/TP
    fresh = read_market(symbol, 1)
    base_ticks = dec.sl_ticks if dec.sl_ticks is not None else SL_TICKS

    if desired == Side.BUY:
        sl_price_raw = fresh.position.avg_price - base_ticks * f.tick_size
        tp_edge = body_high
        tp_price_raw = tp_edge - TP_BODY_OFFSET_TICKS * f.tick_size
    else:
        sl_price_raw = fresh.position.avg_price + base_ticks * f.tick_size
        tp_edge = body_low
        tp_price_raw = tp_edge + TP_BODY_OFFSET_TICKS * f.tick_size

    # нормализация
    sl_price = normalize_price(sl_price_raw, f.tick_size)
    tp_price = normalize_price(tp_price_raw, f.tick_size)

    # клампы
    avg = fresh.position.avg_price
    last = fresh.last_price
    sl_price = clamp_sl_for_exchange(desired, avg, last, sl_price, f.tick_size)
    tp_price = clamp_tp_min_distance(desired, avg, last, tp_price, f.tick_size, MIN_TP_TICKS)

    await ensure_sl_tp(symbol, sl_price=sl_price, tp_price=tp_price)
    STATE.last_sl_price = sl_price
    await tg_send(f"🛡 SL={sl_price} 🎯 TP={tp_price}")
    STATE.last_flat_prev_ts = prev_ts


async def do_hold(md: MarketData, dec: AIDecision):
    # Нет торговли; режим ожидания
    log.info("[HOLD] No trades")


# ======== Тонкая настройка распознавания разворота тренда ========


def _ensure_trend_state_fields():
    # Ленивая инициализация полей состояния
    if not hasattr(STATE, "_trend_queue"):
        from collections import deque
        STATE._trend_queue = deque(maxlen=TREND_CONFIRM_BARS)
    if not hasattr(STATE, "last_flip_at"):
        STATE.last_flip_at = 0.0

async def _flip_to_trend(symbol: str, side: Side, now_mono: float):
    """Жёсткое переключение на тренд side с закрытием встречной позиции и гистерезисом."""
    await tg_send("🔁 Переключение на ТРЕНД: закрываю встречные позиции, снимаю заявки")
    try:
        bybit_cancel_all(symbol)
    except Exception:
        pass
    await market_close_all(symbol)
    STATE.reenter_block_until = now_mono + REVERSE_HYSTERESIS_SEC
    STATE.current_regime = Regime.TREND
    STATE.current_side = side
    STATE.last_flip_at = now_mono
    await tg_send(
        "🤖 Обновление:\n"
        f"Режим: <b>{STATE.current_regime.value}</b>\nНаправление: <b>{STATE.current_side.value}</b>"
    )

async def apply_trend_confirmation(dec_new: AIDecision, md: MarketData, now_mono: float) -> bool:
    """
    Подтверждение разворота тренда:
    - Копим последние TREND_CONFIRM_BARS решений TREND по стороне (Buy/Sell);
    - Переключаемся только если ВСЕ подряд совпали и это отличается от текущей стороны;
    - После flip ставим гистерезис REVERSE_HYSTERESIS_SEC (STATE.reenter_block_until).
    Возвращает True, если произошло переключение режима/направления.
    """
    _ensure_trend_state_fields()

    # Не TREND — сбрасываем буфер и выходим
    if dec_new.regime != Regime.TREND or dec_new.side not in (Side.BUY, Side.SELL):
        STATE._trend_queue.clear()
        return False

    # Добавляем текущий сигнал
    STATE._trend_queue.append(dec_new.side)

    # Если буфер ещё не набран — рано
    if len(STATE._trend_queue) < TREND_CONFIRM_BARS:
        return False

    # Есть ли подтверждение одной и той же стороны?
    all_same = all(s == STATE._trend_queue[0] for s in STATE._trend_queue)
    confirmed_side = STATE._trend_queue[0] if all_same else None
    if not confirmed_side:
        return False

    # Уже в нужной стороне тренда — ничего не меняем
    if STATE.current_regime == Regime.TREND and STATE.current_side == confirmed_side:
        return False

    # Гистерезис: не дёргаемся, если ещё идёт блок пере-входа
    if now_mono < getattr(STATE, "reenter_block_until", 0.0):
        return False

    # Если есть позиция на встречной стороне — закрыть
    if md.position.size > 0 and md.position.side != confirmed_side:
        await _flip_to_trend(STATE.symbol, confirmed_side, now_mono)
        return True

    # Позиции нет или она совпадает по стороне, но режим другой → просто переключаем режим/сторону
    await _flip_to_trend(STATE.symbol, confirmed_side, now_mono)
    return True


# ------------------ Main trading loop ------------------
async def trading_loop():
    symbol = STATE.symbol
    STATE.prev_regime_was_trend = False
    await tg_send("🚀 Старт торговли")

    # leverage
    try:
        bybit_set_leverage(symbol, LEVERAGE)
        log.info("[LEV] set leverage=%dx", LEVERAGE)
    except Exception as e:
        log.warning("[LEV] fail set leverage: %s", e)

    # initial bootstrap (5 hours history) with fallback
    try:
        md = read_market(symbol, BOOTSTRAP_HOURS)
        df = _kline_to_df(md.kline_1m)
        await tg_send("📥 Подгрузка данных за 5 часов завершена")
    except Exception as e:
        log.exception("[BOOTSTRAP] failed 5h load: %s", e)
        await tg_send("❌ Ошибка загрузки истории за 5 часов. Пробую загрузить 60 минут…")
        md = read_market(symbol, 1)
        df = _kline_to_df(md.kline_1m)
        await tg_send("📥 Подгружено за 60 минут (fallback)")

    # Инициализация маркера последней закрытой 1m свечи
    if len(df) >= 2:
        STATE.last_ai_prev_ts = int(df.iloc[-2]["ts"])

    # Первое решение — по закрытой свече
    dec = await get_decision(symbol, df, md)
    STATE.current_regime = dec.regime
    STATE.current_side = dec.side
    STATE.last_decision_sl_ticks = dec.sl_ticks
    await tg_send(f"🧭 Режим: <b>{dec.regime.value}</b> | Направление: <b>{dec.side.value}</b>")

    last_ai_time = 0.0
    while STATE.is_trading:
        try:
            now_mono = asyncio.get_event_loop().time()

            # Читаем рынок
            history_hours = 4 if USE_LOCAL_DECIDER else 1
            md = read_market(symbol, history_hours)
            df = _kline_to_df(md.kline_1m)

            # ---------- РАННИЙ КУЛДАУН ПОСЛЕ ВЫХОДА В НОЛЬ ----------
            # Если на прошлом шаге позиция была >0, а сейчас 0 — считаем, что был полный выход (в т.ч. SL).
            if STATE.last_pos_size > 0 and md.position.size == 0:
                STATE.trail_anchor = None
                STATE.last_sl_hit_at = datetime.now(timezone.utc)
                STATE.reenter_block_until = now_mono + REENTER_AFTER_SL_SEC
                log.info("[COOLDOWN] (EARLY) блок входа до %.1f (now=%.1f) после выхода в 0",
                         STATE.reenter_block_until, now_mono)
                # перестраховка: снимаем только входные лимитки (SL/TP уже неактуальны без позиции)
                try:
                    bybit_cancel_entry_orders(symbol)
                    log.info("[CANCEL-ENTRY] cancelled entry orders after exit")
                except Exception:
                    pass
                # обновляем маркер и уходим в следующий тик, чтобы в этом не было входов
                STATE.last_pos_size = md.position.size
                await asyncio.sleep(POLL_TICK_MS / 1000.0)
                continue
            # --------------------------------------------------------

            # --- Получаем решение ТОЛЬКО после закрытия новой минутки (+ резервная страховка) ---
            if len(df) >= 2:
                prev_ts_now = int(df.iloc[-2]["ts"])
            else:
                prev_ts_now = None

            need_decision = False
            if prev_ts_now is not None and prev_ts_now != STATE.last_ai_prev_ts:
                need_decision = True
                STATE.last_ai_prev_ts = prev_ts_now
            elif (now_mono - last_ai_time) >= AI_POLL_SEC * 3:
                # страховка: если рынок «залип» и свеча не закрывается долго
                need_decision = True

            if need_decision:
                dec_new = await get_decision(symbol, df, md)
                STATE.last_decision_sl_ticks = dec_new.sl_ticks
                regime_changed = dec_new.regime != STATE.current_regime
                side_changed = (dec_new.side != STATE.current_side) and dec_new.regime != Regime.HOLD

                # Подтверждение разворота тренда
                if dec_new.regime == Regime.TREND:
                    flipped = await apply_trend_confirmation(dec_new, md, now_mono)
                    if flipped:
                        last_ai_time = now_mono
                        # после flip сразу в следующий тик (пускай гистерезис/кулдаун отработают)
                        await asyncio.sleep(POLL_TICK_MS / 1000.0)
                        continue

                # Спец-логика для FLAT
                if STATE.current_regime == Regime.FLAT and dec_new.regime == Regime.FLAT:
                    pos_now = md.position
                    if pos_now.size > 0:
                        if dec_new.side == pos_now.side:
                            STATE.current_side = dec_new.side
                            STATE.current_regime = Regime.FLAT
                            last_ai_time = now_mono
                        else:
                            await tg_send("🔁 FLAT: смена направления при открытой позиции → закрываю по рынку")
                            # РАНЕЕ ставим короткий кулдаун, чтобы в этот же тик не войти обратно
                            STATE.reenter_block_until = now_mono + 3
                            try:
                                bybit_cancel_all(symbol)
                            except Exception:
                                pass
                            await market_close_all(symbol)
                            log.info("[COOLDOWN] блок входа до %.1f (now=%.1f) после смены стороны FLAT",
                                     STATE.reenter_block_until, now_mono)
                            STATE.current_regime = Regime.FLAT
                            STATE.current_side = dec_new.side
                            last_ai_time = now_mono
                            await tg_send(
                                "🤖 Обновление:\n"
                                f"Режим: <b>{dec_new.regime.value}</b>\nНаправление: <b>{dec_new.side.value}</b>"
                                + (f"\nКомментарий: {dec_new.comment}" if dec_new.comment else "")
                            )
                    else:
                        if side_changed:
                            try:
                                bybit_cancel_all(symbol)
                            except Exception:
                                pass
                            STATE.current_side = dec_new.side
                            STATE.current_regime = Regime.FLAT
                            last_ai_time = now_mono
                            await tg_send(
                                "🤖 Обновление:\n"
                                f"Режим: <b>{dec_new.regime.value}</b>\nНаправление: <b>{dec_new.side.value}</b>"
                                + (f"\nКомментарий: {dec_new.comment}" if dec_new.comment else "")
                            )
                else:
                    if regime_changed or side_changed:
                        if dec_new.regime == Regime.HOLD:
                            # Переход в HOLD: позиции НЕ закрываем, снимаем только входные лимитки
                            await tg_send("⏸ Переход в HOLD → позицию оставляю, снимаю только входные лимит-заявки")
                            try:
                                bybit_cancel_entry_orders(symbol)
                            except Exception:
                                pass
                            STATE.prev_regime_was_trend = (STATE.current_regime == Regime.TREND)
                            STATE.current_regime = Regime.HOLD
                            await tg_send(
                                "🤖 Обновление:\n"
                                f"Режим: <b>{dec_new.regime.value}</b>\nНаправление: <b>{STATE.current_side.value}</b>"
                                + (f"\nКомментарий: {dec_new.comment}" if dec_new.comment else "")
                            )
                        elif dec_new.regime == Regime.FLAT:
                            # Переход в FLAT: ВСЕГДА закрываем позиции и удаляем ВСЕ заявки
                            await tg_send("↔️ Переход в FLAT → снимаю все заявки и закрываю позиции")
                            # ставим короткий кулдаун заранее, чтобы вход не произошёл в этот же тик
                            STATE.reenter_block_until = now_mono + 3
                            try:
                                bybit_cancel_all(symbol)
                            except Exception:
                                pass
                            await market_close_all(symbol)
                            log.info("[COOLDOWN] блок входа до %.1f (now=%.1f) при переходе во FLAT",
                                     STATE.reenter_block_until, now_mono)
                            STATE.current_regime = Regime.FLAT
                            STATE.current_side = dec_new.side
                            STATE.last_flat_prev_ts = None  # принудим свежую перекотировку лимиток на следующей минуте
                            await tg_send(
                                "🤖 Обновление:\n"
                                f"Режим: <b>{dec_new.regime.value}</b>\nНаправление: <b>{dec_new.side.value}</b>"
                                + (f"\nКомментарий: {dec_new.comment}" if dec_new.comment else "")
                            )
                        else:
                            # Переход в TREND или смена стороны вне HOLD/FLAT-спецкейса
                            await tg_send("🔁 Переключение режима/направления → Закрываю позиции")
                            # РАНЕЕ ставим кулдаун перед закрытием
                            STATE.reenter_block_until = now_mono + 3
                            try:
                                bybit_cancel_all(symbol)
                            except Exception:
                                pass
                            await market_close_all(symbol)
                            log.info("[COOLDOWN] блок входа до %.1f (now=%.1f) после переключения режима",
                                     STATE.reenter_block_until, now_mono)
                            STATE.current_regime = dec_new.regime
                            STATE.current_side = dec_new.side
                            await tg_send(
                                "🤖 Обновление:\n"
                                f"Режим: <b>{dec_new.regime.value}</b>\nНаправление: <b>{dec_new.side.value}</b>"
                                + (f"\nКомментарий: {dec_new.comment}" if dec_new.comment else "")
                            )
                    last_ai_time = now_mono

            # Если идёт кулдаун — вообще не трогаем исполнение режимов
            if now_mono < getattr(STATE, "reenter_block_until", 0.0):
                await asyncio.sleep(POLL_TICK_MS / 1000.0)
                # обновим маркер размера и продолжим
                STATE.last_pos_size = md.position.size
                continue

            # --- Исполнение по текущему режиму ---
            if STATE.current_regime == Regime.TREND:
                await do_trend(md, AIDecision(STATE.current_regime, STATE.current_side))
            elif STATE.current_regime == Regime.FLAT:
                await do_flat(md, AIDecision(STATE.current_regime, STATE.current_side))
            else:
                await do_hold(md, AIDecision(STATE.current_regime, STATE.current_side))
                await hold_guard_sl(symbol, md)  # в HOLD следим, что SL живой

            # --- Если позиция только что появилась (0 → >0), 10с на установку SL ---
            if STATE.last_pos_size <= 0 and md.position.size > 0:
                try:
                    base_ticks = STATE.last_decision_sl_ticks if STATE.last_decision_sl_ticks is not None else SL_TICKS
                    try:
                        base_ticks = int(base_ticks)
                    except Exception:
                        base_ticks = int(SL_TICKS)
                    sl_ticks_enforce = int(base_ticks * (TREND_SL_MULT if STATE.current_regime == Regime.TREND else 1))
                    STATE.trail_anchor = md.position.avg_price
                    await enforce_sl_must_have(
                        symbol,
                        md.position.side,
                        md.filters,
                        sl_ticks=sl_ticks_enforce,
                        timeout_sec=10,
                    )
                except Exception as e:
                    log.exception("[ENFORCE-SL] непредвиденная ошибка: %s", e)

            # --- Добивание после SL, если остаток висит ---
            await sweep_after_sl(symbol, md, md.filters)

            # Обновляем маркер размера позиции (для ранней ветки выше)
            STATE.last_pos_size = md.position.size

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
    await msg.answer(f"✅ Торговля запущена. Источник сигналов: {'ЛОКАЛЬНЫЙ' if USE_LOCAL_DECIDER else 'ИИ'}", reply_markup=kb())

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
    await tg_send("🤖 Бот запущен. Нажмите Старт для начала торговли.\n" +
                  f"Символ: <b>{STATE.symbol}</b>, Леверидж: <b>{LEVERAGE}x</b>\n" +
                  ("ТЕСТНЕТ" if os.getenv("BYBIT_TESTNET","true").lower()=="true" else "РЕАЛ") +
                  f"\nИсточник сигналов: <b>{'ЛОКАЛЬНЫЙ' if USE_LOCAL_DECIDER else 'ИИ'}</b>")
    await DP.start_polling(BOT)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        log.info("Stopped.")
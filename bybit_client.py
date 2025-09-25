import os
import asyncio

from typing import Optional, Dict, Any, Tuple, List

from clients import BYBIT, RETRY_ATTEMPTS, RETRY_DELAY_SEC, \
    SL_TICKS, TREND_SL_MULT
import pandas as pd
import time

from helpers import normalize_price, tg_send
from models import STATE, Side, MdFilters, PositionInfo, MarketData
from logger import log
# ---- Optional SSL relax (env BYBIT_VERIFY_SSL=false) ----
if os.getenv("BYBIT_VERIFY_SSL", "true").lower() == "false":
    os.environ["PYTHONHTTPSVERIFY"] = "0"

from dotenv import load_dotenv

load_dotenv()




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




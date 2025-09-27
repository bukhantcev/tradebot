import os
import asyncio

from bybit_client import market_close_all, bybit_open_orders, read_market, retry_place, bybit_place_order, \
    ensure_order_filled_or_cancel, bybit_best_prices, ensure_sl_tp, _kline_to_df, _body_edges, bybit_cancel_all, \
    clamp_sl_for_exchange, clamp_tp_min_distance
from clients import \
    SL_TICKS, TREND_SL_MULT, LOT_SIZE_USDT, MARKET_BAND_EXTRA_TICKS, FLAT_ENTRY_TICKS, TP_BODY_OFFSET_TICKS, \
    MIN_TP_TICKS, TREND_CONFIRM_BARS, REVERSE_HYSTERESIS_SEC, FLAT_CHANNEL_BARS, SL_FLAT_CHANNEL_PCT
from helpers import normalize_qty, tg_send, normalize_price
from models import MarketData, AIDecision, STATE, Side, Regime
from logger import log
# ---- Optional SSL relax (env BYBIT_VERIFY_SSL=false) ----
if os.getenv("BYBIT_VERIFY_SSL", "true").lower() == "false":
    os.environ["PYTHONHTTPSVERIFY"] = "0"

from dotenv import load_dotenv

load_dotenv()


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
    - Если позиции нет: на каждой новой закрытой минуте отменяем старую лимитку и ставим новую
      на границу канала, но чуть внутри (± FLAT_ENTRY_TICKS * tick от границы).
    - Если позиция есть: TP — ровно на противоположной границе канала, SL — на X% (SL_FLAT_CHANNEL_PCT)
      за внешней границей канала, плюс биржевые клампы/нормализация.
    - Смена стороны при открытой позиции → закрытие и короткий кулдаун.
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

    # === Канал флэта по последним закрытым барам ===
    win = max(2, int(FLAT_CHANNEL_BARS))
    # Берём только ЗАКРЫТЫЕ бары: [-win-1 : -1] (исключая текущий формирующийся)
    window_df = df.iloc[-(win + 1):-1] if len(df) >= win + 1 else df.iloc[:-1]
    chan_high = float(window_df["high"].max()) if not window_df.empty else prev_high
    chan_low = float(window_df["low"].min()) if not window_df.empty else prev_low

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

    # Позиции нет → лимитки на границу канала, но немного внутри
    if md.position.size == 0:
        if need_requote and open_orders:
            try:
                bybit_cancel_all(symbol)
                await tg_send("♻️ FLAT ре-котировка: отменил старую заявку")
            except Exception:
                pass
            open_orders = []

        if not open_orders:
            ts = f.tick_size if f.tick_size > 0 else 0.1
            if desired == Side.BUY:
                # вход от нижней границы внутрь канала
                entry_price = normalize_price(chan_low + FLAT_ENTRY_TICKS * ts, ts)
            else:
                # вход от верхней границы внутрь канала
                entry_price = normalize_price(chan_high - FLAT_ENTRY_TICKS * ts, ts)
            qty = normalize_qty(LOT_SIZE_USDT / max(entry_price, 1e-8), f.qty_step, f.min_qty)
            await retry_place(lambda: bybit_place_order(
                symbol=symbol, side=desired, order_type="Limit", qty=qty,
                price=entry_price, time_in_force="GTC"
            ), descr=f"flat_limit_enter_{desired.value}@{entry_price}")
            await tg_send(f"🤏 FLAT заявка {desired.value} {qty} @ {entry_price}")
        STATE.last_flat_prev_ts = prev_ts
        return

    # Позиция есть → SL/TP (TP на противоположной границе канала, SL за каналом на %)
    fresh = read_market(symbol, 1)
    ts = f.tick_size if f.tick_size > 0 else 0.1
    pct = max(0.0, float(SL_FLAT_CHANNEL_PCT)) / 100.0

    avg = fresh.position.avg_price
    last = fresh.last_price

    if desired == Side.BUY:
        # TP — ровно на верхней границе канала, SL — на % ниже нижней границы (вне канала)
        tp_raw = chan_high
        sl_raw = chan_low * (1.0 - pct)
        # Биржевое правило: SL для BUY ниже базы минимум на 1 тик
        sl_raw = min(sl_raw, min(avg, last) - ts)
    else:  # SELL
        # TP — ровно на нижней границе канала, SL — на % выше верхней границы (вне канала)
        tp_raw = chan_low
        sl_raw = chan_high * (1.0 + pct)
        # Биржевое правило: SL для SELL выше базы минимум на 1 тик
        sl_raw = max(sl_raw, max(avg, last) + ts)

    # Нормализация/клампы
    sl_price = normalize_price(sl_raw, ts)
    tp_price = normalize_price(tp_raw, ts)

    sl_price = clamp_sl_for_exchange(desired, avg, last, sl_price, ts)

    # Страховка: TP должен остаться внутри канала
    tp_price = max(min(tp_price, chan_high), chan_low)

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
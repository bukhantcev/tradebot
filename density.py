# density.py — стаканная “плотность”: вход в сторону перевеса объёма
from typing import Dict, Any

def _top_sum(levels, n=5):
    s = 0.0
    for i, lv in enumerate(levels[:n]):
        # Bybit REST: ["price","size"] как строки
        try:
            s += float(lv[1])
        except Exception:
            pass
    return s

def run_once(Filters, ctx: Dict[str, Any]):
    f = ctx["features"]; p = ctx["params"]
    ob = f["orderbook"]
    bids = ob.get("bids") or ob.get("b", [])
    asks = ob.get("asks") or ob.get("a", [])
    if not bids or not asks:
        return

    top_b = _top_sum(bids, 5)
    top_a = _top_sum(asks, 5)
    price = f["price"]; atr5 = f["atr_m5"]

    if top_b > top_a * 1.3:
        side = "long"
    elif top_a > top_b * 1.3:
        side = "short"
    else:
        return

    equity = ctx["get_wallet_balance"]()
    risk_usdt = min(p["max_risk_usdt"], max(1.0, equity * p["risk_pct"]))
    sl = atr5 * p["sl_atr_mult"]
    if sl <= 0:
        return
    qty = max(Filters.min_qty, round(risk_usdt / sl, 6))

    tp = price + (p["tp_r_mult"] * sl) * (1 if side == "long" else -1)
    slp = price - sl if side == "long" else price + sl

    ctx["place_order"](side, qty, stop_loss=slp, take_profit=tp, reduce_only=False)
    ctx["on_entry"](strategy="density", side=side, indicator="ob-density", qty=qty, price=price)
# density.py — стаканная «плотность»: вход в сторону перевеса объёма с учётом шагов биржи и мин. нотиона
from typing import Dict, Any
import math


def _round_step(x: float, step: float) -> float:
    if step and step > 0:
        return math.floor(x / step) * step
    return x


def _top_sum(levels, n: int = 5) -> float:
    s = 0.0
    for lv in (levels or [])[:n]:
        try:
            # Bybit REST level: [price, size] как строки
            s += float(lv[1])
        except Exception:
            pass
    return s


def run_once(Filters, ctx: Dict[str, Any]):
    """
    Ожидается ctx:
      - features: {
            price: float,
            atr_m5: float,
            orderbook: { bids: [[p, q], ...], asks: [[p, q], ...] }
        }
      - params: {
            risk_pct: float,
            max_risk_usdt: float,
            sl_atr_mult: float,
            tp_r_mult: float,
            entry_atr_thresh: float,
            allow_min_qty_entry: bool,
            min_notional_usdt: float,
            density_ratio: float (>=1.0, default 1.3),
            depth_n: int (default 5)
        }
      - get_open_position, get_wallet_balance, place_order, on_entry, notify (optional)
    """
    f = ctx["features"]
    p = ctx["params"]
    notify = ctx.get("notify") or (lambda *_a, **_k: None)

    # исходные данные
    price = float(f.get("price", 0.0))
    atr5 = float(f.get("atr_m5", 0.0))
    ob = f.get("orderbook") or {}
    bids = ob.get("bids") or ob.get("b") or []
    asks = ob.get("asks") or ob.get("a") or []

    if price <= 0 or atr5 <= 0 or not bids or not asks:
        return

    # параметры плотности
    ratio = float(p.get("density_ratio", 1.3))
    depth_n = int(p.get("depth_n", 5))

    top_b = _top_sum(bids, depth_n)
    top_a = _top_sum(asks, depth_n)

    # фильтр по волатильности: не торгуем при слишком маленьком ATR
    if atr5 < float(p.get("entry_atr_thresh", 0.0)):
        return

    # определяем сторону по перевесу ликвидности
    if top_b > top_a * ratio:
        side = "long"
    elif top_a > top_b * ratio:
        side = "short"
    else:
        return

    # не добавляемся в ту же сторону
    pos = ctx["get_open_position"]()
    if pos:
        side_now = "long" if pos.get("size", 0) > 0 else "short"
        if side_now == side:
            return

    # риск и размер позиции
    equity = float(ctx["get_wallet_balance"]() or 0.0)
    if equity <= 0:
        notify("[density] пропуск: equity<=0.")
        return

    risk_usdt = min(p["max_risk_usdt"], max(1.0, equity * p["risk_pct"]))
    sl_dist = atr5 * float(p["sl_atr_mult"])
    if sl_dist <= 0:
        return

    raw_qty = risk_usdt / sl_dist
    qty_step = float(getattr(Filters, "qty_step", 0.0) or 0.0)
    min_qty = float(getattr(Filters, "min_qty", 0.0) or 0.0)
    price_tick = float(getattr(Filters, "price_tick", 0.0) or 0.0)

    qty = max(min_qty, _round_step(raw_qty, qty_step))

    # обеспечить минимальный нотионал
    min_notional = float(p.get("min_notional_usdt", 5.0))
    if qty * price < min_notional:
        need_qty = min_notional / price
        qty = max(qty, _round_step(need_qty, qty_step))

    if qty < min_qty and not p.get("allow_min_qty_entry", True):
        notify(f"[density] пропуск: qty<{min_qty} и min-лот запрещён.")
        return

    # SL / TP по ATR
    sl_price = price - sl_dist if side == "long" else price + sl_dist
    tp_price = price + (p["tp_r_mult"] * sl_dist) * (1 if side == "long" else -1)

    if price_tick > 0:
        sl_price = _round_step(sl_price, price_tick)
        tp_price = _round_step(tp_price, price_tick)

    # лог / уведомление
    notify(
        "📊 [density] signal: {side}\n"
        f"price={price:.2f} qty={qty} risk≈{risk_usdt:.2f}$\n"
        f"OB(top {depth_n}): bids={top_b:.0f} asks={top_a:.0f} ratio={ratio:.2f}\n"
        f"ATR5={atr5:.3f} SL={sl_price:.2f} TP={tp_price:.2f}"
    )

    # отправка ордера
    try:
        ctx["place_order"](side, qty, stop_loss=sl_price, take_profit=tp_price, reduce_only=False)
        ctx["on_entry"](strategy="density", side=side, indicator="ob-density", qty=qty, price=price)
    except Exception as e:
        notify(f"❌ [density] ошибка выставления ордера: {e}")
# density.py — стаканная «плотность»: вход в сторону перевеса объёма с учётом шагов биржи и мин. нотиона
from typing import Dict, Any
import math
import logging

log = logging.getLogger("STRAT.density")


def _round_step(x: float, step: float) -> float:
    if step and step > 0:
        return math.floor(x / step) * step
    return x


def _top_sum(levels, n: int = 5) -> float:
    s = 0.0
    for lv in (levels or [])[:n]:
        try:
            # Bybit REST level: [price, size] как строки
            s += float(lv[1])
        except Exception:
            pass
    return s


def run_once(Filters, ctx: Dict[str, Any]):
    """
    Ожидается ctx:
      - features: {
            price: float,
            atr_m5: float,
            orderbook: { bids: [[p, q], ...], asks: [[p, q], ...] }
        }
      - params: {
            risk_pct: float,
            max_risk_usdt: float,
            sl_atr_mult: float,
            tp_r_mult: float,
            entry_atr_thresh: float,
            allow_min_qty_entry: bool,
            min_notional_usdt: float,
            density_ratio: float (>=1.0, default 1.3),
            depth_n: int (default 5)
        }
      - get_open_position, get_wallet_balance, place_order, on_entry, notify (optional)
    """
    f = ctx["features"]
    p = ctx["params"]
    notify = ctx.get("notify") or (lambda *_a, **_k: None)

    # исходные данные
    price = float(f.get("price", 0.0))
    atr5 = float(f.get("atr_m5", 0.0))
    ob = f.get("orderbook") or {}
    bids = ob.get("bids") or ob.get("b") or []
    asks = ob.get("asks") or ob.get("a") or []

    if price <= 0 or atr5 <= 0 or not bids or not asks:
        log.debug("[density][no-entry] bad inputs price=%s atr5=%s bids=%d asks=%d", price, atr5, len(bids), len(asks))
        notify(f"[density][no-entry] некорректный вход: price={price}, atr5={atr5}, bids={len(bids)}, asks={len(asks)}")
        return

    # параметры плотности
    ratio = float(p.get("density_ratio", 1.3))
    depth_n = int(p.get("depth_n", 5))

    top_b = _top_sum(bids, depth_n)
    top_a = _top_sum(asks, depth_n)

    # фильтр по волатильности: не торгуем при слишком маленьком ATR
    if atr5 < float(p.get("entry_atr_thresh", 0.0)):
        log.debug("[density][no-entry] ATR gate atr5=%.6f < thresh=%.6f", atr5, float(p.get("entry_atr_thresh", 0.0)))
        notify(f"[density][no-entry] ATR слишком мал: atr5={atr5:.6f} < thresh={float(p.get('entry_atr_thresh', 0.0)):.6f}")
        return

    # определяем сторону по перевесу ликвидности
    if top_b > top_a * ratio:
        side = "long"
    elif top_a > top_b * ratio:
        side = "short"
    else:
        log.debug("[density][no-entry] нет перевеса: top_b=%.0f top_a=%.0f ratio=%.2f", top_b, top_a, ratio)
        notify(f"[density][no-entry] нет перевеса ликвидности: bids={top_b:.0f} vs asks={top_a:.0f}, ratio={ratio:.2f}")
        return

    # не добавляемся в ту же сторону
    pos = ctx["get_open_position"]()
    if pos:
        side_now = "long" if pos.get("size", 0) > 0 else "short"
        if side_now == side:
            log.debug("[density][no-entry] already in same side: %s", side)
            notify(f"[density][no-entry] уже в позиции {side}")
            return

    # риск и размер позиции
    equity = float(ctx["get_wallet_balance"]() or 0.0)
    if equity <= 0:
        log.debug("[density][no-entry] equity<=0")
        notify("[density][no-entry] пропуск: equity<=0.")
        return

    risk_usdt = min(p["max_risk_usdt"], max(1.0, equity * p["risk_pct"]))
    sl_dist = atr5 * float(p["sl_atr_mult"])
    if sl_dist <= 0:
        log.debug("[density][no-entry] sl_dist<=0")
        notify("[density][no-entry] SL<=0")
        return

    raw_qty = risk_usdt / sl_dist
    qty_step = float(getattr(Filters, "qty_step", 0.0) or 0.0)
    min_qty = float(getattr(Filters, "min_qty", 0.0) or 0.0)
    price_step = float(getattr(Filters, "price_step", 0.0) or getattr(Filters, "price_tick", 0.0) or 0.0)

    qty = max(min_qty, _round_step(raw_qty, qty_step))

    # обеспечить минимальный нотионал
    min_notional = float(p.get("min_notional_usdt", 5.0))
    if qty * price < min_notional:
        need_qty = min_notional / price
        qty = max(qty, _round_step(need_qty, qty_step))

    if qty < min_qty and not p.get("allow_min_qty_entry", True):
        log.debug("[density][no-entry] qty<min_qty and min lot disabled qty=%.8f min_qty=%.8f", qty, min_qty)
        notify(f"[density][no-entry] qty<{min_qty} и min-лот запрещён.")
        return

    # SL / TP по ATR
    sl_price = price - sl_dist if side == "long" else price + sl_dist
    tp_price = price + (p["tp_r_mult"] * sl_dist) * (1 if side == "long" else -1)

    if price_step > 0:
        sl_price = _round_step(sl_price, price_step)
        tp_price = _round_step(tp_price, price_step)

    # лог / уведомление
    log.info(
        "[density] signal %s price=%.2f qty=%.8f risk=%.2f OB(top%d) bids=%.0f asks=%.0f ratio=%.2f ATR5=%.6f SL=%.2f TP=%.2f",
        side, price, qty, risk_usdt, depth_n, top_b, top_a, ratio, atr5, sl_price, tp_price,
    )
    notify(
        f"📊 [density] signal: {side}\n"
        f"price={price:.2f} qty={qty} risk≈{risk_usdt:.2f}$\n"
        f"OB(top {depth_n}): bids={top_b:.0f} asks={top_a:.0f} ratio={ratio:.2f}\n"
        f"ATR5={atr5:.3f} SL={sl_price:.2f} TP={tp_price:.2f}"
    )

    # отправка ордера
    try:
        resp = ctx["place_order"](side, qty, stop_loss=sl_price, take_profit=tp_price, reduce_only=False)
        log.info("[density] order resp: %r", resp)
        notify(f"✅ [density] ордер отправлен. resp={resp}")
        ctx["on_entry"](strategy="density", side=side, indicator="ob-density", qty=qty, price=price)
    except Exception as e:
        log.exception("[density] ошибка выставления ордера")
        notify(f"❌ [density] ошибка выставления ордера: {e}")
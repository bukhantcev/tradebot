# momentum.py — трендовая импульсная логика (long/short по направлению тренда)
from typing import Dict, Any
import math
import logging

log = logging.getLogger("STRAT.momentum")


def _round_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return math.floor(x / step) * step


def run_once(Filters, ctx: Dict[str, Any]):
    """
    Ожидается ctx:
      - features: {
            price, atr_m1, atr_m5, trend_m1, trend_m5,
            tape: { buy_ratio }, tick_rate
        }
      - params: {
            risk_pct, max_risk_usdt, sl_atr_mult, tp_r_mult,
            entry_atr_thresh, allow_min_qty_entry, min_notional_usdt
        }
      - on_entry, on_exit, notify (optional)
      - get_open_position, get_wallet_balance, place_order
    """
    f = ctx["features"]
    p = ctx["params"]

    price = float(f.get("price", 0.0))
    atr1 = float(f.get("atr_m1", 0.0))
    atr5 = float(f.get("atr_m5", 0.0))
    t1 = f.get("trend_m1")
    t5 = f.get("trend_m5")
    buy_ratio = float(f.get("tape", {}).get("buy_ratio", 0.5))
    tick_rate = float(f.get("tick_rate", 0.0))

    notify = ctx.get("notify") or (lambda *_a, **_k: None)

    log.debug("[MOM] f: price=%.2f atr1=%.4f atr5=%.4f t1=%s t5=%s buy_ratio=%.3f tick_rate=%.1f", price, atr1, atr5, t1, t5, buy_ratio, tick_rate)

    # базовые проверки
    if price <= 0 or atr1 <= 0 or atr5 <= 0:
        log.debug("[MOM][no-entry] invalid data price/atr: price=%.4f atr1=%.4f atr5=%.4f", price, atr1, atr5)
        return

    # Условия импульса: согласованный тренд M1+M5 и достаточная волатильность
    up = (t1 == "up" and t5 == "up" and buy_ratio >= 0.55 and atr1 > p["entry_atr_thresh"])
    down = (t1 == "down" and t5 == "down" and buy_ratio <= 0.45 and atr1 > p["entry_atr_thresh"])

    if not (up or down):
        log.debug("[MOM][no-entry] conditions not met: t1=%s t5=%s buy_ratio=%.3f (th: up>=0.55/down<=0.45) atr1=%.3f (th=%.3f)", t1, t5, buy_ratio, atr1, p["entry_atr_thresh"])
        return

    # Проверка: нет ли уже открытой позиции в ту же сторону
    pos = ctx["get_open_position"]()
    if pos:
        side_now = "long" if pos.get("size", 0) > 0 else "short"
        want = "long" if up else "short"
        if side_now == want:
            log.debug("[MOM][no-entry] same-side position already open: want=%s have=%s size=%s", want, side_now, pos.get("size"))
            # уже едем по тренду — не добавляем
            return

    # Риск/размер
    equity = float(ctx["get_wallet_balance"]() or 0.0)
    if equity <= 0:
        log.debug("[MOM][no-entry] equity<=0")
        return

    risk_usdt = min(p["max_risk_usdt"], max(1.0, equity * p["risk_pct"]))
    # Стоп по более «медленному» ATR (меньше шума)
    sl_dist = atr5 * float(p["sl_atr_mult"])
    if sl_dist <= 0:
        return

    # Кол-во контрактов ≈ риск/стоп в $ (линейный USDT-перп)
    raw_qty = risk_usdt / sl_dist
    log.debug("[MOM] sizing: risk_usdt=%.2f sl_dist=%.5f raw_qty=%.6f", risk_usdt, sl_dist, raw_qty)

    # Биржевые фильтры
    qty = max(Filters.min_qty, _round_step(raw_qty, Filters.qty_step))
    # Минимальный нотионал
    if qty * price < float(p.get("min_notional_usdt", 5.0)):
        # попробуем поднять до минимума нотионала
        needed_qty = (p["min_notional_usdt"] / price) if price > 0 else qty
        qty = max(qty, _round_step(needed_qty, Filters.qty_step))

    log.debug("[MOM] qty after filters: qty=%.6f min_qty=%s qty_step=%s price=%.2f", qty, getattr(Filters, "min_qty", None), getattr(Filters, "qty_step", None), price)

    if qty < Filters.min_qty and not p.get("allow_min_qty_entry", True):
        log.debug("[MOM][no-entry] qty below min and min-lot not allowed: qty=%.6f min_qty=%.6f", qty, Filters.min_qty)
        return

    side = "long" if up else "short"
    # TP как мультипликатор от стопа
    tp_price = price + (p["tp_r_mult"] * sl_dist) * (1 if side == "long" else -1)
    sl_price = price - sl_dist if side == "long" else price + sl_dist

    # Доп. защита: округлим цены по шагу тика, если есть
    try:
        if getattr(Filters, "price_tick", None):
            tp_price = _round_step(tp_price, Filters.price_tick)
            sl_price = _round_step(sl_price, Filters.price_tick)
    except Exception:
        pass

    # Отправка ордера
    try:
        resp = ctx["place_order"](side, qty, stop_loss=sl_price, take_profit=tp_price, reduce_only=False)
        # Явно проверяем успешность ответа Bybit v5
        ret = None
        try:
            ret = resp.get("retCode") if isinstance(resp, dict) else None
        except Exception:
            pass
        if ret is None:
            # Если структура другая, оставим поведение как успешное, но залогируем
            log.debug("[MOM][order-raw] %s", resp)
        elif ret != 0:
            raise RuntimeError(f"retCode={ret} retMsg={getattr(resp, 'get', lambda *_: None)('retMsg')}")

        log.debug("[MOM][order-ok] %s", resp)
        notify(f"✅ [momentum] ордер отправлен ok: {resp}")
        ctx["on_entry"](strategy="momentum", side=side, indicator="impulse", qty=qty, price=price)
    except Exception as e:
        log.exception("[MOM][order-fail] %s", e)
        notify(f"❌ [momentum] ошибка выставления ордера: {e}")
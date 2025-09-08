# breakout.py — пробой ближайших SR (m5/high-low) + фильтры тренда, строгий риск-менеджмент
from typing import Dict, Any
import math
import logging

log = logging.getLogger("STRAT.breakout")


def _round_step(x: float, step: float) -> float:
    if step and step > 0:
        return math.floor(x / step) * step
    return x


def run_once(Filters, ctx: Dict[str, Any]):
    """
    Ожидается ctx:
      - features: {
            price, atr_m5, trend_m1, trend_m5, sr: { nearest_support, nearest_resistance },
            tick_rate, tape: { buy_ratio }
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
    notify = ctx.get("notify") or (lambda *_a, **_k: None)

    # ----- данные -----
    price = float(f.get("price", 0.0))
    atr5 = float(f.get("atr_m5", 0.0))
    t1 = f.get("trend_m1")
    t5 = f.get("trend_m5")
    sr = f.get("sr") or {}
    hi = sr.get("nearest_resistance")
    lo = sr.get("nearest_support")

    # шаги биржи
    qty_step = float(getattr(Filters, "qty_step", 0.0) or 0.0)
    price_step = float(getattr(Filters, "price_step", getattr(Filters, "price_tick", 0.0)) or 0.0)
    min_qty = float(getattr(Filters, "min_qty", 0.0) or 0.0)

    # ----- базовые проверки -----
    if price <= 0 or atr5 <= 0 or hi is None or lo is None:
        notify("[BRK][no-entry] пропуск: нет price/ATR5/SR.")
        log.debug("skip: price=%s atr5=%s hi=%s lo=%s", price, atr5, hi, lo)
        return

    # сетап пробоя c фильтром тренда на M1+M5
    long_setup = (price > float(hi) and t1 == "up" and t5 == "up" and atr5 >= p["entry_atr_thresh"])
    short_setup = (price < float(lo) and t1 == "down" and t5 == "down" and atr5 >= p["entry_atr_thresh"])

    if not (long_setup or short_setup):
        notify(f"[BRK][no-entry] нет сетапа. price={price:.2f} hi={float(hi):.2f} lo={float(lo):.2f} trends m1/m5={t1}/{t5} ATR5={atr5:.3f} th={p['entry_atr_thresh']}")
        log.debug("no setup: price=%.4f hi=%.4f lo=%.4f t1=%s t5=%s atr5=%.4f th=%.4f",
                  price, float(hi), float(lo), t1, t5, atr5, float(p["entry_atr_thresh"]))
        return

    # не добавляемся в ту же сторону
    pos = ctx["get_open_position"]()
    if pos:
        side_now = "long" if pos.get("size", 0) > 0 else "short"
        want = "long" if long_setup else "short"
        if side_now == want:
            notify(f"[BRK][no-entry] уже в позиции {side_now}, добавление запрещено.")
            log.debug("same-side position: side_now=%s want=%s pos=%s", side_now, want, pos)
            return

    # риск и размер
    equity = float(ctx["get_wallet_balance"]() or 0.0)
    if equity <= 0:
        notify("[BRK][no-entry] equity<=0.")
        log.debug("equity<=0: equity=%s", equity)
        return

    risk_usdt = min(p["max_risk_usdt"], max(1.0, equity * p["risk_pct"]))
    sl_dist = atr5 * float(p["sl_atr_mult"])
    if sl_dist <= 0:
        notify("[BRK][no-entry] sl_dist<=0.")
        log.debug("sl_dist<=0: atr5=%s sl_atr_mult=%s", atr5, p["sl_atr_mult"])
        return

    raw_qty = risk_usdt / sl_dist
    qty = max(min_qty, _round_step(raw_qty, qty_step))

    # проверка минимального нотионала (например $5)
    min_notional = float(p.get("min_notional_usdt", 5.0))
    if qty * price < min_notional:
        needed_qty = min_notional / price
        adj_qty = _round_step(needed_qty, qty_step)
        log.debug("adjust by min_notional: qty=%.8f -> %.8f (need %.2f$)", qty, max(qty, adj_qty), min_notional)
        qty = max(qty, adj_qty)

    if qty < min_qty and not p.get("allow_min_qty_entry", True):
        notify(f"[BRK][no-entry] qty<{min_qty} и min-лот запрещён.")
        log.debug("qty below min and not allowed: qty=%.8f min_qty=%.8f", qty, min_qty)
        return

    side = "long" if long_setup else "short"
    # Цели: SL=1*sl_dist, TP=tp_r_mult*sl_dist
    sl_price = price - sl_dist if side == "long" else price + sl_dist
    tp_price = price + (p["tp_r_mult"] * sl_dist) * (1 if side == "long" else -1)

    # округление по шагу цены
    if price_step:
        sl_price = _round_step(sl_price, price_step)
        tp_price = _round_step(tp_price, price_step)

    # финальная проверка минимального нотионала после возможных округлений
    if qty * price < min_notional and not p.get("allow_min_qty_entry", True):
        notify(f"[BRK][no-entry] после округления qty*price<{min_notional}$, вход запрещён.")
        log.debug("post-round notional too small: qty=%.8f price=%.4f notional=%.4f", qty, price, qty * price)
        return

    # нотификация о сигнале
    notify(
        f"🚀 [breakout] signal: {side}\n"
        f"price={price:.2f}  qty={qty}\n"
        f"SR: lo={float(lo):.2f} hi={float(hi):.2f} | trends m1/m5: {t1}/{t5}\n"
        f"ATR5={atr5:.2f}  SL={sl_price:.2f}  TP={tp_price:.2f}  risk≈{risk_usdt:.2f}$"
    )
    log.info("BRK enter %s | price=%.2f qty=%.8f sl=%.2f tp=%.2f risk=%.2f",
             side, price, qty, sl_price, tp_price, risk_usdt)

    # выставление ордера
    try:
        resp = ctx["place_order"](side, qty, stop_loss=sl_price, take_profit=tp_price, reduce_only=False)
        log.info("Bybit resp: %s", resp)
        notify(f"✅ [breakout] ордер выставлен. Ответ биржи: {resp}")
        ctx["on_entry"](strategy="breakout", side=side, indicator="sr-break", qty=qty, price=price)
    except Exception as e:
        log.exception("order placement failed")
        notify(f"❌ [breakout] ошибка выставления ордера: {e}")
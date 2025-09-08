from __future__ import annotations
from typing import Dict, Any

def run_once(Filters, ctx: Dict[str, Any]):
    f = ctx["features"]; p = ctx["params"]
    price = f["price"]; atr1 = f["atr_m1"]; atr5 = f["atr_m5"]
    t1 = f["trend_m1"]; t5 = f["trend_m5"]
    sr = f["sr"]
    if not sr:
        return

    # условия: высокий atr, цена близко к уровням
    vol_ok = (atr1 > p["entry_atr_thresh"] and atr5 > p["entry_atr_thresh"])
    near_low = abs(price - sr["nearest_support"]) <= atr1 * 0.5
    near_high = abs(price - sr["nearest_resistance"]) <= atr1 * 0.5

    # против тренда — короткий скальп с быстрым TP
    if vol_ok and near_low and (t1 == "down" or t5 == "down"):
        side = "long"
    elif vol_ok and near_high and (t1 == "up" or t5 == "up"):
        side = "short"
    else:
        return

    equity = ctx["get_wallet_balance"]()
    risk_usdt = min(p["max_risk_usdt"], max(1.0, equity * p["risk_pct"]))
    sl = atr1 * (p["sl_atr_mult"] * 0.7)  # покороче стоп
    if sl <= 0:
        return
    qty = max(Filters.min_qty, round(risk_usdt / sl, 6))

    tp = price + (p["tp_r_mult"] * 0.8 * sl) * (1 if side == "long" else -1)  # быстрее фиксируем
    slp = price - sl if side == "long" else price + sl

    ctx["place_order"](side, qty, stop_loss=slp, take_profit=tp, reduce_only=False)
    ctx["on_entry"](strategy="knife", side=side, indicator="sr-knife", qty=qty, price=price)
# knife.py — “ловля ножа”: короткие отскоки от экстремумов при острой волатильности

def _round_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    # безопасное округление к шагу биржи
    return round(round(x / step) * step, 10)

def run_once(Filters, ctx: Dict[str, Any]):
    """
    Требования к ctx:
      ctx["features"]: Dict с ключами price, atr_m1, atr_m5, trend_m1, trend_m5, sr, tape, orderbook
      ctx["params"]:   Dict с ключами risk_pct, max_risk_usdt, sl_atr_mult, tp_r_mult,
                                  entry_atr_thresh, allow_min_qty_entry, min_notional_usdt
      ctx["get_wallet_balance"](): -> float
      ctx["place_order"](side:str, qty:float, stop_loss:float, take_profit:float, reduce_only:bool) -> Any
      ctx["on_entry"](**kw) — синхронный колбэк (main оборачивает async если нужно)
      ctx.get("notify")(msg:str) — опционально, шлёт в ТГ
    """
    f = ctx["features"]; p = ctx["params"]
    price = float(f.get("price", 0.0))
    atr1  = float(f.get("atr_m1", 0.0))
    atr5  = float(f.get("atr_m5", 0.0))
    t1    = f.get("trend_m1", "flat")
    t5    = f.get("trend_m5", "flat")
    sr    = f.get("sr") or {}
    tape  = f.get("tape") or {}
    buy_ratio = float(tape.get("buy_ratio", 0.5))

    notify = ctx.get("notify")
    def _n(msg: str):
        try:
            if notify:
                notify(msg)
        except Exception:
            pass

    if not price or atr1 <= 0 or not sr:
        _n("🧪 [knife] skip: no data (price/atr/sr)")
        return

    # Основные условия: высокая волатильность и близость к уровням
    vol_ok   = (atr1 >= p["entry_atr_thresh"] and atr5 >= p["entry_atr_thresh"])
    near_low  = abs(price - sr.get("nearest_support", price*2))     <= atr1 * 0.5
    near_high = abs(price - sr.get("nearest_resistance", -1.0))     <= atr1 * 0.5

    # Доп. фильтры:
    #  - для лонга у уровня поддержки желательно перепроданность (низкий buy_ratio)
    #  - для шорта у сопротивления — перекупленность (высокий buy_ratio)
    oversold  = buy_ratio <= 0.45
    overbought= buy_ratio >= 0.55

    side: str | None = None
    reason = []

    if vol_ok and near_low and (t1 == "down" or t5 == "down") and oversold:
        side = "long"; reason = ["vol_ok","near_low","trend_down","oversold"]
    elif vol_ok and near_high and (t1 == "up" or t5 == "up") and overbought:
        side = "short"; reason = ["vol_ok","near_high","trend_up","overbought"]

    if not side:
        # Removed verbose no-entry notification to reduce spam
        # _n(f"🧪 [knife] no-entry: vol_ok={vol_ok} near_low={near_low} near_high={near_high} "
        #    f"t1={t1} t5={t5} buy_ratio={buy_ratio:.2f} atr1={atr1:.4f}")
        return

    # Risk & sizing
    equity     = float(ctx["get_wallet_balance"]())
    risk_usdt  = min(float(p["max_risk_usdt"]), max(1.0, equity * float(p["risk_pct"])))
    sl_points  = atr1 * float(p["sl_atr_mult"]) * 0.7  # более короткий стоп для “ножа”
    if sl_points <= 0:
        _n("🧪 [knife] skip: sl_points<=0")
        return

    # начальный размер (в точках цены): риск / дистанция стопа
    raw_qty = risk_usdt / sl_points
    # округление к шагу лота
    qty = max(float(getattr(Filters, "min_qty", 0.0)), _round_step(raw_qty, float(getattr(Filters, "qty_step", 0.0))))

    # Проверки на min notional и флаг allow_min_qty_entry
    notional = price * qty
    if notional < float(p["min_notional_usdt"]):
        if not p.get("allow_min_qty_entry", True):
            _n(f"🧪 [knife] no-entry: notional {notional:.2f} &lt; min_notional {p['min_notional_usdt']}")
            return
        # увеличим размер до минимального нотионала и снова округлим
        min_qty_notional = float(p["min_notional_usdt"]) / max(price, 1e-9)
        qty = _round_step(max(qty, min_qty_notional), float(getattr(Filters, "qty_step", 0.0)))
        qty = max(qty, float(getattr(Filters, "min_qty", 0.0)))
        notional = price * qty

    if qty <= 0:
        _n("🧪 [knife] skip: qty<=0 after rounding")
        return

    # Цели
    tp_points = float(p["tp_r_mult"]) * 0.8 * sl_points  # быстрый тейк
    if side == "long":
        tp  = price + tp_points
        slp = price - sl_points
    else:
        tp  = price - tp_points
        slp = price + sl_points

    # Отправим ордер
    ctx["place_order"](side, qty, stop_loss=slp, take_profit=tp, reduce_only=False)
    ctx["on_entry"](strategy="knife", side=side, indicator=f"knife({'/'.join(reason)})",
                    qty=qty, price=price)
    _n(f"✅ [knife] entry side={side} qty={qty} px={price:.2f} sl={slp:.2f} tp={tp:.2f} "
       f"(atr1={atr1:.4f}, buy_ratio={buy_ratio:.2f})")
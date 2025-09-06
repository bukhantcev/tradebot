from typing import Dict, List
import math


def run_once(filters, context) -> None:
    """
    Breakout trading (пробои уровней):
    Торгуем пробой ближайших S/R по направлению локального тренда (M5),
    используя подтверждение потоком (buy_ratio) и минимальной волатильностью (ATR).

    После успешного входа вызываем callback из контекста:
      - on_entry(strategy, side, indicator, qty, price)
    При неудаче — подробный DEBUG/ERROR логи для диагностики.
    """

    # Late imports to avoid circular deps
    from main import (
        logger,
        tg_send,
        place_market_order,
        get_open_position,
        get_wallet_balance,
        SYMBOL,
        RISK_PCT,
        MAX_RISK_USDT,
        SL_ATR_MULT,
        TP_R_MULT,
        ENTRY_ATR_THRESH,
    )

    feats: Dict = context.get("features", {}) or {}
    if not feats:
        logger.debug("[BRK] no features")
        return

    price = float(feats.get("price", 0.0) or 0.0)
    atr_m1 = float(feats.get("atr_m1", 0.0) or 0.0)
    trend_m5: str = feats.get("trend_m5", "flat")

    sr = feats.get("sr", {}) or {}
    near_sup = sr.get("nearest_support")
    near_res = sr.get("nearest_resistance")
    d_sup = sr.get("dist_to_support")
    d_res = sr.get("dist_to_resistance")

    tape = feats.get("tape", {}) or {}
    buy_ratio = float(tape.get("buy_ratio", 0.5) or 0.5)

    if price <= 0 or atr_m1 <= 0:
        logger.debug("[BRK] invalid price/atr")
        return

    # Не заходим, если уже есть позиция
    pos = get_open_position()
    if pos is not None:
        side, qty = pos
        if qty >= filters.min_qty:
            logger.debug("[BRK] position exists: %s %.6f — skip", side, qty)
            return

    # Параметры логики пробоя
    # Порог близости к уровню: чем меньше ATR, тем ближе ждём к самому уровню
    max_dist_atr = max(ENTRY_ATR_THRESH, 0.25)
    flow_up_ok = buy_ratio >= 0.55
    flow_dn_ok = buy_ratio <= 0.45

    # Условия LONG (пробой сопротивления по тренду)
    long_trend_ok = trend_m5 == "up"
    long_level_ok = near_res is not None and d_res is not None and d_res <= max_dist_atr * atr_m1
    enter_long = long_trend_ok and long_level_ok and flow_up_ok

    # Условия SHORT (пробой поддержки по тренду)
    short_trend_ok = trend_m5 == "down"
    short_level_ok = near_sup is not None and d_sup is not None and d_sup <= max_dist_atr * atr_m1
    enter_short = short_trend_ok and short_level_ok and flow_dn_ok

    if not enter_long and not enter_short:
        logger.debug(
            "[BRK] no setup | trend5=%s buy_ratio=%.2f dS=%s dR=%s price=%.2f ATR=%.2f",
            trend_m5, buy_ratio, d_sup, d_res, price, atr_m1,
        )
        return

    entry_side = "long" if enter_long and not enter_short else "short"

    # Расчёт SL/TP вокруг уровня + k*ATR
    k_sl = max(1.0, SL_ATR_MULT)
    if entry_side == "long":
        base = near_res if near_res is not None else price
        sl_price = max(0.01, base - k_sl * atr_m1)
        risk_per_coin = price - sl_price
    else:
        base = near_sup if near_sup is not None else price
        sl_price = base + k_sl * atr_m1
        risk_per_coin = sl_price - price

    if risk_per_coin <= 0:
        logger.debug("[BRK] non-positive risk")
        return

    equity = get_wallet_balance()
    risk_usdt = min(equity * RISK_PCT, MAX_RISK_USDT)
    raw_qty = risk_usdt / risk_per_coin
    qty = math.floor(raw_qty / filters.qty_step) * filters.qty_step
    if qty < filters.min_qty:
        logger.debug("[BRK] qty below min: %.8f < %.8f", qty, filters.min_qty)
        return

    # TP в R-мультипликаторах
    if entry_side == "long":
        tp_price = price + TP_R_MULT * (price - sl_price)
    else:
        tp_price = price - TP_R_MULT * (sl_price - price)

    # Подготовим краткую сводку индикаторов
    indicator_summary = (
        f"trend5={trend_m5} buy_ratio={buy_ratio:.2f} "
        f"dS={d_sup} dR={d_res} ATR1={atr_m1:.2f}"
    )

    # Callback из контекста (если передан)
    on_entry_cb = context.get("on_entry")

    try:
        res = place_market_order(entry_side, qty, stop_loss=sl_price, take_profit=tp_price, reduce_only=False)
        # Пытаемся достать фактическую fill-цену
        result = (res or {}).get("result", {})
        fills = result.get("list") or []
        # Bybit v5 spot/contract ответ может отличаться — пробуем несколько вариантов
        fill_price = None
        if fills and isinstance(fills, list):
            fp = fills[0]
            fill_price = (
                fp.get("avgPrice") or fp.get("cumExecAvgPrice") or fp.get("orderPrice") or fp.get("price")
            )
        if fill_price is None:
            fill_price = price
        else:
            try:
                fill_price = float(fill_price)
            except Exception:
                fill_price = price

        oid = result.get("orderId") or result.get("orderID") or "?"

        # Отправка в TG
        tg_send(
            f"🚀 <b>BREAKOUT</b> {SYMBOL}\n"
            f"Side: <b>{entry_side.upper()}</b> | Qty: <code>{qty}</code>\n"
            f"Price: <code>{fill_price:.2f}</code> | SL: <code>{sl_price:.2f}</code> | TP: <code>{tp_price:.2f}</code>\n"
            f"{indicator_summary}"
        )

        logger.info(
            "[BRK][ORDER] %s qty=%.6f @%.2f SL=%.2f TP=%.2f id=%s",
            entry_side, qty, fill_price, sl_price, tp_price, oid,
        )

        # ВАЖНО: триггерим колбэк on_entry, если он предоставлен
        if callable(on_entry_cb):
            try:
                on_entry_cb(
                    strategy="breakout",
                    side=entry_side,
                    indicator=indicator_summary,
                    qty=qty,
                    price=fill_price,
                )
            except Exception as cb_e:
                logger.warning("[BRK][on_entry] callback failed: %s", cb_e)

    except Exception as e:
        logger.error("[BRK][ORDER] failed: %s", e)
        tg_send(f"❌ <b>BREAKOUT order failed</b> {SYMBOL}\n<code>{e}</code>")

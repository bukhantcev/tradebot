

from typing import Dict
import math


def run_once(filters, context) -> None:
    """
    Momentum trading (торговля по импульсу):
    Заходим в сторону явного движения, когда лента и дисбаланс подтверждают силу тренда.

    LONG:
      - Тренд M1 или M5 = up
      - Лента: buy_ratio >= 0.65, tick_rate высокий
      - Orderbook imbalance > 0
    SHORT:
      - Тренд M1 или M5 = down
      - Лента: buy_ratio <= 0.35
      - Orderbook imbalance < 0

    SL — k*ATR за текущей ценой, TP = R*TP_R_MULT.
    """

    # Late imports
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
    )

    feats: Dict = context.get("features", {}) or {}
    if not feats:
        logger.debug("[MOM] no features")
        return

    price = float(feats.get("price", 0.0) or 0.0)
    atr_m1 = float(feats.get("atr_m1", 0.0) or 0.0)
    trend_m1 = feats.get("trend_m1", "flat")
    trend_m5 = feats.get("trend_m5", "flat")

    ob = feats.get("orderbook", {}) or {}
    imb = float(ob.get("imbalance_bp", 0.0) or 0.0)

    tape = feats.get("tape", {}) or {}
    buy_ratio = float(tape.get("buy_ratio", 0.5) or 0.5)
    tick_rate = float(tape.get("tick_rate_per_min", 0.0) or 0.0)

    if price <= 0 or atr_m1 <= 0:
        logger.debug("[MOM] invalid price/atr")
        return

    pos = get_open_position()
    if pos is not None:
        side, qty = pos
        if qty >= filters.min_qty:
            logger.debug("[MOM] position exists: %s %.6f — skip", side, qty)
            return

    min_tick_rate = 150.0

    enter_long = (
        (trend_m1 == "up" or trend_m5 == "up")
        and buy_ratio >= 0.65
        and tick_rate >= min_tick_rate
        and imb > 0.0
    )

    enter_short = (
        (trend_m1 == "down" or trend_m5 == "down")
        and buy_ratio <= 0.35
        and tick_rate >= min_tick_rate
        and imb < 0.0
    )

    if not enter_long and not enter_short:
        logger.debug("[MOM] no setup | t1=%s t5=%s buy_ratio=%.2f tick=%.0f imb=%.2f",
                     trend_m1, trend_m5, buy_ratio, tick_rate, imb)
        return

    entry_side = "long" if enter_long and not enter_short else "short"

    k_sl = max(1.0, SL_ATR_MULT)
    if entry_side == "long":
        sl_price = max(0.01, price - k_sl * atr_m1)
        risk_per_coin = price - sl_price
    else:
        sl_price = price + k_sl * atr_m1
        risk_per_coin = sl_price - price

    if risk_per_coin <= 0:
        logger.debug("[MOM] non-positive risk")
        return

    equity = get_wallet_balance()
    risk_usdt = min(equity * RISK_PCT, MAX_RISK_USDT)
    raw_qty = risk_usdt / risk_per_coin
    qty = math.floor(raw_qty / filters.qty_step) * filters.qty_step
    if qty < filters.min_qty:
        logger.debug("[MOM] qty below min: %.8f < %.8f", qty, filters.min_qty)
        return

    if entry_side == "long":
        tp_price = price + TP_R_MULT * (price - sl_price)
    else:
        tp_price = price - TP_R_MULT * (sl_price - price)

    try:
        res = place_market_order(entry_side, qty, stop_loss=sl_price, take_profit=tp_price, reduce_only=False)
        oid = res["result"]["orderId"]
        # Unified entry callback (if provided by main)
        on_entry = context.get("on_entry") if isinstance(context, dict) else None
        if callable(on_entry):
            try:
                on_entry(
                    strategy="momentum",
                    side=entry_side,
                    indicator="tape+imbalance",
                    qty=qty,
                    price=price,
                )
            except Exception as cb_err:
                logger.error("[MOM][CB][on_entry] %s", cb_err)
        tg_send(
            f"⚡ <b>MOMENTUM</b> {SYMBOL}\n"
            f"Side: <b>{entry_side.upper()}</b> | Qty: <code>{qty}</code>\n"
            f"Price: <code>{price:.2f}</code> | SL: <code>{sl_price:.2f}</code> | TP: <code>{tp_price:.2f}</code>\n"
            f"t1={trend_m1} t5={trend_m5} buy_ratio={buy_ratio:.2f} tick={tick_rate:.0f}/m imb={imb:.2f}"
        )
        logger.info("[MOM][ORDER] %s qty=%.6f @%.2f SL=%.2f TP=%.2f id=%s",
                    entry_side, qty, price, sl_price, tp_price, oid)
    except Exception as e:
        logger.error("[MOM][ORDER] failed: %s", e)
        tg_send(f"❌ <b>MOMENTUM order failed</b> {SYMBOL}\n<code>{e}</code>")
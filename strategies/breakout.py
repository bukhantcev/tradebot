

from typing import Dict
import math


def run_once(filters, context) -> None:
    """
    Breakout trading (торговля на пробой уровней):
    Входим по направлению, когда цена атакует ближайший уровень сопротивления/поддержки
    + подтверждается лентой (агрессия) и свечным импульсом.

    LONG:
      - Тренд M5 = up
      - Цена близко к сопротивлению (<= ENTRY_ATR_THRESH * ATR)
      - Лента в покупках (buy_ratio >= 0.6)
    SHORT:
      - Тренд M5 = down
      - Цена близко к поддержке
      - Лента в продажах (buy_ratio <= 0.4)

    SL ставим за уровень ± k*ATR, TP = R*TP_R_MULT.
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
    atr_m5 = float(feats.get("atr_m5", 0.0) or 0.0)
    trend_m5 = feats.get("trend_m5", "flat")

    sr = feats.get("sr", {}) or {}
    near_sup = sr.get("nearest_support")
    near_res = sr.get("nearest_resistance")
    d_sup = sr.get("dist_to_support")
    d_res = sr.get("dist_to_resistance")

    tape = feats.get("tape", {}) or {}
    buy_ratio = float(tape.get("buy_ratio", 0.5) or 0.5)
    tick_rate = float(tape.get("tick_rate_per_min", 0.0) or 0.0)

    if price <= 0 or atr_m5 <= 0:
        logger.debug("[BRK] invalid price/atr")
        return

    pos = get_open_position()
    if pos is not None:
        side, qty = pos
        if qty >= filters.min_qty:
            logger.debug("[BRK] position exists: %s %.6f — skip", side, qty)
            return

    max_dist_atr = max(ENTRY_ATR_THRESH, 0.3)
    min_tick_rate = 100.0

    enter_long = False
    enter_short = False

    if (
        trend_m5 == "up"
        and near_res is not None and d_res is not None and d_res <= max_dist_atr * atr_m5
        and buy_ratio >= 0.60
        and tick_rate >= min_tick_rate
    ):
        enter_long = True

    if (
        trend_m5 == "down"
        and near_sup is not None and d_sup is not None and d_sup <= max_dist_atr * atr_m5
        and buy_ratio <= 0.40
        and tick_rate >= min_tick_rate
    ):
        enter_short = True

    if not enter_long and not enter_short:
        logger.debug("[BRK] no setup | trend5=%s buy_ratio=%.2f dS=%s dR=%s tick=%.1f",
                     trend_m5, buy_ratio, d_sup, d_res, tick_rate)
        return

    entry_side = "long" if enter_long and not enter_short else "short"

    k_sl = max(1.0, SL_ATR_MULT)
    if entry_side == "long":
        base = near_res if near_res is not None else price
        sl_price = max(0.01, base - k_sl * atr_m5)
        risk_per_coin = price - sl_price
    else:
        base = near_sup if near_sup is not None else price
        sl_price = base + k_sl * atr_m5
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

    if entry_side == "long":
        tp_price = price + TP_R_MULT * (price - sl_price)
    else:
        tp_price = price - TP_R_MULT * (sl_price - price)

    try:
        res = place_market_order(entry_side, qty, stop_loss=sl_price, take_profit=tp_price, reduce_only=False)
        oid = res["result"]["orderId"]
        # ---- Strategy callbacks (optional) ----
        # Build compact indicator summary for journaling
        indicator_summary = (
            f"trend5={trend_m5}; buy={buy_ratio:.2f}; tick={tick_rate:.0f}/m; "
            f"dS={d_sup if d_sup is not None else 'na'}; "
            f"dR={d_res if d_res is not None else 'na'}; "
            f"ATR5={atr_m5:.2f}"
        )

        # Try to infer fill/entry price from response
        fill_price = None
        try:
            rr = res.get("result") or {}
            # Common fields that Bybit may return depending on endpoint
            for k in ("avgPrice", "cumExecAvgPrice", "orderPrice", "price"):
                if k in rr and rr[k] not in (None, ""):
                    fill_price = float(rr[k])
                    break
        except Exception:
            fill_price = None
        if not fill_price:
            fill_price = float(price)

        # Fire on_entry callback if provided by main
        on_entry_cb = context.get("on_entry")
        if callable(on_entry_cb):
            try:
                on_entry_cb(
                    strategy="breakout",
                    side=entry_side,
                    indicator=indicator_summary,
                    qty=qty,
                    price=fill_price,
                )
            except Exception as _cb_err:
                logger.debug("[BRK][CB] on_entry failed: %s", _cb_err)
        tg_send(
            f"🚀 <b>BREAKOUT</b> {SYMBOL}\n"
            f"Side: <b>{entry_side.upper()}</b> | Qty: <code>{qty}</code>\n"
            f"Price: <code>{price:.2f}</code> | SL: <code>{sl_price:.2f}</code> | TP: <code>{tp_price:.2f}</code>\n"
            f"trend5={trend_m5} buy_ratio={buy_ratio:.2f} tick={tick_rate:.0f}/m"
        )
        logger.info("[BRK][ORDER] %s qty=%.6f @%.2f SL=%.2f TP=%.2f id=%s",
                    entry_side, qty, price, sl_price, tp_price, oid)
    except Exception as e:
        logger.debug(
            "[BRK][ORDER] context | side=%s qty=%.6f price=%.2f dS=%s dR=%s buy=%.2f tick=%.1f atr5=%.2f",
            entry_side, qty, price, d_sup, d_res, buy_ratio, tick_rate, atr_m5,
        )
        logger.error("[BRK][ORDER] failed: %s", e)
        tg_send(f"❌ <b>BREAKOUT order failed</b> {SYMBOL}\n<code>{e}</code>")
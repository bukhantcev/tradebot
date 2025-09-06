from typing import Dict
import math


def run_once(filters, context) -> None:
    """
    Knife catching scalp (контртренд от уровня/плотности):
    - LONG: агрессия продавцов (sell-dominant) + цена у поддержки + плотные биды → берём откат вверх.
    - SHORT: агрессия покупателей (buy-dominant) + цена у сопротивления + плотные офферы → берём откат вниз.

    Требуется окружение из main.py: logger, tg_send, place_market_order, get_open_position,
    close_position_market, get_wallet_balance, SYMBOL, RISK_PCT, MAX_RISK_USDT, SL_ATR_MULT, TP_R_MULT.
    Чтобы избежать циклического импорта, импортируем внутри функции.
    Вызывает context["on_entry"] после успешного размещения ордера.
    """

    # Late imports to avoid circular deps
    from main import (
        logger,
        tg_send,
        place_market_order,
        get_open_position,
        close_position_market,
        get_wallet_balance,
        SYMBOL,
        RISK_PCT,
        MAX_RISK_USDT,
        SL_ATR_MULT,
        TP_R_MULT,
        ENTRY_ATR_THRESH,
    )

    feats: Dict = context.get("features", {})
    if not feats:
        logger.debug("[KNIFE] no features")
        return

    price = float(feats.get("price", 0.0) or 0.0)
    atr_m1 = float(feats.get("atr_m1", 0.0) or 0.0)
    sr = feats.get("sr", {}) or {}
    near_sup = sr.get("nearest_support")
    near_res = sr.get("nearest_resistance")
    d_sup = sr.get("dist_to_support")
    d_res = sr.get("dist_to_resistance")

    ob = feats.get("orderbook", {}) or {}
    imb = float(ob.get("imbalance_bp", 0.0) or 0.0)
    top_bids = ob.get("top_bids", [])
    top_asks = ob.get("top_asks", [])

    tape = feats.get("tape", {}) or {}
    buy_ratio = float(tape.get("buy_ratio", 0.5) or 0.5)
    tick_rate = float(tape.get("tick_rate_per_min", 0.0) or 0.0)

    # Safety checks
    if price <= 0 or atr_m1 <= 0:
        logger.debug("[KNIFE] invalid price/atr: price=%.6f atr=%.6f", price, atr_m1)
        return

    # Skip if already in position (стратегия не усредняет и не переворачивается сама)
    pos = get_open_position()
    if pos is not None:
        side, qty = pos
        if qty >= filters.min_qty:
            logger.debug("[KNIFE] position exists: %s %.6f — skip", side, qty)
            return

    # Thresholds (мягкие, можно выносить в ENV при желании)
    max_dist_atr = max(ENTRY_ATR_THRESH, 0.2)  # допустимое расстояние до уровня в ATR
    min_tick_rate = 120.0                       # импульс/паника по тикам в минуту
    strong_density_mult = 5.0                   # что считаем «плотностью»: кратно лучшему объёму

    # Функции оценки плотностей
    def _max_sz(items):
        return max((float(x.get("sz", 0.0) or 0.0) for x in items), default=0.0)

    best_bid_sz = float(ob.get("best_bid_sz", 0.0) or 0.0)
    best_ask_sz = float(ob.get("best_ask_sz", 0.0) or 0.0)
    dens_bid = _max_sz(top_bids)
    dens_ask = _max_sz(top_asks)

    # Сигналы на вход
    enter_long = False
    enter_short = False

    # LONG: паника вниз (sell>buy), высокая скорость тиков, рядом поддержка, сильные бид-плотности, дисбаланс в плюсе
    if (
        buy_ratio < 0.40
        and tick_rate >= min_tick_rate
        and near_sup is not None and d_sup is not None and d_sup <= max_dist_atr * atr_m1
        and dens_bid >= strong_density_mult * max(1e-9, best_bid_sz)
        and imb > 0.0
    ):
        enter_long = True

    # SHORT: всплеск вверх (buy доминируют), рядом сопротивление, сильные офферы, дисбаланс в минус
    if (
        buy_ratio > 0.60
        and tick_rate >= min_tick_rate
        and near_res is not None and d_res is not None and d_res <= max_dist_atr * atr_m1
        and dens_ask >= strong_density_mult * max(1e-9, best_ask_sz)
        and imb < 0.0
    ):
        enter_short = True

    if not enter_long and not enter_short:
        logger.debug("[KNIFE] no setup | buy_ratio=%.2f tick=%.1f dS=%s dR=%s densB=%.2f densA=%.2f imb=%.2f",
                     buy_ratio, tick_rate, d_sup, d_res, dens_bid, dens_ask, imb)
        return

    entry_side = "long" if enter_long and not enter_short else "short"

    # Стоп-лосс ставим за ближайшим уровнем ± k*ATR
    k_sl = max(0.8, SL_ATR_MULT)  # чуть агрессивнее стандартного SL
    if entry_side == "long":
        base_level = near_sup if near_sup is not None else price - atr_m1
        sl_price = max(0.01, base_level - k_sl * atr_m1)
        risk_per_coin = price - sl_price
    else:
        base_level = near_res if near_res is not None else price + atr_m1
        sl_price = base_level + k_sl * atr_m1
        risk_per_coin = sl_price - price

    if risk_per_coin <= 0:
        logger.debug("[KNIFE] non-positive risk per coin")
        return

    # Размер позиции по риску
    equity = get_wallet_balance()
    risk_usdt = min(equity * RISK_PCT, MAX_RISK_USDT)
    raw_qty = risk_usdt / risk_per_coin
    qty = math.floor(raw_qty / filters.qty_step) * filters.qty_step
    if qty < filters.min_qty:
        logger.debug("[KNIFE] qty below min: %.8f < %.8f", qty, filters.min_qty)
        return

    # TP как 1R (можно поправить в ENV через TP_R_MULT)
    if entry_side == "long":
        tp_price = price + TP_R_MULT * (price - sl_price)
    else:
        tp_price = price - TP_R_MULT * (sl_price - price)

    # Place order
    try:
        res = place_market_order(entry_side, qty, stop_loss=sl_price, take_profit=tp_price, reduce_only=False)
        oid = res["result"]["orderId"]
        logger.info("[KNIFE][ORDER] %s qty=%.6f @%.2f SL=%.2f TP=%.2f id=%s",
                    entry_side, qty, price, sl_price, tp_price, oid)
        tg_send(
            f"🔪 <b>KNIFE</b> {SYMBOL}\n"
            f"Side: <b>{entry_side.upper()}</b> | Qty: <code>{qty}</code>\n"
            f"Price: <code>{price:.2f}</code> | SL: <code>{sl_price:.2f}</code> | TP: <code>{tp_price:.2f}</code>\n"
            f"buy_ratio={buy_ratio:.2f} tick={tick_rate:.0f}/m imb={imb:.2f}"
        )
        # Strategy-level on-entry callback for unified TG/reporting pipeline
        cb = context.get("on_entry") if isinstance(context, dict) else None
        if callable(cb):
            try:
                cb(
                    strategy="knife",
                    side=entry_side,
                    indicator="SR-density+flow",
                    qty=qty,
                    price=price,
                )
            except Exception as cb_e:
                logger.error("[KNIFE][CB][on_entry] failed: %s", cb_e)
    except Exception as e:
        logger.error("[KNIFE][ORDER] failed: %s", e)
        tg_send(f"❌ <b>KNIFE order failed</b> {SYMBOL}\n<code>{e}</code>")
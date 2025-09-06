from typing import Dict, List
import math


def run_once(filters, context) -> None:
    """
    Density trading (торговля от плотностей в стакане):
    Играем от стоящих лимитных «стенок», предпочтительно по направлению локального тренда (M5).

    LONG (от бида):
      - Тренд M5 = up (или flat при очень сильной плотности)
      - Цена близко к поддержке (из S/R) ИЛИ к ближайшей сильной бид-плотности
      - В стакане есть «сильный» bid (мультипликатор к лучшему объёму)
      - Лента не доминируется продажами (buy_ratio >= 0.45)

    SHORT (от оффера):
      - Тренд M5 = down (или flat при очень сильной плотности)
      - Цена близко к сопротивлению ИЛИ к ближайшей сильной ask-плотности
      - В стакане есть «сильный» ask
      - Лента не доминируется покупками (buy_ratio <= 0.55)

    Размер позиции — по риску, SL — за уровень ± k*ATR, TP = R*TP_R_MULT.
    Важные объекты импортируем поздно из main.py, чтобы избежать циклических импортов.
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
        logger.debug("[DENS] no features")
        return

    price = float(feats.get("price", 0.0) or 0.0)
    atr_m1 = float(feats.get("atr_m1", 0.0) or 0.0)
    trend_m5: str = feats.get("trend_m5", "flat")

    sr = feats.get("sr", {}) or {}
    near_sup = sr.get("nearest_support")
    near_res = sr.get("nearest_resistance")
    d_sup = sr.get("dist_to_support")
    d_res = sr.get("dist_to_resistance")

    ob = feats.get("orderbook", {}) or {}
    top_bids: List[Dict] = ob.get("top_bids", []) or []
    top_asks: List[Dict] = ob.get("top_asks", []) or []
    best_bid_sz = float(ob.get("best_bid_sz", 0.0) or 0.0)
    best_ask_sz = float(ob.get("best_ask_sz", 0.0) or 0.0)

    tape = feats.get("tape", {}) or {}
    buy_ratio = float(tape.get("buy_ratio", 0.5) or 0.5)

    if price <= 0 or atr_m1 <= 0:
        logger.debug("[DENS] invalid price/atr")
        return

    # Не влезаем, если уже есть позиция
    pos = get_open_position()
    if pos is not None:
        side, qty = pos
        if qty >= filters.min_qty:
            logger.debug("[DENS] position exists: %s %.6f — skip", side, qty)
            return

    # Параметры (можно вынести в ENV при желании)
    max_dist_atr = max(ENTRY_ATR_THRESH, 0.25)    # близость к уровню (в ATR)
    strong_mult = 5.0                              # критерий «сильной» стенки к лучшему объёму
    allow_flat_if_super_strong = 10.0              # если стенка >10x best, допускаем flat

    def nearest_density(dens: List[Dict], px: float) -> Dict:
        if not dens:
            return {}
        return min(dens, key=lambda x: abs(px - float(x.get("price", px))))

    dens_bid = nearest_density(top_bids, price)
    dens_ask = nearest_density(top_asks, price)
    dens_bid_sz = float(dens_bid.get("sz", 0.0) or 0.0)
    dens_ask_sz = float(dens_ask.get("sz", 0.0) or 0.0)
    dens_bid_px = float(dens_bid.get("price", price) or price)
    dens_ask_px = float(dens_ask.get("price", price) or price)

    d_to_bid = abs(price - dens_bid_px)
    d_to_ask = abs(price - dens_ask_px)

    # Условия LONG
    long_trend_ok = (trend_m5 == "up") or (best_bid_sz > 0 and dens_bid_sz >= allow_flat_if_super_strong * best_bid_sz)
    long_level_ok = (near_sup is not None and d_sup is not None and d_sup <= max_dist_atr * atr_m1) or (d_to_bid <= max_dist_atr * atr_m1)
    long_density_ok = best_bid_sz > 0 and dens_bid_sz >= strong_mult * best_bid_sz
    long_flow_ok = buy_ratio >= 0.45  # не обязателен явный buy-dominance, главное — не продавцы

    enter_long = long_trend_ok and long_level_ok and long_density_ok and long_flow_ok

    # Условия SHORT
    short_trend_ok = (trend_m5 == "down") or (best_ask_sz > 0 and dens_ask_sz >= allow_flat_if_super_strong * best_ask_sz)
    short_level_ok = (near_res is not None and d_res is not None and d_res <= max_dist_atr * atr_m1) or (d_to_ask <= max_dist_atr * atr_m1)
    short_density_ok = best_ask_sz > 0 and dens_ask_sz >= strong_mult * best_ask_sz
    short_flow_ok = buy_ratio <= 0.55

    enter_short = short_trend_ok and short_level_ok and short_density_ok and short_flow_ok

    if not enter_long and not enter_short:
        logger.debug(
            "[DENS] no setup | trend5=%s buy_ratio=%.2f dS=%s dR=%s dBid=%.2f dAsk=%.2f szBid=%.2f szAsk=%.2f",
            trend_m5, buy_ratio, d_sup, d_res, d_to_bid, d_to_ask, dens_bid_sz, dens_ask_sz,
        )
        return

    entry_side = "long" if enter_long and not enter_short else "short"

    # SL/TP: за стенку/уровень + k*ATR
    k_sl = max(1.0, SL_ATR_MULT)
    if entry_side == "long":
        base = min(x for x in [near_sup, dens_bid_px, price - atr_m1] if x is not None)
        sl_price = max(0.01, base - k_sl * atr_m1)
        risk_per_coin = price - sl_price
    else:
        base = max(x for x in [near_res, dens_ask_px, price + atr_m1] if x is not None)
        sl_price = base + k_sl * atr_m1
        risk_per_coin = sl_price - price

    if risk_per_coin <= 0:
        logger.debug("[DENS] non-positive risk")
        return

    equity = get_wallet_balance()
    risk_usdt = min(equity * RISK_PCT, MAX_RISK_USDT)
    raw_qty = risk_usdt / risk_per_coin
    qty = math.floor(raw_qty / filters.qty_step) * filters.qty_step
    if qty < filters.min_qty:
        logger.debug("[DENS] qty below min: %.8f < %.8f", qty, filters.min_qty)
        return

    if entry_side == "long":
        tp_price = price + TP_R_MULT * (price - sl_price)
    else:
        tp_price = price - TP_R_MULT * (sl_price - price)

    try:
        res = place_market_order(entry_side, qty, stop_loss=sl_price, take_profit=tp_price, reduce_only=False)
        oid = res["result"]["orderId"]
        tg_send(
            f"🧱 <b>DENSITY</b> {SYMBOL}\n"
            f"Side: <b>{entry_side.upper()}</b> | Qty: <code>{qty}</code>\n"
            f"Price: <code>{price:.2f}</code> | SL: <code>{sl_price:.2f}</code> | TP: <code>{tp_price:.2f}</code>\n"
            f"trend5={trend_m5} buy_ratio={buy_ratio:.2f} | bid(sz={dens_bid_sz:.2f}@{dens_bid_px:.2f}) ask(sz={dens_ask_sz:.2f}@{dens_ask_px:.2f})"
        )
        logger.info("[DENS][ORDER] %s qty=%.6f @%.2f SL=%.2f TP=%.2f id=%s",
                    entry_side, qty, price, sl_price, tp_price, oid)
    except Exception as e:
        logger.error("[DENS][ORDER] failed: %s", e)
        tg_send(f"❌ <b>DENSITY order failed</b> {SYMBOL}\n<code>{e}</code>")

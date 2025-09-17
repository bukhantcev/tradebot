import logging
from typing import Dict, Any, List
from features import compute_indicators

log = logging.getLogger("strategy")

def decide(m1_candles: List[dict], m5_candles: List[dict], params: Dict[str, Any]) -> Dict[str, Any]:
    ema_slow = params["indicators"]["ema_slow"]
    required_m1 = max(50, ema_slow) + 1
    required_m5 = ema_slow + 1

    # Sticky/anti-churn params (defaults if not provided)
    ind = params.get("indicators", {})
    cross_confirm_bars = int(ind.get("cross_confirm_bars", 1))  # how many closed bars EMA relation must persist
    # Softer entry gates by default
    rsi_enter_long = float(ind.get("rsi_enter_long", 55.0))
    rsi_enter_short = float(ind.get("rsi_enter_short", 45.0))
    # optional exits (not used for entries, but kept for future logic)
    rsi_exit_long = float(ind.get("rsi_exit_long", 60.0))
    rsi_exit_short = float(ind.get("rsi_exit_short", 40.0))

    if len(m1_candles) < required_m1:
        log.warning(
            f"[WARMUP] not enough M1 candles: have={len(m1_candles)} need>={required_m1} (ema_slow={ema_slow})"
        )
        return {"decision": "hold", "reason": "warmup_m1"}

    # M5 warmup policy: allow fallback to M1 trend while M5 history is short
    allow_m5_warmup = params.get("allow_m5_warmup", True)
    require_m5_confirm = bool(params.get("require_m5_confirm", True)) == False
    m5_confirm = True
    if len(m5_candles) < required_m5:
        if not allow_m5_warmup:
            log.warning(
                f"[WARMUP] not enough M5 candles: have={len(m5_candles)} need>={required_m5} (ema_slow={ema_slow}); holding"
            )
            return {"decision": "hold", "reason": "warmup_m5"}
        else:
            log.warning(
                f"[WARMUP] not enough M5 candles: have={len(m5_candles)} need>={required_m5}. Using M1 trend as temporary confirmation (allow_m5_warmup=True)"
            )
            m5_confirm = False

    feats = compute_indicators(m1_candles, params)
    ema_fast_1 = feats["ema_fast"]; ema_slow_1 = feats["ema_slow"]
    rsi_series = feats["rsi"]; atr_series = feats["atr"]
    ef = ema_fast_1[-1]; es = ema_slow_1[-1]
    rsi = rsi_series[-1]; atr = atr_series[-1]
    close = m1_candles[-1]["close"]

    virt_tp = params["indicators"]["virtual_tp_atr"] * atr
    virt_sl = params["indicators"]["virtual_sl_atr"] * atr
    tp = virt_tp * params["indicators"]["tp_widen_mult"]
    sl = virt_sl * params["indicators"]["sl_widen_mult"]

    if m5_confirm:
        f5 = compute_indicators(m5_candles, params)
        ema_fast_5 = f5["ema_fast"]; ema_slow_5 = f5["ema_slow"]
        ema5_f = ema_fast_5[-1]
        ema5_s = ema_slow_5[-1]
    else:
        # During warmup: mirror M1 trend for confirmation
        ema_fast_5 = [ef]
        ema_slow_5 = [es]
        ema5_f = ef
        ema5_s = es

    # EMA cross must persist for cross_confirm_bars closed bars
    def _persist_up(ema_f, ema_s, bars: int) -> bool:
        if bars <= 1:
            return ema_f[-1] > ema_s[-1]
        if len(ema_f) < bars or len(ema_s) < bars:
            return False
        return all(ema_f[-i] > ema_s[-i] for i in range(1, bars + 1))

    def _persist_down(ema_f, ema_s, bars: int) -> bool:
        if bars <= 1:
            return ema_f[-1] < ema_s[-1]
        if len(ema_f) < bars or len(ema_s) < bars:
            return False
        return all(ema_f[-i] < ema_s[-i] for i in range(1, bars + 1))

    up_1 = _persist_up(ema_fast_1, ema_slow_1, cross_confirm_bars)
    down_1 = _persist_down(ema_fast_1, ema_slow_1, cross_confirm_bars)
    up_5 = _persist_up(ema_fast_5, ema_slow_5, cross_confirm_bars)
    down_5 = _persist_down(ema_fast_5, ema_slow_5, cross_confirm_bars)

    # --- SOFT EXIT SIGNALS ---
    # Optional context about current position can be passed in params["position_ctx"], e.g.:
    # {"side": "long|short", "entry_price": 12345.6, "high_since_entry": 12400.0, "low_since_entry": 12200.0}
    pos_ctx = params.get("position_ctx", {}) or {}
    pos_side = (pos_ctx.get("side") or "").lower()
    entry_price = pos_ctx.get("entry_price")
    high_se = pos_ctx.get("high_since_entry")
    low_se = pos_ctx.get("low_since_entry")
    atr_trail_k = float(ind.get("atr_trail_k", 1.5))

    # ATR trail checks only if we have context
    trail_hit_long = False
    trail_hit_short = False
    if isinstance(entry_price, (int, float)) and isinstance(atr, (int, float)):
        if isinstance(high_se, (int, float)):
            trail_long = high_se - atr_trail_k * atr
            trail_hit_long = close < trail_long
        if isinstance(low_se, (int, float)):
            trail_short = low_se + atr_trail_k * atr
            trail_hit_short = close > trail_short

    # EMA cross persistence + RSI exit hysteresis
    exit_long_cond = (down_1 and down_5) or (rsi >= rsi_exit_long) or trail_hit_long
    exit_short_cond = (up_1 and up_5) or (rsi <= rsi_exit_short) or trail_hit_short

    if pos_side == "long" and exit_long_cond:
        reason = []
        if (down_1 and down_5):
            reason.append(f"ema_down({cross_confirm_bars}b)")
        if rsi >= rsi_exit_long:
            reason.append("rsi>=exit_long")
        if trail_hit_long:
            reason.append(f"atr_trail_k={atr_trail_k}")
        d = {"decision": "exit_long", "reason": "+".join(reason) or "exit_long"}
        log.info(f"[SIGNAL] EXIT_LONG price={close:.2f} rsi={rsi:.2f} ema1={ef:.2f}/{es:.2f} ema5={ema5_f:.2f}/{ema5_s:.2f} reason={d['reason']}")
        log.debug(f"[DECIDE] {d}")
        return d

    if pos_side == "short" and exit_short_cond:
        reason = []
        if (up_1 and up_5):
            reason.append(f"ema_up({cross_confirm_bars}b)")
        if rsi <= rsi_exit_short:
            reason.append("rsi<=exit_short")
        if trail_hit_short:
            reason.append(f"atr_trail_k={atr_trail_k}")
        d = {"decision": "exit_short", "reason": "+".join(reason) or "exit_short"}
        log.info(f"[SIGNAL] EXIT_SHORT price={close:.2f} rsi={rsi:.2f} ema1={ef:.2f}/{es:.2f} ema5={ema5_f:.2f}/{ema5_s:.2f} reason={d['reason']}")
        log.debug(f"[DECIDE] {d}")
        return d

    # Entry conditions with optional M5 confirmation
    long_ok = up_1 and ((up_5) or (not require_m5_confirm)) and (rsi <= rsi_enter_long)
    short_ok = down_1 and ((down_5) or (not require_m5_confirm)) and (rsi >= rsi_enter_short)

    if long_ok:
        d = {"decision": "long", "tp": close + tp, "sl": close - sl,
             "virt_tp": virt_tp, "virt_sl": virt_sl,
             "reason": f"ema_up({cross_confirm_bars}b)+rsi<=enter_long" + ("+m5" if (m5_confirm and require_m5_confirm) else "+m5_relaxed")}
        log.info(f"[SIGNAL] ENTER_LONG price={close:.2f} tp={d['tp']:.2f} sl={d['sl']:.2f} rsi={rsi:.2f} ema1={ef:.2f}>{es:.2f} ema5={ema5_f:.2f}>{ema5_s:.2f} virt_tp={d['virt_tp']:.2f} virt_sl={d['virt_sl']:.2f} require_m5_confirm={require_m5_confirm} reason={d['reason']}")
        log.debug(f"[DECIDE] {d}")
        return d

    if short_ok:
        d = {"decision": "short", "tp": close - tp, "sl": close + sl,
             "virt_tp": virt_tp, "virt_sl": virt_sl,
             "reason": f"ema_down({cross_confirm_bars}b)+rsi>=enter_short" + ("+m5" if (m5_confirm and require_m5_confirm) else "+m5_relaxed")}
        log.info(f"[SIGNAL] ENTER_SHORT price={close:.2f} tp={d['tp']:.2f} sl={d['sl']:.2f} rsi={rsi:.2f} ema1={ef:.2f}<{es:.2f} ema5={ema5_f:.2f}<{ema5_s:.2f} virt_tp={d['virt_tp']:.2f} virt_sl={d['virt_sl']:.2f} require_m5_confirm={require_m5_confirm} reason={d['reason']}")
        log.debug(f"[DECIDE] {d}")
        return d

    d = {"decision": "hold", "reason": "filters_fail"}
    hold_details = (f"flags: up1={up_1} down1={down_1} up5={up_5} down5={down_5} "
                    f"rsi={rsi:.2f}<=L{rsi_enter_long}? {rsi<=rsi_enter_long} "
                    f">=S{rsi_enter_short}? {rsi>=rsi_enter_short} "
                    f"m5_confirm={m5_confirm} require_m5_confirm={require_m5_confirm} "
                    f"long_ok={long_ok if 'long_ok' in locals() else 'n/a'} short_ok={short_ok if 'short_ok' in locals() else 'n/a'}")
    log.info(f"[SIGNAL] HOLD price={close:.2f} rsi={rsi:.2f} ema1={ef:.2f}/{es:.2f} ema5={ema5_f:.2f}/{ema5_s:.2f} reason={d['reason']} {hold_details}")
    log.debug(f"[DECIDE] {d}")
    return d
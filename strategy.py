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
    rsi_enter_long = float(ind.get("rsi_enter_long", 40.0))
    rsi_enter_short = float(ind.get("rsi_enter_short", 60.0))
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

    if up_1 and up_5 and rsi <= rsi_enter_long:
        d = {"decision": "long", "tp": close + tp, "sl": close - sl,
             "virt_tp": virt_tp, "virt_sl": virt_sl,
             "reason": f"ema_up({cross_confirm_bars}b)+rsi<=enter_long" + ("+m5" if m5_confirm else "+warmup")}
        log.info(f"[SIGNAL] ENTER_LONG price={close:.2f} tp={d['tp']:.2f} sl={d['sl']:.2f} rsi={rsi:.2f} ema1={ef:.2f}>{es:.2f} ema5={ema5_f:.2f}>{ema5_s:.2f} virt_tp={d['virt_tp']:.2f} virt_sl={d['virt_sl']:.2f} reason={d['reason']}")
        log.debug(f"[DECIDE] {d}")
        return d
    if down_1 and down_5 and rsi >= rsi_enter_short:
        d = {"decision": "short", "tp": close - tp, "sl": close + sl,
             "virt_tp": virt_tp, "virt_sl": virt_sl,
             "reason": f"ema_down({cross_confirm_bars}b)+rsi>=enter_short" + ("+m5" if m5_confirm else "+warmup")}
        log.info(f"[SIGNAL] ENTER_SHORT price={close:.2f} tp={d['tp']:.2f} sl={d['sl']:.2f} rsi={rsi:.2f} ema1={ef:.2f}<{es:.2f} ema5={ema5_f:.2f}<{ema5_s:.2f} virt_tp={d['virt_tp']:.2f} virt_sl={d['virt_sl']:.2f} reason={d['reason']}")
        log.debug(f"[DECIDE] {d}")
        return d

    d = {"decision": "hold", "reason": "filters_fail"}
    log.info(f"[SIGNAL] HOLD price={close:.2f} rsi={rsi:.2f} ema1={ef:.2f}/{es:.2f} ema5={ema5_f:.2f}/{ema5_s:.2f} reason={d['reason']}")
    log.debug(f"[DECIDE] {d}")
    return d
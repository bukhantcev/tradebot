import logging
from typing import Dict, Any, List
from features import compute_indicators

log = logging.getLogger("strategy")

def decide(m1_candles: List[dict], m5_candles: List[dict], params: Dict[str, Any]) -> Dict[str, Any]:
    ema_slow = params["indicators"]["ema_slow"]
    required_m1 = max(50, ema_slow) + 1
    required_m5 = ema_slow + 1

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
    ef = feats["ema_fast"][-1]; es = feats["ema_slow"][-1]
    rsi = feats["rsi"][-1]; atr = feats["atr"][-1]
    close = m1_candles[-1]["close"]

    virt_tp = params["indicators"]["virtual_tp_atr"] * atr
    virt_sl = params["indicators"]["virtual_sl_atr"] * atr
    tp = virt_tp * params["indicators"]["tp_widen_mult"]
    sl = virt_sl * params["indicators"]["sl_widen_mult"]

    if m5_confirm:
        f5 = compute_indicators(m5_candles, params)
        ema5_f = f5["ema_fast"][ -1]
        ema5_s = f5["ema_slow"][ -1]
    else:
        # During warmup: mirror M1 trend for confirmation
        ema5_f = ef
        ema5_s = es

    if ef > es and ema5_f > ema5_s and rsi < params["indicators"]["rsi_overbought"]:
        d = {"decision": "long", "tp": close + tp, "sl": close - sl,
             "virt_tp": virt_tp, "virt_sl": virt_sl, "reason": "ema_up+rsi_ok" + ("+m5" if m5_confirm else "+warmup")}
        log.info(f"[SIGNAL] ENTER_LONG price={close:.2f} tp={d['tp']:.2f} sl={d['sl']:.2f} rsi={rsi:.2f} ema1={ef:.2f}>{es:.2f} ema5={ema5_f:.2f}>{ema5_s:.2f} virt_tp={d['virt_tp']:.2f} virt_sl={d['virt_sl']:.2f} reason={d['reason']}")
        log.debug(f"[DECIDE] {d}")
        return d
    if ef < es and ema5_f < ema5_s and rsi > params["indicators"]["rsi_oversold"]:
        d = {"decision": "short", "tp": close - tp, "sl": close + sl,
             "virt_tp": virt_tp, "virt_sl": virt_sl, "reason": "ema_down+rsi_ok" + ("+m5" if m5_confirm else "+warmup")}
        log.info(f"[SIGNAL] ENTER_SHORT price={close:.2f} tp={d['tp']:.2f} sl={d['sl']:.2f} rsi={rsi:.2f} ema1={ef:.2f}<{es:.2f} ema5={ema5_f:.2f}<{ema5_s:.2f} virt_tp={d['virt_tp']:.2f} virt_sl={d['virt_sl']:.2f} reason={d['reason']}")
        log.debug(f"[DECIDE] {d}")
        return d

    d = {"decision": "hold", "reason": "filters_fail"}
    log.info(f"[SIGNAL] HOLD price={close:.2f} rsi={rsi:.2f} ema1={ef:.2f}/{es:.2f} ema5={ema5_f:.2f}/{ema5_s:.2f} reason={d['reason']}")
    log.debug(f"[DECIDE] {d}")
    return d
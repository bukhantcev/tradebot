import logging
from typing import Dict, Any, Optional
from core.strategies.base import StrategyBase

log = logging.getLogger("STRAT")

class Momentum(StrategyBase):
    name = "Momentum"

    async def run_once(self, snapshot: Dict[str, Any], ex) -> str:
        return "Momentum ready."

    async def on_tick(self, tick: Dict[str, Any], *, ctx: Dict[str, Any]) -> Optional[str]:
        pm = ctx.get("pm")
        if tick.get("type") != "kline":
            return None

        d = tick.get("data") or {}
        bar_ts = int(d.get("start") or d.get("timestamp") or 0)
        if not self.should_process_bar(bar_ts):
            return None

        o = float(d.get("open") or d.get("o") or 0)
        h = float(d.get("high") or d.get("h") or 0)
        l = float(d.get("low") or d.get("l") or 0)
        c = float(d.get("close") or d.get("c") or 0)

        rng = max(h - l, 1e-9)
        body = abs(c - o)

        # thresholds
        dbg = bool(self.p.get("debug_extreme", False))
        min_body_frac = float(self.p.get("min_body_frac", 0.6))
        min_rng_frac = float(self.p.get("min_rng_frac", 0.0002))
        min_rng_abs  = float(self.p.get("min_rng_abs", 5.0))
        dyn_rng_thr = max(c, o) * min_rng_frac
        rng_ok = (rng >= min_rng_abs) or (rng >= dyn_rng_thr)
        body_ok = (body >= min_body_frac * rng) and rng_ok

        if dbg:
            rng_ok = True
            body_ok = True
            log.warning("[Momentum][extreme] rng/body bypass rng=%.4f body=%.4f", rng, body)

        side_now = getattr(getattr(pm, "state", None), "side", "FLAT")
        log.info("[Momentum][diag] o=%.2f c=%.2f h=%.2f l=%.2f rng=%.2f body=%.2f | rng_ok=%s body_ok=%s side=%s",
                 o, c, h, l, rng, body, rng_ok, body_ok, side_now)

        if side_now == "FLAT" and body_ok:
            side = "LONG" if c >= o else "SHORT"
            log.info("[Momentum] ENTER %s", side)
            return f"enter_{side.lower()}"

        # выходы стратегией запрещены — закрываемся только SL/TP
        return None
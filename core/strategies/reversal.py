import logging
from typing import Dict, Any, Optional
from core.strategies.base import StrategyBase

log = logging.getLogger("STRAT")

class Reversal(StrategyBase):
    name = "Reversal"

    async def run_once(self, snapshot: Dict[str, Any], ex) -> str:
        return "Reversal ready."

    async def on_tick(self, tick: Dict[str, Any], *, ctx: Dict[str, Any]) -> Optional[str]:
        pm = ctx.get("pm")
        if tick.get("type") != "kline":
            return None
        d = tick.get("data") or {}
        bar_ts = int(d.get("start") or 0)
        if not self.should_process_bar(bar_ts):
            return None

        o = float(d.get("open") or 0)
        h = float(d.get("high") or 0)
        l = float(d.get("low") or 0)
        c = float(d.get("close") or 0)

        rng = max(h - l, 1e-9)
        body = abs(c - o)
        upper_wick = max(0.0, h - max(c, o))
        lower_wick = max(0.0, min(c, o) - l)

        # Агрессивная логика: длинный пин-бар разворота
        if body < 0.4 * rng and upper_wick > 0.5 * rng:
            log.info("[Reversal] ENTER SHORT (bear pin)")
            return "enter_short"
        if body < 0.4 * rng and lower_wick > 0.5 * rng:
            log.info("[Reversal] ENTER LONG (bull pin)")
            return "enter_long"

        return None
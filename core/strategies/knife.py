import logging
from typing import Dict, Any, Optional
from core.strategies.base import StrategyBase

log = logging.getLogger("STRAT")

class Knife(StrategyBase):
    name = "Knife"

    async def run_once(self, snapshot: Dict[str, Any], ex) -> str:
        return "Knife ready."

    async def on_tick(self, tick: Dict[str, Any], *, ctx: Dict[str, Any]) -> Optional[str]:
        if tick.get("type") != "kline":
            return None
        d = tick.get("data") or {}
        bar_ts = int(d.get("start") or 0)
        if not self.should_process_bar(bar_ts):
            return None

        # Импульсный «нож»: если диапазон бара >> среднего (храним в stats)
        stats = ctx.get("stats", {})
        atr = float(stats.get("atr", 1.0))
        o = float(d.get("open") or 0)
        h = float(d.get("high") or 0)
        l = float(d.get("low") or 0)
        c = float(d.get("close") or 0)

        rng = max(h - l, 1e-9)
        if rng > 2.0 * atr:
            side = "LONG" if c >= o else "SHORT"
            log.info("[Knife] ENTER %s (rng>2*ATR)", side)
            return f"enter_{side.lower()}"

        return None
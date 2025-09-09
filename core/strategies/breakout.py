import logging
from typing import Dict, Any, Optional
from core.strategies.base import StrategyBase

log = logging.getLogger("STRAT")

class Breakout(StrategyBase):
    name = "Breakout"

    async def run_once(self, snapshot: Dict[str, Any], ex) -> str:
        return "Breakout ready."

    async def on_tick(self, tick: Dict[str, Any], *, ctx: Dict[str, Any]) -> Optional[str]:
        pm = ctx.get("pm")
        if tick.get("type") != "kline":
            return None
        d = tick.get("data") or {}
        bar_ts = int(d.get("start") or 0)
        if not self.should_process_bar(bar_ts):
            return None

        # Берём последние high/low из контекста (накоплено в хэндлере)
        stats = ctx.get("stats", {})
        hi = float(stats.get("recent_high", 0))
        lo = float(stats.get("recent_low", 0))

        c = float(d.get("close") or 0)
        o = float(d.get("open") or 0)

        if hi and c > hi:
            log.info("[Breakout] ENTER LONG (close>recent_high)")
            return "enter_long"
        if lo and c < lo:
            log.info("[Breakout] ENTER SHORT (close<recent_low)")
            return "enter_short"

        return None
import logging
from typing import Dict, Any, Optional
from core.strategies.base import StrategyBase

log = logging.getLogger("STRAT")

class Density(StrategyBase):
    name = "Orderbook Density"

    async def run_once(self, snapshot: Dict[str, Any], ex) -> str:
        return "Density ready."

    async def on_tick(self, tick: Dict[str, Any], *, ctx: Dict[str, Any]) -> Optional[str]:
        if tick.get("type") != "kline":
            return None
        d = tick.get("data") or {}
        bar_ts = int(d.get("start") or 0)
        if not self.should_process_bar(bar_ts):
            return None

        ob = ctx.get("orderbook") or {}
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        if not bids or not asks:
            return None

        # Грубая агрессивная эвристика: если сумма top-5 бидов сильно > асков → long, и наоборот
        top = 5
        s_bids = sum(float(x[1]) for x in bids[:top])
        s_asks = sum(float(x[1]) for x in asks[:top])
        if s_bids > 1.5 * s_asks:
            log.info("[Density] ENTER LONG (bid density)")
            return "enter_long"
        if s_asks > 1.5 * s_bids:
            log.info("[Density] ENTER SHORT (ask density)")
            return "enter_short"
        return None
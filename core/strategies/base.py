import logging
from typing import Dict, Any, Optional

log = logging.getLogger("STRAT")

class StrategyBase:
    name = "Base"

    def __init__(self, params: dict):
        self.p = params or {}
        self.last_bar_ts = None

    async def run_once(self, snapshot: Dict[str, Any], ex) -> str:
        return "ready"

    def should_process_bar(self, bar_ts: int | None) -> bool:
        if not bar_ts:
            return True
        if self.last_bar_ts == bar_ts:
            return False
        self.last_bar_ts = bar_ts
        return True

    async def on_tick(self, tick: Dict[str, Any], *, ctx: Dict[str, Any]) -> Optional[str]:
        raise NotImplementedError
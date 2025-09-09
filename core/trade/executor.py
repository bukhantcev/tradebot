import logging
from core.bybit.client import BybitClient

log = logging.getLogger("EXEC")

class Executor:
    def __init__(self, bybit: BybitClient, category: str, symbol: str):
        self.bybit = bybit
        self.category = category
        self.symbol = symbol
        self._filters = None

    async def get_filters(self):
        if not self._filters:
            self._filters = await self.bybit.get_instrument_filters(self.category, self.symbol)
        return self._filters

    async def market_buy(self, qty: float):
        try:
            r = await self.bybit.create_order(self.category, self.symbol, side="Buy", qty=qty)
            log.info("[EXEC] Market Buy qty=%.6f OK", qty)
            return True
        except Exception as e:
            log.error("[EXEC] Market Buy error: %s", e)
            return False

    async def market_sell(self, qty: float):
        try:
            r = await self.bybit.create_order(self.category, self.symbol, side="Sell", qty=qty)
            log.info("[EXEC] Market Sell qty=%.6f OK", qty)
            return True
        except Exception as e:
            log.error("[EXEC] Market Sell error: %s", e)
            return False
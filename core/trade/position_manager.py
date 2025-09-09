import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("PM")

@dataclass
class PositionState:
    side: str = "FLAT"   # LONG/SHORT/FLAT
    size: float = 0.0
    entry_price: float = 0.0
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None

class PositionManager:
    def __init__(self, symbol: str, base_order_usdt: float, max_loss_usdt: float, ex, category: str):
        self.symbol = symbol
        self.base_usdt = base_order_usdt
        self.max_loss = max_loss_usdt
        self.ex = ex
        self.category = category
        self.state = PositionState()
        self.loss_streak = 0

    def is_open(self) -> bool:
        return self.state.side in ("LONG", "SHORT")

    def set_sl_tp(self, sl: float | None = None, tp: float | None = None):
        self.state.sl_price = sl
        self.state.tp_price = tp
        log.info("[PM] SL/TP set: sl=%s tp=%s", sl, tp)

    async def _qty_from_usdt(self, price: float) -> float:
        filters = await self.ex.get_filters()
        qty = self.base_usdt / max(price, 1e-9)
        # Step & min
        step = filters["qtyStep"]
        minq = filters["minOrderQty"]
        qty = (int(qty / step)) * step
        if qty < minq:
            qty = minq
        log.info("[PM] qty calc: usdt=%.2f price=%.2f -> qty=%.6f (min=%.6f step=%.6f)", self.base_usdt, price, qty, minq, step)
        return float(qty)

    async def open_long(self, last_price: float):
        if self.is_open():
            log.info("[PM] skip open_long — already open: %s", self.state)
            return
        qty = await self._qty_from_usdt(last_price)
        res = await self.ex.market_buy(qty)
        if res:
            self.state.side = "LONG"
            self.state.size = qty
            self.state.entry_price = last_price
            sl = last_price - self.max_loss / qty
            tp = last_price + 2 * (self.max_loss / qty)
            self.set_sl_tp(sl, tp)
            log.info("[PM] LONG opened @ %.2f qty=%.6f sl=%.2f tp=%.2f", last_price, qty, sl, tp)

    async def open_short(self, last_price: float):
        if self.is_open():
            log.info("[PM] skip open_short — already open: %s", self.state)
            return
        qty = await self._qty_from_usdt(last_price)
        res = await self.ex.market_sell(qty)
        if res:
            self.state.side = "SHORT"
            self.state.size = qty
            self.state.entry_price = last_price
            sl = last_price + self.max_loss / qty
            tp = last_price - 2 * (self.max_loss / qty)
            self.set_sl_tp(sl, tp)
            log.info("[PM] SHORT opened @ %.2f qty=%.6f sl=%.2f tp=%.2f", last_price, qty, sl, tp)

    async def close_all(self, last_price: float | None = None):
        if not self.is_open():
            log.info("[PM] close_all: FLAT")
            return
        qty = self.state.size
        if self.state.side == "LONG":
            await self.ex.market_sell(qty)
        else:
            await self.ex.market_buy(qty)
        # Псевдо-PnL
        pnl = 0.0
        if last_price:
            if self.state.side == "LONG":
                pnl = (last_price - self.state.entry_price) * qty
            else:
                pnl = (self.state.entry_price - last_price) * qty
        self.loss_streak = (self.loss_streak + 1) if pnl < 0 else 0
        log.info("[PM] Closed %s qty=%.6f @%.2f | PnL≈%.2f | loss_streak=%d",
                 self.state.side, qty, last_price or -1, pnl, self.loss_streak)
        self.state = PositionState()

    async def check_sl_tp(self, last_price: float):
        """Закрываем позицию ТОЛЬКО по SL/TP."""
        if not self.is_open():
            return False
        if self.state.side == "LONG":
            if self.state.sl_price and last_price <= self.state.sl_price:
                log.info("[PM] LONG stop triggered @ %.2f (sl=%.2f)", last_price, self.state.sl_price)
                await self.close_all(last_price)
                return True
            if self.state.tp_price and last_price >= self.state.tp_price:
                log.info("[PM] LONG take-profit triggered @ %.2f (tp=%.2f)", last_price, self.state.tp_price)
                await self.close_all(last_price)
                return True
        else:  # SHORT
            if self.state.sl_price and last_price >= self.state.sl_price:
                log.info("[PM] SHORT stop triggered @ %.2f (sl=%.2f)", last_price, self.state.sl_price)
                await self.close_all(last_price)
                return True
            if self.state.tp_price and last_price <= self.state.tp_price:
                log.info("[PM] SHORT take-profit triggered @ %.2f (tp=%.2f)", last_price, self.state.tp_price)
                await self.close_all(last_price)
                return True
        return False
from typing import Dict, Any, Optional
from decimal import Decimal, ROUND_DOWN
from bybit_client import BybitClient
from config import CATEGORY

class Trader:
    def __init__(self, client: BybitClient, symbol: str, tick_size: float, qty_step: float, notifier: Optional[Any] = None):
        self.client = client
        self.symbol = symbol
        self.tick_size = Decimal(str(tick_size))
        self.qty_step = Decimal(str(qty_step))
        self.positionIdx = 0  # one-way
        self.notifier = notifier

    def _fmt_price(self, p: float) -> str:
        d = Decimal(str(p))
        q = d.quantize(self._price_quant(), rounding=ROUND_DOWN)
        return format(q, 'f')

    def _price_quant(self) -> Decimal:
        s = str(self.tick_size)
        return Decimal(s)

    def _fmt_qty(self, q: float) -> str:
        d = Decimal(str(q))
        qd = d.quantize(Decimal(str(self.qty_step)), rounding=ROUND_DOWN)
        return format(qd, 'f')

    async def ensure_leverage(self, lev: int):
        await self.client.set_leverage(CATEGORY, self.symbol, str(lev), str(lev))

    async def get_position(self) -> Dict[str, Any]:
        r = await self.client.position_list(CATEGORY, self.symbol)
        return (r.get("result", {}).get("list") or [{}])[0]

    async def open_market(self, side: str, qty: float, tp_price: Optional[float], sl_price: Optional[float]) -> Dict[str, Any]:
        q = self._fmt_qty(qty)
        tp = self._fmt_price(tp_price) if tp_price else None
        sl = self._fmt_price(sl_price) if sl_price else None
        o = await self.client.place_order(
            CATEGORY, self.symbol, side=side, orderType="Market", qty=q,
            timeInForce="IOC", positionIdx=self.positionIdx,
            tpslMode="Full", tpOrderType="Market", slOrderType="Market"
        )
        # После исполнения — проставим TP/SL на позицию
        if tp or sl:
            await self.client.trading_stop(CATEGORY, self.symbol, positionIdx=self.positionIdx,
                                           takeProfit=tp, stopLoss=sl,
                                           tpOrderType="Market", slOrderType="Market",
                                           tpslMode="Full", tpTriggerBy="LastPrice", slTriggerBy="LastPrice")
        if self.notifier:
            msg = f"🚀 Открыта позиция {side} {q} {self.symbol}\nTP={tp} SL={sl}"
            await self.notifier.notify(msg)
        return o

    async def close_all(self):
        # reduce-only market close (qty=0 с флагами закрытия)
        await self.client.place_order(CATEGORY, self.symbol, side="Sell", orderType="Market", qty="0",
                                      reduceOnly=True, closeOnTrigger=True, positionIdx=self.positionIdx)
        await self.client.place_order(CATEGORY, self.symbol, side="Buy", orderType="Market", qty="0",
                                      reduceOnly=True, closeOnTrigger=True, positionIdx=self.positionIdx)
        await self.client.cancel_all(CATEGORY, self.symbol)
        if self.notifier:
            await self.notifier.notify(f"🔻 Все позиции по {self.symbol} закрыты")
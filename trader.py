import logging
from typing import Dict, Any, Optional
from decimal import Decimal, ROUND_DOWN
from bybit_client import BybitClient
from config import CATEGORY

class Trader:
    def __init__(self, client: BybitClient, symbol: str, tick_size: float, qty_step: float, notifier=None):
        self.client = client
        self.symbol = symbol
        self.tick_size = Decimal(str(tick_size))
        self.qty_step = Decimal(str(qty_step))
        self.positionIdx = 0  # one-way
        self.notifier = notifier
        self.log = logging.getLogger("trader")

    def _price_quant(self) -> Decimal:
        return Decimal(str(self.tick_size))

    def _fmt_price(self, p: float) -> str:
        d = Decimal(str(p))
        q = d.quantize(self._price_quant(), rounding=ROUND_DOWN)
        return format(q, 'f')

    def _fmt_qty(self, q: float) -> str:
        d = Decimal(str(q))
        qd = d.quantize(Decimal(str(self.qty_step)), rounding=ROUND_DOWN)
        return format(qd, 'f')

    async def ensure_leverage(self, lev: int):
        self.log.info(f"Set leverage {lev}")
        await self.client.set_leverage(CATEGORY, self.symbol, str(lev), str(lev))

    async def get_position(self) -> Dict[str, Any]:
        r = await self.client.position_list(CATEGORY, self.symbol)
        return (r.get("result", {}).get("list") or [{}])[0]

    async def _guess_entry_fill_price_from_position(self) -> Optional[float]:
        try:
            p = await self.get_position()
            ap = p.get("avgPrice") or p.get("entryPrice")
            return float(ap) if ap is not None else None
        except Exception:
            return None

    def _extract_fill_price_from_order_resp(self, o: Dict[str, Any]) -> Optional[float]:
        try:
            r = o.get("result", {})
            ap = r.get("avgPrice") or r.get("price")
            return float(ap) if ap is not None else None
        except Exception:
            return None

    async def open_market(self, side: str, qty: float, tp_price: Optional[float], sl_price: Optional[float]) -> Dict[str, Any]:
        q = self._fmt_qty(qty)
        tp = self._fmt_price(tp_price) if tp_price else None
        sl = self._fmt_price(sl_price) if sl_price else None
        self.log.info(f"[ORDER_ATTEMPT] side={side} qty={q} tp={tp} sl={sl}")
        self.log.info(f"OPEN {side} qty={q} tp={tp} sl={sl}")
        body = {
            "category": CATEGORY, "symbol": self.symbol, "side": side,
            "orderType": "Market", "qty": q, "timeInForce": "IOC",
            "positionIdx": self.positionIdx, "tpslMode": "Full",
            "tpOrderType": "Market", "slOrderType": "Market"
        }
        o = await self.client.place_order(body)
        if o.get("retCode") == 0:
            self.log.info(f"[ORDER_OK] side={side} qty={q} orderId={o.get('result',{}).get('orderId')} resp={o}")
        else:
            self.log.error(f"[ORDER_FAIL] side={side} qty={q} code={o.get('retCode')} msg={o.get('retMsg')} resp={o}")
        self.log.debug(f"place_order resp={o}")

        # После исполнения — проставим TP/SL на позицию
        if tp or sl:
            ts_body = {"category": CATEGORY, "symbol": self.symbol, "positionIdx": self.positionIdx,
                       "tpslMode": "Full", "tpOrderType": "Market", "slOrderType": "Market",
                       "tpTriggerBy": "LastPrice", "slTriggerBy": "LastPrice"}
            if tp: ts_body["takeProfit"] = tp
            if sl: ts_body["stopLoss"] = sl
            rts = await self.client.trading_stop(ts_body)
            self.log.debug(f"trading_stop resp={rts}")

        # Try to estimate actual fill price from position snapshot after order
        fill_price = await self._guess_entry_fill_price_from_position()

        if self.notifier:
            await self.notifier.notify(f"🚀 Открыта позиция {side} {q} {self.symbol}\nTP={tp} SL={sl}")
        o["fill_price"] = fill_price
        return o

    async def close_market(self, side: str, qty: Optional[float] = None) -> Dict[str, Any]:
        """Reduce-only market close on given side (opposite to current position). If qty is None, closes full size."""
        if qty is None:
            pos = await self.get_position()
            size = pos.get("size") or pos.get("positionValue") or 0
            try:
                qty = float(size)
            except Exception:
                qty = 0.0
        q = self._fmt_qty(qty)
        self.log.info(f"[EXIT_ATTEMPT] side={side} qty={q} reduceOnly=True")
        body = {"category": CATEGORY, "symbol": self.symbol, "side": side,
                "orderType": "Market", "qty": q, "reduceOnly": True,
                "closeOnTrigger": True, "positionIdx": self.positionIdx}
        o = await self.client.place_order(body)
        if o.get("retCode") == 0:
            self.log.info(f"[EXIT_OK] side={side} qty={q} orderId={o.get('result',{}).get('orderId')} resp={o}")
        else:
            self.log.error(f"[EXIT_FAIL] side={side} qty={q} code={o.get('retCode')} msg={o.get('retMsg')} resp={o}")
        fp = self._extract_fill_price_from_order_resp(o)
        o["fill_price"] = fp
        if self.notifier:
            await self.notifier.notify(f"✅ Рынком закрыто {q} {self.symbol} (side={side})")
        return o

    async def close_all(self):
        self.log.info("CLOSE ALL via reduceOnly")
        # reduce-only market close (qty=0 с флагами закрытия)
        await self.client.place_order({"category": CATEGORY, "symbol": self.symbol, "side": "Sell",
                                       "orderType": "Market", "qty": "0", "reduceOnly": True,
                                       "closeOnTrigger": True, "positionIdx": self.positionIdx})
        await self.client.place_order({"category": CATEGORY, "symbol": self.symbol, "side": "Buy",
                                       "orderType": "Market", "qty": "0", "reduceOnly": True,
                                       "closeOnTrigger": True, "positionIdx": self.positionIdx})
        await self.client.cancel_all(CATEGORY, self.symbol)
        if self.notifier:
            await self.notifier.notify(f"🔻 Все позиции по {self.symbol} закрыты")
import logging
from typing import Optional

from core.bybit.client import BybitClient

log = logging.getLogger("EXEC")


class Executor:
    def __init__(self, bybit: BybitClient, category: str, symbol: str):
        self.bybit = bybit
        self.category = category
        self.symbol = symbol
        self._filters = None

    async def get_filters(self):
        """Fetch and cache instrument filters (qtyStep, minOrderQty, tickSize, ...)."""
        if not self._filters:
            self._filters = await self.bybit.get_instrument_filters(self.category, self.symbol)
        return self._filters

    # =====================
    # Market orders
    # =====================
    async def market_buy(self, qty: float) -> bool:
        try:
            r = await self.bybit.create_order(
                self.category,
                self.symbol,
                side="Buy",
                qty=qty,
            )
            log.info("[EXEC] Market Buy qty=%.6f OK | resp=%s", qty, r)
            return True
        except Exception as e:
            log.error("[EXEC] Market Buy error: %s", e)
            return False

    async def market_sell(self, qty: float) -> bool:
        try:
            r = await self.bybit.create_order(
                self.category,
                self.symbol,
                side="Sell",
                qty=qty,
            )
            log.info("[EXEC] Market Sell qty=%.6f OK | resp=%s", qty, r)
            return True
        except Exception as e:
            log.error("[EXEC] Market Sell error: %s", e)
            return False

    # =====================
    # Real exchange SL/TP via Bybit v5 `position/trading-stop`
    # =====================
    async def set_trading_stop(
        self,
        *,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        sl_trigger_by: str = "LastPrice",
        tp_trigger_by: str = "LastPrice",
        position_idx: int = 0,  # 0: one-way; 1/2: hedge long/short
    ) -> bool:
        """
        Set/replace real SL/TP on the exchange for current position.
        Bybit v5 endpoint: POST /v5/position/trading-stop
        Notes:
          - For one-way mode `positionIdx=0` is fine.
          - Pass only the fields you want to (re)set; others will be kept if omitted.
          - Use empty strings to clear values (see `cancel_trading_stop`).
        """
        payload = {
            "category": self.category,
            "symbol": self.symbol,
            "positionIdx": str(position_idx),
        }
        if stop_loss is not None:
            payload["stopLoss"] = f"{stop_loss}"
            payload["slTriggerBy"] = sl_trigger_by
        if take_profit is not None:
            payload["takeProfit"] = f"{take_profit}"
            payload["tpTriggerBy"] = tp_trigger_by

        try:
            # IMPORTANT: send as JSON BODY, not query params
            r = await self.bybit._signed("POST", "/v5/position/trading-stop", body=payload)
            rc = r.get("retCode")
            if rc == 0:
                log.info(
                    "[EXEC] set_trading_stop OK | SL=%s TP=%s | resp=%s",
                    payload.get("stopLoss"), payload.get("takeProfit"), r,
                )
                return True
            else:
                log.error("[EXEC] set_trading_stop FAIL rc=%s | payload=%s | resp=%s", rc, payload, r)
                return False
        except Exception as e:
            log.error("[EXEC] set_trading_stop error: %s | payload=%s", e, payload)
            return False

    async def set_stop_loss(
        self,
        *,
        stop_loss: float,
        sl_trigger_by: str = "LastPrice",
        position_idx: int = 0,
    ) -> bool:
        """
        Set ONLY real exchange Stop-Loss on current position.
        Uses Bybit v5: POST /v5/position/trading-stop with stopLoss only.
        """
        payload = {
            "category": self.category,
            "symbol": self.symbol,
            "positionIdx": str(position_idx),
            "stopLoss": f"{stop_loss}",
            "slTriggerBy": sl_trigger_by,
        }
        try:
            r = await self.bybit._signed("POST", "/v5/position/trading-stop", body=payload)
            rc = r.get("retCode")
            if rc == 0:
                log.info("[EXEC] set_stop_loss OK | SL=%s | resp=%s", payload["stopLoss"], r)
                return True
            else:
                log.error("[EXEC] set_stop_loss FAIL rc=%s | payload=%s | resp=%s", rc, payload, r)
                return False
        except Exception as e:
            log.error("[EXEC] set_stop_loss error: %s | payload=%s", e, payload)
            return False

    async def cancel_trading_stop(self, position_idx: int = 0) -> bool:
        """Clear SL/TP on exchange (keep position open)."""
        payload = {
            "category": self.category,
            "symbol": self.symbol,
            "positionIdx": str(position_idx),
            # Per Bybit v5, empty strings clear SL/TP
            "stopLoss": "",
            "takeProfit": "",
        }
        try:
            # IMPORTANT: send as JSON BODY, not query params
            r = await self.bybit._signed("POST", "/v5/position/trading-stop", body=payload)
            rc = r.get("retCode")
            if rc == 0:
                log.info("[EXEC] cancel_trading_stop OK | resp=%s", r)
                return True
            else:
                log.error("[EXEC] cancel_trading_stop FAIL rc=%s | payload=%s | resp=%s", rc, payload, r)
                return False
        except Exception as e:
            log.error("[EXEC] cancel_trading_stop error: %s | payload=%s", e, payload)
            return False
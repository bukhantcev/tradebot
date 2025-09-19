# trader_mod/account.py
import logging
from typing import Dict, Any, Optional
from bybit_client import BybitClient

log = logging.getLogger("TRADER")

class Account:
    def __init__(self, client: BybitClient, symbol: str, leverage: float, notifier=None):
        self.client = client
        self.symbol = symbol
        self.leverage = float(leverage)
        self.notifier = notifier
        self._filters: Optional[Dict[str, float]] = None
        self.equity: float = 0.0
        self.available: float = 0.0

    def ensure_filters(self) -> Dict[str, float]:
        if self._filters:
            return self._filters
        r = self.client.instruments_info(category="linear", symbol=self.symbol)
        it = r.get("result", {}).get("list", [{}])[0]
        tick = float(it.get("priceFilter", {}).get("tickSize", "0.1"))
        qty_step = float(it.get("lotSizeFilter", {}).get("qtyStep", "0.001"))
        min_qty = float(it.get("lotSizeFilter", {}).get("minOrderQty", "0.001"))
        mov = float(it.get("lotSizeFilter", {}).get("minOrderValue", "0") or 0.0)
        self._filters = {"tickSize": tick, "qtyStep": qty_step, "minQty": min_qty, "minNotional": mov}
        return self._filters

    def refresh_equity(self) -> float:
        try:
            r = self.client.wallet_balance(account_type="UNIFIED")
            lst = r["result"]["list"][0]
            usdt_total = float(lst["totalEquity"])
            usdt_avail = float(lst.get("totalAvailableBalance") or lst.get("availableBalance") or usdt_total)
            self.equity = usdt_total
            self.available = usdt_avail
            log.info(f"[BALANCE] equity={usdt_total:.2f} avail={usdt_avail:.2f} USDT")
            if self.notifier:
                try:
                    import asyncio
                    asyncio.create_task(self.notifier.notify(f"💰 Баланс: {usdt_total:.2f} USDT (доступно {usdt_avail:.2f})"))
                except Exception:
                    pass
            return usdt_total
        except Exception as e:
            log.error(f"[BALANCE][ERR] {e}")
            return 0.0

    def ensure_leverage(self):
        try:
            pl = self.client.position_list(self.symbol)
            cur_lev = float(pl.get("result", {}).get("list", [{}])[0].get("leverage") or 0.0)
            if abs(cur_lev - self.leverage) < 1e-9:
                return
        except Exception:
            pass
        r = self.client.set_leverage(self.symbol, self.leverage, self.leverage)
        rc = r.get("retCode")
        if rc in (0, 110043):
            log.info(f"[LEV] {self.leverage}x OK")
        else:
            log.warning(f"[LEV] rc={rc} msg={r.get('retMsg')}")
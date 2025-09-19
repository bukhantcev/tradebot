# trader_mod/account.py
import asyncio
import logging
from typing import Dict

log = logging.getLogger("TRADER")


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
                asyncio.create_task(self.notifier.notify(f"💰 Баланс: {usdt_total:.2f} USDT (доступно {usdt_avail:.2f})"))
            except Exception:
                pass
        return usdt_total
    except Exception as e:
        log.error(f"[BALANCE][ERR] {e}")
        return 0.0


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
    log.debug(f"[FILTERS] {self._filters}")
    return self._filters


def ensure_leverage(self):
    try:
        pl = self.client.position_list(self.symbol)
        cur_lev = float(pl.get("result", {}).get("list", [{}])[0].get("leverage") or 0.0)
        if abs(cur_lev - self.leverage) < 1e-9:
            log.debug(f"[LEV] already {cur_lev}x")
            return
    except Exception as e:
        log.debug(f"[LEV] read failed: {e}")
    r = self.client.set_leverage(self.symbol, self.leverage, self.leverage)
    rc = r.get("retCode")
    if rc in (0, 110043):
        log.info(f"[LEV] {self.leverage}x OK")
    else:
        log.warning(f"[LEV] rc={rc} msg={r.get('retMsg')}")


def _last_price(self) -> float:
    try:
        r = self.client._request("GET", "/v5/market/tickers", params={"category": "linear", "symbol": self.symbol})
        it = (r.get("result", {}) or {}).get("list", [])
        if it:
            self.__lp_fail = 0
            return float(it[0].get("lastPrice") or 0.0)
    except Exception:
        pass
    self.__lp_fail += 1
    if self.__lp_fail % 20 == 0:
        log.warning(f"[EXT][LP] lastPrice=0 (#{self.__lp_fail})")
    return 0.0


def _position_side_and_size(self) -> tuple[str | None, float]:
    try:
        pl = self.client.position_list(self.symbol)
        lst = pl.get("result", {}).get("list", [])
        for it in lst:
            size = float(it.get("size") or 0.0)
            if size > 0:
                return (it.get("side"), size)
    except Exception:
        pass
    return (None, 0.0)


def _opposite(self, side: str) -> str:
    return "Sell" if side == "Buy" else "Buy"
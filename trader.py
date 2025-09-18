import logging
import time
from typing import Optional, Dict, Any

from bybit_client import BybitClient
from config import SYMBOL, LEVERAGE, RISK_PCT

log = logging.getLogger("TRADER")

class Trader:
    def __init__(self, symbol: str = SYMBOL, risk_pct: float = RISK_PCT):
        self.client = BybitClient()
        self.symbol = symbol
        self.risk_pct = risk_pct
        self.position_open = False

    def _equity_usdt(self) -> float:
        data = self.client.wallet_balance(account_type="UNIFIED")
        try:
            coins = data["result"]["list"][0]["coin"]
            for c in coins:
                if c["coin"] == "USDT":
                    eq = float(c["equity"])
                    log.debug(f"[EQUITY] USDT={eq}")
                    return eq
        except Exception as e:
            log.error(f"[EQUITY][ERR] {e}")
        return 0.0

    def _filters(self) -> Dict[str, Any]:
        info = self.client.instruments_info(category="linear", symbol=self.symbol)
        it = info.get("result", {}).get("list", [])[0]
        pf = it["priceFilter"]; lf = it["lotSizeFilter"]
        out = {
            "tickSize": float(pf["tickSize"]),
            "qtyStep": float(lf["qtyStep"]),
            "minNotional": float(lf.get("minNotional", "0") or 0.0)
        }
        log.debug(f"[FILTERS] {out}")
        return out

    def ensure_leverage(self):
        log.debug(f"[LEV] ensure {LEVERAGE}x")
        self.client.set_leverage(self.symbol, LEVERAGE, LEVERAGE)

    def calc_qty(self, entry: float, sl: float, equity_usdt: float, qty_step: float, min_notional: float) -> float:
        risk_amount = max(equity_usdt * (self.risk_pct / 100.0), 1.0)
        stop_dist = abs(entry - sl)
        if stop_dist <= 0:
            return 0.0
        raw_qty = risk_amount / stop_dist
        min_qty = min_notional / max(entry, 1e-8)
        qty = max(raw_qty, min_qty)
        # округлим к шагу
        if qty_step > 0:
            qty = round(qty / qty_step) * qty_step
        log.debug(f"[QTY] eq={equity_usdt:.2f} risk%={self.risk_pct} risk_amt={risk_amount:.4f} stop={stop_dist:.4f} raw={raw_qty:.6f} min_qty={min_qty:.6f} -> qty={qty:.6f}")
        return max(qty, qty_step)

    def _confirm_open(self) -> bool:
        pl = self.client.position_list(self.symbol)
        try:
            sz = float(pl["result"]["list"][0]["size"])
            log.debug(f"[CONFIRM] size={sz}")
            return sz > 0
        except Exception as e:
            log.error(f"[CONFIRM][ERR] {e}")
            return False

    def place_trade(self, side: str, sl: float, tp: float) -> Optional[Dict[str, Any]]:
        self.ensure_leverage()
        filters = self._filters()
        eq = self._equity_usdt()
        # приблизим entry текущей ценой из tp/sl — достаточно для логов формулы
        entry_estimate = (tp + sl) / 2.0
        qty = self.calc_qty(entry=entry_estimate, sl=sl, equity_usdt=eq, qty_step=filters["qtyStep"], min_notional=filters["minNotional"])
        if qty <= 0:
            log.error("[TRADE] qty<=0 — отмена")
            return None

        log.info(f"[ENTER] side={side} qty={qty}")
        resp = self.client.place_order(self.symbol, side=side, qty=qty, order_type="Market")
        rc = resp.get("retCode")
        if rc != 0:
            log.error(f"[ORDER][FAIL] rc={rc} msg={resp.get('retMsg')}")
            return None

        time.sleep(0.5)
        if not self._confirm_open():
            log.error("[ORDER] не подтверждено /position/list")
            return None

        log.info(f"[TPSL] side={side} SL={sl:.2f} TP={tp:.2f}")
        ts = self.client.trading_stop(self.symbol, side=side, stop_loss=sl, take_profit=tp)
        if ts.get("retCode") != 0:
            log.warning(f"[TPSL][WARN] {ts}")

        self.position_open = True
        return {"order": resp, "qty": qty, "sl": sl, "tp": tp}

    def close_all(self):
        log.info("[CLOSE_ALL] try")
        pl = self.client.position_list(self.symbol)
        try:
            pos = pl["result"]["list"][0]
            size = float(pos["size"])
            if size == 0:
                log.info("[CLOSE_ALL] no position")
                return
            side = pos.get("side", "").lower()
            close_side = "Sell" if side == "buy" else "Buy"
            log.info(f"[CLOSE] size={size} side={side} -> {close_side}")
            self.client.place_order(self.symbol, side=close_side, qty=size, order_type="Market")
        except Exception as e:
            log.error(f"[CLOSE_ALL][ERR] {e}")
        self.position_open = False
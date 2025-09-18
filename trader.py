"""
Трейдер: расчёт размера позиции от риска, постановка ордеров, SL/TP, подтверждение через /position/list.
- risk_pct: % от equity (USDT)
- qty от ATR-стопа: risk_amount / (|entry - SL|) в квоте -> затем пересчёт в количество по mark price
"""
import logging
import time
from typing import Optional, Dict, Any

from bybit_client import BybitClient
from config import SYMBOL, LEVERAGE, RISK_PCT

logger = logging.getLogger("TRADER")

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
                    return float(c["equity"])
        except Exception:
            pass
        return 0.0

    def _filters(self) -> Dict[str, Any]:
        info = self.client.instruments_info(category="linear", symbol=self.symbol)
        it = info.get("result", {}).get("list", [])[0]
        pf = it["priceFilter"]; lf = it["lotSizeFilter"]
        return {
            "tickSize": float(pf["tickSize"]),
            "qtyStep": float(lf["qtyStep"]),
            "minNotional": float(lf.get("minNotional", "0") or 0.0)
        }

    @staticmethod
    def _round_step(x: float, step: float) -> float:
        if step <= 0: return x
        return round(x / step) * step

    def ensure_leverage(self):
        self.client.set_leverage(self.symbol, LEVERAGE, LEVERAGE)

    def calc_qty(self, entry: float, sl: float, equity_usdt: float, qty_step: float, min_notional: float) -> float:
        risk_amount = max(equity_usdt * (self.risk_pct / 100.0), 1.0)
        stop_dist = abs(entry - sl)  # in price
        if stop_dist <= 0:
            return 0.0
        # quote value per 1 contract ≈ entry (for USDT perp)
        qty = risk_amount / stop_dist
        qty = max(qty, min_notional / max(entry, 1e-8))
        qty = self._round_step(qty, qty_step)
        return max(qty, qty_step)

    def _confirm_open(self) -> bool:
        pl = self.client.position_list(self.symbol)
        try:
            sz = float(pl["result"]["list"][0]["size"])
            return sz > 0
        except Exception:
            return False

    def place_trade(self, side: str, sl: float, tp: float) -> Optional[Dict[str, Any]]:
        self.ensure_leverage()
        filters = self._filters()
        eq = self._equity_usdt()
        # entry = market price ~ assume last price via position list or order result (simplified)
        # Для простоты: используем tp/sl на базе текущего close — в проде лучше тянуть mark price
        entry_price = None
        qty = self.calc_qty(entry=tp if side=="Buy" else sl,  # приблизительно — безопаснее завышает стоп
                            sl=sl, equity_usdt=eq, qty_step=filters["qtyStep"], min_notional=filters["minNotional"])

        if qty <= 0:
            logger.error("[TRADE] qty<=0")
            return None

        resp = self.client.place_order(self.symbol, side=side, qty=qty, order_type="Market")
        if resp.get("retCode") != 0:
            logger.error(f"[ORDER][FAIL] {resp}")
            return None

        # Confirm position
        time.sleep(0.5)
        if not self._confirm_open():
            logger.error("[ORDER] not confirmed by /position/list")
            return None

        # Set SL/TP
        ts = self.client.trading_stop(self.symbol, side=side, stop_loss=sl, take_profit=tp)
        if ts.get("retCode") != 0:
            logger.warning(f"[TPSL][WARN] {ts}")

        self.position_open = True
        return {"order": resp, "qty": qty, "sl": sl, "tp": tp}

    def close_all(self):
        # В минимальной версии используем reverse market: если была лонг — шорт на тот же qty и наоборот
        pl = self.client.position_list(self.symbol)
        try:
            pos = pl["result"]["list"][0]
            side = "Sell" if float(pos["size"]) > 0 and pos["side"].lower()=="buy" else "Buy"
            qty = pos["size"]
            if float(qty) > 0:
                self.client.place_order(self.symbol, side=side, qty=qty, order_type="Market")
        except Exception as e:
            logger.error(f"[CLOSE_ALL] {e}")
        self.position_open = False
# trader_mod/risk.py
from typing import Dict
from .utils import round_step

class Risk:
    def __init__(self, account, risk_pct: float):
        self.account = account
        self.risk_pct = float(risk_pct)

    def round_down_qty(self, qty: float) -> float:
        f = self.account.ensure_filters()
        step = float(f.get("qtyStep", 0.001) or 0.001)
        n = int(qty / step)
        return max(step, n * step)

    def calc_qty(self, side: str, price: float, sl: float) -> float:
        f: Dict[str, float] = self.account.ensure_filters()
        stop_dist = abs(price - sl)
        if stop_dist <= 0:
            return 0.0

        risk_amt = max(self.account.equity * self.risk_pct, 0.0)
        qty_risk = risk_amt / stop_dist

        FEE_BUF = 1.003
        margin_per_qty = price / max(self.account.leverage, 1.0)
        if margin_per_qty <= 0:
            return 0.0
        qty_afford = (self.account.available / (margin_per_qty * FEE_BUF)) if self.account.available > 0 else qty_risk

        raw = max(0.0, min(qty_risk, qty_afford))
        qty = round_step(raw, f["qtyStep"])
        if qty < f["minQty"]:
            return 0.0
        mn = f.get("minNotional", 0.0) or 0.0
        if mn > 0 and (qty * price) < mn:
            return 0.0
        return qty
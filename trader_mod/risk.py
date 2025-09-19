# trader_mod/risk.py
import math

def _round_step(self, value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor(value / step + 1e-12) * step

def _ceil_step(self, value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.ceil(value / step - 1e-12) * step

def _fmt(self, x: float) -> str:
    return f"{x:.6f}".rstrip("0").rstrip(".")

def _round_down_qty(self, qty: float) -> float:
    try:
        f = self.ensure_filters()
        step = float(f.get("qtyStep", 0.001))
    except Exception:
        step = 0.001
    if step <= 0:
        step = 0.001
    n = int(qty / step)
    return max(step, n * step)

def _calc_qty(self, side: str, price: float, sl: float) -> float:
    f = self.ensure_filters()
    stop_dist = abs(price - sl)
    if stop_dist <= 0:
        return 0.0

    risk_amt = max(self.equity * self.risk_pct, 0.0)
    qty_risk = risk_amt / stop_dist

    FEE_BUF = 1.003
    margin_per_qty = price / max(self.leverage, 1.0)
    if margin_per_qty <= 0:
        return 0.0
    qty_afford = (self.available / (margin_per_qty * FEE_BUF)) if self.available > 0 else qty_risk

    raw = max(0.0, min(qty_risk, qty_afford))
    qty = _round_step(self, raw, f["qtyStep"])
    if qty < f["minQty"]:
        return 0.0
    min_notional = f.get("minNotional", 0.0) or 0.0
    if min_notional > 0 and (qty * price) < min_notional:
        return 0.0
    return qty
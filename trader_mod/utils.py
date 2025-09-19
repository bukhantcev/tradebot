# trader_mod/utils.py
import math

def fmt(x: float) -> str:
    return f"{float(x):.6f}".rstrip("0").rstrip(".")

def round_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor(value / step + 1e-12) * step

def ceil_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.ceil(value / step - 1e-12) * step
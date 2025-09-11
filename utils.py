# utils.py
from math import floor

def fmt_to_step(value: float, step: float) -> float:
    """
    Округление вниз к шагу цены/тика.
    """
    if step <= 0:
        return value
    k = floor(value / step)
    return k * step
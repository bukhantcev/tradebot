from typing import Tuple
from config import CFG
from log import log
from bybit_client import bybit

def _align(price: float, tick: float) -> float:
    return round(price / tick) * tick

def find_sr(klines: list[dict]) -> Tuple[float, float]:
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    left, right = CFG.sr_pivot_left, CFG.sr_pivot_right
    support, resistance = None, None
    for i in range(left, len(lows) - right):
        if all(lows[i] < lows[j] for j in range(i-left, i)) and all(lows[i] < lows[j] for j in range(i+1, i+right+1)):
            support = lows[i]
        if all(highs[i] > highs[j] for j in range(i-left, i)) and all(highs[i] > highs[j] for j in range(i+1, i+right+1)):
            resistance = highs[i]
    if support: support = _align(support, bybit.tick_size)
    if resistance: resistance = _align(resistance, bybit.tick_size)
    log.info(f"[SR] calc -> support={support} resistance={resistance} tickSize={bybit.tick_size}")
    return support, resistance
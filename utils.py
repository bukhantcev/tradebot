import hmac
import hashlib
import math
import random
import time

def hmac_sha256(secret: str, msg: str) -> str:
    return hmac.new(secret.encode(), msg.encode(), hashlib.sha256).hexdigest()

def now_ms() -> int:
    return int(time.time() * 1000)

def exp_backoff(retry: int, base: float = 0.5, cap: float = 10.0) -> float:
    return min(cap, base * (2 ** retry)) + random.random() * 0.2

def clamp_qty_step(qty: float, step: float, min_qty: float) -> float:
    if step <= 0:
        return max(qty, min_qty)
    k = math.floor(qty / step) * step
    if k < min_qty:
        k = 0.0
    return float(f"{k:.12f}")

def fmt_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    decimals = max(0, len(str(step).split('.')[-1]))
    return float(f"{math.floor(value / step) * step:.{decimals}f}")
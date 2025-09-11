from typing import Literal
from log import log

Mode = Literal["TRND_UP", "TRND_DN", "FLAT"]

def detect_mode(price: float, support: float, resistance: float) -> Mode:
    if support and price > resistance:
        return "TRND_UP"
    if resistance and price < support:
        return "TRND_DN"
    return "FLAT"

def trade_signal(mode: Mode, price: float, support: float, resistance: float):
    if mode == "TRND_UP" and price >= support:
        log.info("[TRADE] signal -> long")
        return "long"
    if mode == "TRND_DN" and price <= resistance:
        log.info("[TRADE] signal -> short")
        return "short"
    if mode == "FLAT":
        if price <= support:
            log.info("[TRADE] signal -> flat long")
            return "long"
        if price >= resistance:
            log.info("[TRADE] signal -> flat short")
            return "short"
    return None
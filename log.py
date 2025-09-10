import logging
from typing import Any

logger = logging.getLogger("scalper")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(_h)

def log(msg: str):
    try:
        logger.info(msg)
    except Exception:
        print(msg)

def mask(s: str, keep: int = 4) -> str:
    if not s:
        return "<empty>"
    if len(s) <= keep:
        return "*" * len(s)
    return s[:keep] + "*" * (len(s) - keep)

def trunc(obj: Any, limit: int = 800) -> str:
    s = str(obj)
    if len(s) > limit:
        return s[:limit] + "…(trunc)"
    return s
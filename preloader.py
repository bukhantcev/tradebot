

"""preloader.py

Прелоадер исторических свечей с Bybit.

Функция `load_history_candles(limit, interval, symbol=None, category=None)`
возвращает список свечей в хронологическом порядке, где каждая свеча — это dict:
{
    't': start_ms (int),
    'o': open (float),
    'h': high (float),
    'l': low  (float),
    'c': close(float),
    'v': volume(float),
}

Параметры:
- limit: сколько свечей получить (int)
- interval: таймфрейм Bybit Kline: "1"=1m, "3"=3m, "5"=5m, "15"=15m, "60"=1h, "240"=4h, "D"=1d и т.п.
- symbol/category: можно не указывать — возьмутся из Config
"""
from __future__ import annotations

from typing import List, Dict

from pybit.unified_trading import HTTP

from config import Config
from logger import get_logger

log = get_logger("PRELOADER")


def _as_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "on"}


def load_history_candles(limit: int, interval: str, *, symbol: str | None = None, category: str | None = None) -> List[Dict[str, float]]:
    """Загрузить последние `limit` свечей с Bybit для заданного `interval`.

    :param limit: количество свечей (int)
    :param interval: таймфрейм Bybit Kline (например, "1", "5", "60", "240", "D")
    :param symbol: тикер (по умолчанию из Config)
    :param category: категория (linear/spot/inverse/option), по умолчанию из Config
    :return: список свечей dict в ХРОНОЛОГИЧЕСКОМ порядке: [{'t','o','h','l','c','v'}, ...]
    """
    cfg = Config()
    is_testnet = _as_bool(getattr(cfg, "testnet", False))

    sym = (symbol or getattr(cfg, "symbol", "BTCUSDT")).upper()
    cat = (category or getattr(cfg, "category", "linear")).lower()

    session = HTTP(
        testnet=is_testnet,
        api_key=cfg.bybit_api_key_test if is_testnet else cfg.bybit_api_key_main,
        api_secret=cfg.bybit_api_secret_test if is_testnet else cfg.bybit_api_secret_main,
        recv_window=20_000,
        timeout=30,
        max_retries=2,
    )

    try:
        resp = session.get_kline(category=cat, symbol=sym, interval=str(interval), limit=int(limit))
        raw_list = (resp or {}).get("result", {}).get("list", [])
    except Exception as e:
        log.error(f"load_history_candles: fetch failed: {e}")
        return []

    candles: List[Dict[str, float]] = []
    for it in raw_list:
        try:
            # Bybit чаще всего отдаёт массив: [start, open, high, low, close, turnover, volume, ...]
            if isinstance(it, (list, tuple)):
                t = int(it[0])
                o = float(it[1])
                h = float(it[2])
                l = float(it[3])
                c = float(it[4])
                # Volume у линейных перпов — в базовой монете в it[6]
                v = float(it[6]) if len(it) > 6 else float(it[5])
            else:
                # На всякий случай поддержим словарный формат
                t = int(it.get("start") or it.get("t") or it.get("startTime") or 0)
                o = float(it.get("open") or it.get("o"))
                h = float(it.get("high") or it.get("h"))
                l = float(it.get("low") or it.get("l"))
                c = float(it.get("close") or it.get("c") or it.get("closePrice"))
                v = float(it.get("volume") or it.get("v") or 0)
            candles.append({"t": t, "o": o, "h": h, "l": l, "c": c, "v": v})
        except Exception as conv_e:
            log.debug(f"load_history_candles: skip bad item: {conv_e} | item={it}")
            continue

    # Bybit возвращает новые первой — развернём в хронику
    candles.reverse()
    return candles


if __name__ == "__main__":
    # Пример быстрого теста: подтянуть 100 свечей 1m из конфига
    data = load_history_candles(limit=100, interval="1")
    if data:
        # выведем последнюю
        last = data[-1]
        log.info("last candle: t=%s o=%.2f h=%.2f l=%.2f c=%.2f v=%.0f", last['t'], last['o'], last['h'], last['l'], last['c'], last['v'])
    else:
        log.warning("no candles fetched")
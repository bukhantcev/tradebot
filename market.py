

"""market.py — простая обёртка для размещения ордеров на Bybit v5.

Функция place_order принимает параметры ордера через словарь и отправляет их в API.
Тестовый запуск внизу формирует пример словаря (без каких‑либо ключей доступа).
"""
from __future__ import annotations
import json
import time
from typing import Any, Dict


from pybit.unified_trading import HTTP
from config import Config
from logger import get_logger
log = get_logger("MARKET")


def _filter_none(d: Dict[str, Any]) -> Dict[str, Any]:
    """Удалить пары, где значение None — чтобы не слать пустые поля в API."""
    return {k: v for k, v in d.items() if v is not None}


def place_order(params: Dict[str, Any], session) -> Dict[str, Any]:
    """Разместить ордер на Bybit v5.

    Ожидает словарь параметров, совместимых с /v5/order/create (category, symbol, side, orderType, qty, price и т.д.).
    Если в словаре нет `category`/`symbol`, подставим значения из Config.
    """
    cfg = Config()
    session = session
    log.info(f"Отправляю ордер: {json.dumps(params, ensure_ascii=False)}")
    resp = session.place_order(**params)
    log.info(f"Ответ Bybit: {json.dumps(resp, ensure_ascii=False)}")
    return resp


def set_trading_stop(params: Dict[str, Any], session) -> Dict[str, Any]:
    """Установить торговый стоп/трейлинг на позицию. Параметры передаются как есть."""
    cfg = Config()
    session = session
    log.info(f"set_trading_stop → {json.dumps(params, ensure_ascii=False)}")
    resp = session.set_trading_stop(**params)
    log.info(f"Ответ set_trading_stop: {json.dumps(resp, ensure_ascii=False)}")
    return resp


if __name__ == "__main__":
    # Тестовый запуск: составляем пример словаря с минимальным набором полей (без ключей доступа)

    example_order: Dict[str, Any] = {
        "category": "linear",          # тип продукта
        "symbol": "ETHUSDT",           # торговая пара
        "side": "Buy",                 # направление сделки: Buy/Sell
        "orderType": "Limit",          # Market или Limit
        "qty": "0.01",                 # количество
        "price": "2500",               # цена (для Limit)
        "timeInForce": "GTC",          # GTC/IOC/FOK/PostOnly
        "reduceOnly": False,           # закрывать ли только позицию
        "closeOnTrigger": False,       # стоп при срабатывании
        "orderLinkId": f"test_{int(time.time() * 1000)}",  # уникальный ID
    }

    try:
        place_order(example_order)

        # Пример: отдельно поставить биржевой скользящий стоп на позицию
        example_trailing = {
            "category": "linear",
            "symbol": "ETHUSDT",
            "positionIdx": 0,           # 0 = one-way; 1/2 для hedge-mode
            "trailingStop": "0.20",    # абсолютный шаг цены
            "trailingActive": "6.80",  # цена активации
            "slTriggerBy": "LastPrice",
        }
        set_trading_stop(example_trailing)
    except Exception as e:
        log.error(f"Ошибка при размещении ордера: {e}")
import json
from config import Config
from pybit.unified_trading import HTTP
from decimal import Decimal
from logger import get_logger
cfg = Config()
is_testnet = True if cfg.testnet=="true".lower() else False
log = get_logger("BYBIT_REQ")

bb_key = cfg.bybit_api_key_test if is_testnet else cfg.bybit_api_key_main
bb_secret = cfg.bybit_api_secret_test if is_testnet else cfg.bybit_api_secret_main


def get_balance():
    """Вернуть баланс USDT."""
    cfg = Config()
    session = HTTP(
        testnet=is_testnet,
        api_key=bb_key,
        api_secret=bb_secret,
    )
    resp = session.get_wallet_balance(accountType="UNIFIED")
    coins = resp["result"]["list"][0]["coin"]
    usdt = next((c for c in coins if c["coin"] == "USDT"), None)
    if usdt is None:
        raise Exception("USDT balance not found")
    balance = float(usdt["walletBalance"])
    balance_rounded = round(balance, 2)
    print(balance_rounded)
    return balance_rounded

def get_last_price():
    """Вернуть последнюю цену символа."""
    cfg = Config()
    session = HTTP(
        testnet=is_testnet,
        api_key=bb_key,
        api_secret=bb_secret,
    )
    symbol = getattr(cfg, "symbol", "BTCUSDT").upper()
    resp = session.get_tickers(category=cfg.category, symbol=symbol)
    last_price = float(resp["result"]["list"][0]["lastPrice"])
    last_price_rounded = round(last_price, 2)
    return last_price_rounded

def get_min_qty():
    """Вернуть минимальное количество монет (BTC и т.д.)."""
    cfg = Config()
    session = HTTP(
        testnet=is_testnet,
        api_key=bb_key,
        api_secret=bb_secret,
    )
    symbol = getattr(cfg, "symbol", "BTCUSDT").upper()
    resp = session.get_instruments_info(category=cfg.category, symbol=symbol)
    item = resp["result"]["list"][0]
    lot = item.get("lotSizeFilter", {}) if isinstance(item, dict) else {}
    min_qty_str = lot.get("minOrderQty") or item.get("minOrderQty")
    if min_qty_str is None:
        raise KeyError("minOrderQty not found in lotSizeFilter or instrument root")
    min_qty = float(min_qty_str)
    min_qty_rounded = round(min_qty, 6)
    decimals = abs(Decimal(min_qty_str).as_tuple().exponent)
    result = {"min_qty": min_qty_rounded, "decimals": decimals}
    return result

def get_min_notional():
    """Вернуть минимальное количество в USDT."""
    cfg = Config()
    session = HTTP(
        testnet=is_testnet,
        api_key=bb_key,
        api_secret=bb_secret,
    )
    symbol = getattr(cfg, "symbol", "BTCUSDT").upper()
    resp = session.get_instruments_info(category=cfg.category, symbol=symbol)
    item = resp["result"]["list"][0]
    lot = item.get("lotSizeFilter", {}) if isinstance(item, dict) else {}
    min_notional_str = lot.get("minNotionalValue") or item.get("minNotionalValue")
    if min_notional_str is None:
        raise KeyError("minNotionalValue not found in lotSizeFilter or instrument root")
    min_notional = float(min_notional_str)
    min_notional_rounded = round(min_notional, 2)
    return min_notional_rounded


def get_open_limit_orders():
    """Вернуть список открытых лимитных ордеров по символу."""
    cfg = Config()
    session = HTTP(
        testnet=is_testnet,
        api_key=bb_key,
        api_secret=bb_secret,
    )
    symbol = getattr(cfg, "symbol", "BTCUSDT").upper()
    resp = session.get_open_orders(category=cfg.category, symbol=symbol, limit=50)
    items = resp.get("result", {}).get("list", []) if isinstance(resp, dict) else []
    limit_orders = [o for o in items if isinstance(o, dict) and o.get("orderType") == "Limit"]
    # Печатаем в читаемом виде и возвращаем список
    print(json.dumps(limit_orders, ensure_ascii=False, indent=2))
    return limit_orders


# Новый метод: получить список открытых позиций по символу
def get_open_positions():
    """Вернуть список открытых позиций по символу."""
    cfg = Config()
    session = HTTP(
        testnet=is_testnet,
        api_key=bb_key,
        api_secret=bb_secret,
    )
    symbol = getattr(cfg, "symbol", "BTCUSDT").upper()
    resp = session.get_positions(category=cfg.category, symbol=symbol)
    items = resp.get("result", {}).get("list", []) if isinstance(resp, dict) else []
    # Печатаем в читаемом виде и возвращаем список
    return items

def get_tick_size():
    """Вернуть тиксайз (минимальный шаг цены) для символа."""
    cfg = Config()
    session = HTTP(
        testnet=is_testnet,
        api_key=bb_key,
        api_secret=bb_secret,
    )
    symbol = getattr(cfg, "symbol", "BTCUSDT").upper()
    resp = session.get_instruments_info(category=cfg.category, symbol=symbol)
    item = resp["result"]["list"][0]
    price_filter = item.get("priceFilter", {}) if isinstance(item, dict) else {}
    tick_size_str = price_filter.get("tickSize") or item.get("tickSize")
    if tick_size_str is None:
        raise KeyError("tickSize not found in priceFilter or instrument root")
    tick_size = float(tick_size_str)
    tick_size_rounded = round(tick_size, 8)
    decimals = abs(Decimal(tick_size_str).as_tuple().exponent)
    return {"tick_size": tick_size_rounded, "decimals": decimals}


def prepare_candles(raw_candles):
    """Преобразовать сырые свечи в список словарей с ключами h, l, c."""
    result = []
    for candle in raw_candles:
        result.append({
            "h": candle[2],
            "l": candle[3],
            "c": candle[4],
        })
    return result

def get_candles_for_atr(period: int = 14, interval: str = "1", *, symbol: str | None = None, category: str | None = None, limit: int | None = None) -> list[dict[str, float]]:
    """
    Загрузить последние свечи с Bybit и вернуть в формате для atr(): [{'h': float, 'l': float, 'c': float}, ...].

    :param period: окно ATR (минимум нужно period+1 свечей)
    :param interval: таймфрейм Bybit Kline ("1"=1m, "3"=3m, "5"=5m, "60"=1h и т.д.)
    :param symbol: тикер (по умолчанию берёт из Config)
    :param category: категория (linear/spot/inverse/option), по умолчанию из Config
    :param limit: сколько свечей запрашивать. Если None — возьмём max(period+1, 50)
    :return: список свечей в формате [{'h': high, 'l': low, 'c': close}] в ХРОНОЛОГИЧЕСКОМ порядке
    """
    cfg = Config()
    session = HTTP(
        testnet=is_testnet,
        api_key=bb_key,
        api_secret=bb_secret,
    )
    sym = (symbol or getattr(cfg, "symbol", "BTCUSDT")).upper()
    cat = (category or getattr(cfg, "category", "linear"))
    req_limit = limit or max(int(period) + 1, 50)

    try:
        resp = session.get_kline(category=cat, symbol=sym, interval=str(interval), limit=req_limit)
        raw = (resp or {}).get("result", {}).get("list", [])
    except Exception as e:
        log.error(f"get_candles_for_atr: kline fetch failed: {e}")
        return []

    candles: list[dict[str, float]] = []
    for it in raw:
        try:
            if isinstance(it, (list, tuple)):
                # Bybit layout: [start, open, high, low, close, turnover, volume, ...]
                high = float(it[2])
                low = float(it[3])
                close = float(it[4])
            elif isinstance(it, dict):
                high = float(it.get("high") or it.get("h"))
                low = float(it.get("low") or it.get("l"))
                close = float(it.get("close") or it.get("c") or it.get("closePrice"))
            else:
                continue
            candles.append({"h": high, "l": low, "c": close})
        except Exception:
            continue

    # Bybit обычно возвращает новые первой — приведём к хронологическому порядку
    candles = list(reversed(candles))

    # Оставим только последние period+1 свечей (для расчёта ATR требуется prev_close)
    need = int(period) + 1
    if len(candles) > need:
        candles = candles[-need:]

    return candles


print(get_open_positions()[0])
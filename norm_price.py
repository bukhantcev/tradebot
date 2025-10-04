from bybit_request import get_tick_size, get_min_qty, get_last_price, get_min_notional, get_open_positions, get_candles_for_atr
from logger import get_logger
from indikators import atr
log = get_logger("NORM_PRICE")


def make_qty(qty: float):
    min_qty = float(get_min_qty()["min_qty"])
    round_qty = int(get_min_qty()["decimals"])
    last_price = float(get_last_price())
    min_notional = float(get_min_notional())
    decimals = int(get_tick_size()["decimals"])
    final_qty = float(qty)
    if float(qty) < min_qty:
        log.error(f"Заявленное количество меньше допустимого - {min_qty}")
        return None
    final_qty_usdt = float(qty) * last_price
    if final_qty_usdt < min_notional:
        final_qty = min_notional/last_price
        log.info(f"Количество меньше допустимого {min_notional} USDT, взял минимально возможное - {round(final_qty, round_qty)}")
        return round(final_qty, round_qty)
    log.info(f"Количество - {final_qty} - ОК")
    return final_qty

atr = atr(get_candles_for_atr(period=14, interval="1"))
def make_sl_market(risk: float, side: str):
    last_price = float(get_last_price())
    sl_price = last_price - atr*risk if side=="Buy" else last_price + atr*risk
    decimals = int(get_tick_size()["decimals"])
    sl_price = round(sl_price, decimals)
    log.info(f"Стоп-лосс расчитан - {sl_price}")
    return sl_price


def make_tp_market(tp: float, side: str):
    last_price = float(get_last_price())
    tp_price = last_price + atr*tp if side=="Buy" else last_price - atr*tp
    decimals = int(get_tick_size()["decimals"])
    tp_price = round(tp_price, decimals)
    log.info(f"Тейк-профит расчитан - {tp_price}")
    return tp_price


def treiling_sl(risk: float, tp:float):
    open_price = float(get_open_positions()[0]["avgPrice"])
    decimals = int(get_tick_size()["decimals"])
    trailingStop = open_price - open_price*risk
    trailingStop = round(trailingStop, decimals)
    trailingActive = open_price + open_price*tp
    trailingActive = round(trailingActive, decimals)
    return {"trailingStop": trailingStop, "trailingActive": trailingActive}






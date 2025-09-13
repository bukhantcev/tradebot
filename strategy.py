from typing import Tuple, Literal, Optional
from config import CFG

Mode = Literal["FLAT", "TRND_UP", "TRND_DN"]

# ====== 5m S/R: для биржевого SL ======
def calc_sr_5m(candles_5m: list[tuple], left: int, right: int) -> Tuple[float, float, float]:
    """
    Фрактальная/оконная S/R по 5м.
    Возвращает (support5, resistance5, tick).
    Простейший вариант: min/max из окна последних N свечей.
    """
    if not candles_5m:
        raise ValueError("calc_sr_5m: empty candles_5m")
    N = max(30, left + right + 10)
    window = candles_5m[-N:]
    support5 = min(c[3] for c in window)   # low
    resistance5 = max(c[2] for c in window)  # high
    tick = 0.1  # для BTCUSDT в тестнете; по-хорошему приходит из instruments-info
    return support5, resistance5, tick


# ====== 1m S/R: для генерации сигналов ======
def calc_sr_1m(candles_1m: list[tuple], left: int, right: int) -> Tuple[float, float]:
    """
    Фрактальная/оконная S/R по 1м.
    Возвращает (support1, resistance1).
    """
    if not candles_1m:
        raise ValueError("calc_sr_1m: empty candles_1m")
    N = max(20, left + right + 6)
    window = candles_1m[-N:]
    support1 = min(c[3] for c in window)
    resistance1 = max(c[2] for c in window)
    return support1, resistance1


# ====== Режим рынка ======
def regime_1m(candles_1m: list[tuple]) -> Mode:
    """
    «Быстрый» минутный режим:
    close выше медианы последних экстремумов — TRND_UP, ниже — TRND_DN, иначе FLAT.
    """
    if len(candles_1m) < 2:
        return "FLAT"
    highs = [c[2] for c in candles_1m[-30:]]
    lows = [c[3] for c in candles_1m[-30:]]
    mid = (max(highs) + min(lows)) / 2
    close = candles_1m[-1][4]
    if close > mid:
        return "TRND_UP"
    if close < mid:
        return "TRND_DN"
    return "FLAT"


def regime_combined(candles_1m: list[tuple], s5: float, r5: float) -> Mode:
    """
    Комбинированный режим: быстрый сигналный тренд с 1м,
    но с HFT-байасом от 5м: если 1м-тренд против 5м-середины — считаем FLAT.
    """
    mode1 = regime_1m(candles_1m)
    close = candles_1m[-1][4]
    mid5 = (s5 + r5) / 2
    if mode1 == "TRND_UP" and close < mid5:
        return "FLAT"
    if mode1 == "TRND_DN" and close > mid5:
        return "FLAT"
    return mode1


# ====== Сигналы по закрытию 1м к 1м-уровням ======
def signal_on_1m_close(
    candles_1m: list[tuple],
    mode: Mode,
    s1: float,
    r1: float,
    *,
    touch_pct: float,
    break_pct: float
) -> Optional[str]:
    """
    Сигнал формируется к минутным уровням:
      - long в TRND_UP, если:
          (a) закрытие выше предыдущего и
          (b1) касание поддержки s1 в пределах touch_pct ИЛИ
          (b2) пробой r1 не менее break_pct
      - short в TRND_DN, симметрично.

    touch_pct, break_pct задаются в процентах (напр. 0.05 = 0.05%).
    """
    if len(candles_1m) < 3:
        return None

    prev_close = candles_1m[-2][4]
    close = candles_1m[-1][4]

    touch_band = close * (touch_pct / 100.0)
    break_up = r1 * (1 + break_pct / 100.0)
    break_dn = s1 * (1 - break_pct / 100.0)

    # условия «касания» уровня
    touch_support = abs(close - s1) <= touch_band
    touch_resist = abs(close - r1) <= touch_band

    if mode == "TRND_UP":
        if close > prev_close and (touch_support or close >= break_up):
            return "long"
    elif mode == "TRND_DN":
        if close < prev_close and (touch_resist or close <= break_dn):
            return "short"

    return None


# ====== Биржевой SL от 5м уровней ======
def compute_sl_exchange(side: str, s5: float, r5: float, tick: float) -> float:
    """
    Биржевой стоп ставим только по 5м уровням с буфером в тиках.
    - long: ниже support5 на CFG.sl_buffer_ticks
    - short: выше resistance5 на CFG.sl_buffer_ticks
    """
    if side == "long":
        return round(s5 - CFG.sl_buffer_ticks * tick, 1)
    else:
        return round(r5 + CFG.sl_buffer_ticks * tick, 1)


# ====== Виртуальные уровни SL/TP от цены входа (оставляем без изменений) ======
def virtual_levels(entry: float, side: str) -> Tuple[float, float]:
    """
    Виртуальный SL/TP как проценты от цены входа (0.2% по умолчанию).
    """
    if side == "long":
        tp = entry * (1 + CFG.virtual_tp_pct)
        sl = entry * (1 - CFG.virtual_sl_pct)
    else:
        tp = entry * (1 - CFG.virtual_tp_pct)
        sl = entry * (1 + CFG.virtual_sl_pct)
    return sl, tp
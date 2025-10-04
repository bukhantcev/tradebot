"""
indikators.py — Композитный индикатор тренда (тик/бар-независимый).

Выдаёт:
- trend: bool — подтверждённый тренд (True/False)
- strength: float — непрерывный коэффициент 0..2 (0 — очень слабый, 1 — есть тренд, 2 — очень сильный)
- direction: str — 'up' / 'down' / 'flat'
- confidence: float — 0..1, насколько уверенно подтверждён сигнал
- details: dict — диагностические метрики

Идея: склейка нескольких быстрых метрик, которые можно обновлять инкрементально,
но здесь реализована оффлайн-функция по массивам (подойдёт и для тиков, и для баров):
  • EMA fast/slow (направление)
  • MACD гистограмма и её наклон (импульс)
  • Rolling VWAP и положение цены относительно него (средняя и перекупленность)
  • OBV (объёмный импульс) и его наклон (подтверждение потоком)
  • EW-волатильность (для адаптивных порогов)

Подтверждение: сигнал считается подтверждённым, когда ≥3 из 4 критериев согласованы
в одном направлении на последних N точках (по умолчанию 3).
"""
from __future__ import annotations
from typing import Sequence, Dict, Any, Optional
from preloader import load_history_candles


# Разрешено без внешних зависимостей; используем только стандартный Python


def _ema(seq: Sequence[float], alpha: float) -> list[float]:
    """Экспоненциальное сглаживание (EWMA) c коэффициентом alpha в (0,1]."""
    out: list[float] = []
    s: Optional[float] = None
    for x in seq:
        if x is None:
            out.append(s if s is not None else 0.0)
            continue
        if s is None:
            s = float(x)
        else:
            s = s + alpha * (float(x) - s)
        out.append(s)
    return out


def _ema_period(seq: Sequence[float], period: int) -> list[float]:
    """EMA по периоду N: alpha = 2/(N+1)."""
    period = max(1, int(period))
    alpha = 2.0 / (period + 1.0)
    return _ema(seq, alpha)


def _diff(seq: Sequence[float]) -> list[float]:
    """Первые разности (x_t - x_{t-1}), для t=0 => 0."""
    out: list[float] = []
    prev: Optional[float] = None
    for x in seq:
        if prev is None:
            out.append(0.0)
        else:
            out.append(float(x) - float(prev))
        prev = x
    return out


def _rolling_vwap(prices: Sequence[float], volumes: Optional[Sequence[float]], window: int) -> list[float]:
    """Простой rolling-VWAP по окну window. Если volume=None, эквивалент скользящей средней."""
    if volumes is None:
        volumes = [1.0] * len(prices)
    out: list[float] = []
    num = 0.0
    den = 0.0
    q: list[tuple[float, float]] = []  # (p*v, v)
    for p, v in zip(prices, volumes):
        pv = float(p) * float(v)
        num += pv
        den += float(v)
        q.append((pv, float(v)))
        if len(q) > window:
            rm_pv, rm_v = q.pop(0)
            num -= rm_pv
            den -= rm_v
        out.append(num / den if den > 0 else float(p))
    return out


def _obv(prices: Sequence[float], volumes: Optional[Sequence[float]]) -> list[float]:
    """On Balance Volume: кумулятивно добавляем объем, если цена выросла, вычитаем — если упала."""
    if volumes is None:
        volumes = [1.0] * len(prices)
    out: list[float] = []
    total = 0.0
    prev: Optional[float] = None
    for p, v in zip(prices, volumes):
        if prev is not None:
            if p > prev:
                total += float(v)
            elif p < prev:
                total -= float(v)
        out.append(total)
        prev = p
    return out


def _slope_last(values: Sequence[float], k: int = 3) -> float:
    """Грубый наклон: средняя разность на последних k точках."""
    if not values:
        return 0.0
    k = max(1, min(k, len(values)-1))
    diffs = _diff(values[-(k+1):])
    return sum(diffs[-k:]) / float(k)

def preloaded_to_series(candles: Sequence[Dict[str, float]]) -> tuple[list[float], list[float]]:
    """Преобразует свечи из preloader (t,o,h,l,c,v) в ряды closes и volumes."""
    closes: list[float] = []
    vols: list[float] = []
    for c in candles:
        try:
            closes.append(float(c["c"]))
            vols.append(float(c.get("v", 0.0)))
        except Exception:
            continue
    return closes, vols

def atr(candles: Sequence[Dict[str, float]], period: int = 14) -> Optional[float]:
    """
    Рассчитывает ATR (Average True Range) по списку свечей.
    :param candles: список словарей с ключами 'h','l','c' (high, low, close)
    :param period: окно усреднения (по умолчанию 14)
    :return: значение ATR или None, если данных меньше чем period
    """
    if len(candles) < period + 1:
        return None

    trs: list[float] = []
    for i in range(1, len(candles)):
        high = float(candles[i]["h"])
        low = float(candles[i]["l"])
        prev_close = float(candles[i-1]["c"])
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )
        trs.append(tr)

    return sum(trs[-period:]) / period

def atr_from_preloader(period: int = 14, interval: str = "1", *, limit: int | None = None) -> Optional[float]:
    """Грузит историю через preloader и возвращает ATR по ней."""
    need = (limit or max(period + 1, 50))
    candles = load_history_candles(limit=need, interval=str(interval))
    if not candles:
        return None
    # подготовим компактный формат для atr(): [{'h','l','c'}, ...]
    slim = [{"h": float(c["h"]), "l": float(c["l"]), "c": float(c["c"])} for c in candles]
    return atr(slim, period=period)


def trend_from_preloader(
    *,
    limit: int = 100,
    interval: str = "1",
    ema_fast: int = 21,
    ema_slow: int = 55,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    vwap_window: int = 300,
    confirm_lookback: int = 3,
) -> Dict[str, Any]:
    """Грузит свечи через preloader и считает композитный индикатор тренда на их основе."""
    candles = load_history_candles(limit=limit, interval=str(interval))
    closes, vols = preloaded_to_series(candles)
    return trend_indicator(
        closes,
        vols,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        vwap_window=vwap_window,
        confirm_lookback=confirm_lookback,
    )

def trend_indicator(
    prices: Sequence[float],
    volumes: Optional[Sequence[float]] = None,
    *,
    ema_fast: int = 21,
    ema_slow: int = 55,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    vwap_window: int = 300,   # ~5 мин при 1 тике/сек; под своё тик-частоту подстрой
    confirm_lookback: int = 3,
) -> Dict[str, Any]:
    """Композитный индикатор тренда с подтверждением.

    :param prices: последовательность цен (тик/close), длиной >= 50 желательно
    :param volumes: объёмы (необязательно, но улучшает OBV/VWAP)
    :param ema_fast, ema_slow: периоды EMA для направления тренда
    :param macd_fast, macd_slow, macd_signal: параметры MACD
    :param vwap_window: окно VWAP (в наблюдениях)
    :param confirm_lookback: сколько последних точек должны согласоваться для подтверждения
    :return: dict с ключами: trend, strength (float 0..2), direction ('up'/'down'/'flat'), confidence (0..1), details
    """
    n = len(prices)
    if n == 0:
        return {"trend": False, "strength": 0, "direction": "flat", "confidence": 0.0, "details": {}}

    # 1) EMA fast/slow
    ema_f = _ema_period(prices, ema_fast)
    ema_s = _ema_period(prices, ema_slow)
    ema_spread = [f - s for f, s in zip(ema_f, ema_s)]

    # 2) MACD (гистограмма и наклон)
    macd_line = [ef - es for ef, es in zip(_ema_period(prices, macd_fast), _ema_period(prices, macd_slow))]
    macd_signal_line = _ema_period(macd_line, macd_signal)
    macd_hist = [m - s for m, s in zip(macd_line, macd_signal_line)]

    # 3) VWAP (rolling)
    vwap = _rolling_vwap(prices, volumes, vwap_window)

    # 4) OBV и его наклон
    obv = _obv(prices, volumes)

    # 5) EW наклоны на концах
    slope_price = _slope_last(prices, k=confirm_lookback)
    slope_ema_spread = _slope_last(ema_spread, k=confirm_lookback)
    slope_macd_hist = _slope_last(macd_hist, k=confirm_lookback)
    slope_obv = _slope_last(obv, k=confirm_lookback)

    # 6) Локальные булевы критерии (последняя точка)
    last = -1
    up_cond = [
        ema_spread[last] > 0,               # быстрый EMA выше медленного
        macd_hist[last] > 0 and slope_macd_hist > 0,  # MACD+ импульс
        prices[last] > vwap[last],          # выше VWAP
        slope_obv > 0,                      # объём подтверждает
    ]
    down_cond = [
        ema_spread[last] < 0,
        macd_hist[last] < 0 and slope_macd_hist < 0,
        prices[last] < vwap[last],
        slope_obv < 0,
    ]

    # 7) Подтверждение на последних confirm_lookback точках
    def _confirmed(direction: str) -> int:
        ok = 0
        for i in range(confirm_lookback):
            idx = -1 - i
            if direction == 'up':
                c = (
                    ema_spread[idx] > 0 and
                    macd_hist[idx] > 0 and
                    prices[idx] > vwap[idx]
                )
            else:
                c = (
                    ema_spread[idx] < 0 and
                    macd_hist[idx] < 0 and
                    prices[idx] < vwap[idx]
                )
            ok += 1 if c else 0
        return ok

    up_score = sum(1 for c in up_cond if c)
    down_score = sum(1 for c in down_cond if c)

    # Выбор направления по большей сумме условий и знаку наклонов
    if up_score > down_score and (slope_price > 0 or slope_ema_spread > 0):
        direction = 'up'
        confirm_hits = _confirmed('up')
        base_score = up_score
    elif down_score > up_score and (slope_price < 0 or slope_ema_spread < 0):
        direction = 'down'
        confirm_hits = _confirmed('down')
        base_score = down_score
    else:
        direction = 'flat'
        confirm_hits = 0
        base_score = 0

    # Подтверждение: не меньше 2 из последних confirm_lookback точек согласованы
    confirmed = confirm_hits >= max(2, confirm_lookback // 2 + 1)

    # Непрерывная сила тренда 0..2: комбинируем согласованность условий, подтверждение и импульс (наклоны)
    score_frac = base_score / 4.0  # 0..1
    confirm_frac = (confirm_hits / max(1, confirm_lookback))  # 0..1

    # Нормируем импульс через типичный масштаб последних L точек
    L = max(10, min(50, n))
    avg_abs_macd = (sum(abs(x) for x in macd_hist[-L:]) / L) if n >= 1 else 0.0
    avg_abs_espd = (sum(abs(x) for x in ema_spread[-L:]) / L) if n >= 1 else 0.0
    eps = 1e-12
    momentum_norm = 0.5 * (abs(slope_macd_hist) / (avg_abs_macd + eps) + abs(slope_ema_spread) / (avg_abs_espd + eps))
    # ограничим импульс 0..1 чтобы не взрывался
    momentum_norm = max(0.0, min(1.0, momentum_norm))

    raw = 0.5 * score_frac + 0.3 * confirm_frac + 0.2 * momentum_norm  # 0..1
    strength = 2.0 * raw  # 0..2

    # Если не подтверждено или flat — подавим силу наполовину
    if not confirmed or direction == 'flat':
        strength *= 0.5

    trend = confirmed and direction != 'flat'

    # confidence: нормируем base_score и подтверждение
    confidence = 0.0
    if direction != 'flat':
        confidence = min(1.0, (base_score / 4.0) * (confirm_hits / max(1, confirm_lookback)))

    details: Dict[str, Any] = {
        "ema_spread": ema_spread[last],
        "macd_hist": macd_hist[last],
        "vwap": vwap[last],
        "price": prices[last],
        "obv_slope": slope_obv,
        "scores": {"up": up_score, "down": down_score},
        "confirm_hits": confirm_hits,
        "slopes": {
            "price": slope_price,
            "ema_spread": slope_ema_spread,
            "macd_hist": slope_macd_hist,
        },
        "strength_components": {
            "score_frac": score_frac,
            "confirm_frac": confirm_frac,
            "momentum_norm": momentum_norm,
        },
        "params": {
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "macd": (macd_fast, macd_slow, macd_signal),
            "vwap_window": vwap_window,
            "confirm_lookback": confirm_lookback,
        },
    }

    return {
        "trend": trend,
        "strength": strength,
        "direction": direction,
        "confidence": round(float(confidence), 3),
        "details": details,
    }


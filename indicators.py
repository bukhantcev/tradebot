from collections import deque
from typing import Deque, Optional, Tuple, List

class OnlineIndicators:
    def __init__(self, ema_fast: int, ema_slow: int, atr_len: int, channel_lookback: int):
        self.ema_fast_n = ema_fast
        self.ema_slow_n = ema_slow
        self.atr_n = atr_len
        self.ch_n = channel_lookback

        self.closes: Deque[float] = deque(maxlen=300)
        self.highs: Deque[float] = deque(maxlen=300)
        self.lows: Deque[float] = deque(maxlen=300)
        self.vols: Deque[float] = deque(maxlen=300)

        self.ema_fast: Optional[float] = None
        self.ema_slow: Optional[float] = None
        self.prev_close: Optional[float] = None
        self.atr: Optional[float] = None
        self.vwap_num = 0.0
        self.vwap_den = 0.0

    def reset_vwap(self):
        self.vwap_num = 0.0
        self.vwap_den = 0.0

    def push_candle(self, high: float, low: float, close: float, volume: float):
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        self.vols.append(volume)

        if self.ema_fast is None:
            self.ema_fast = close
            self.ema_slow = close
        else:
            kf = 2 / (self.ema_fast_n + 1)
            ks = 2 / (self.ema_slow_n + 1)
            self.ema_fast = self.ema_fast + kf * (close - self.ema_fast)
            self.ema_slow = self.ema_slow + ks * (close - self.ema_slow)

        if self.prev_close is None:
            tr = high - low
            self.atr = tr
        else:
            tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))
            if self.atr is None:
                self.atr = tr
            else:
                a = self.atr
                self.atr = (a * (self.atr_n - 1) + tr) / self.atr_n
        self.prev_close = close

        typical = (high + low + close) / 3.0
        self.vwap_num += typical * volume
        self.vwap_den += max(volume, 1e-9)

    @property
    def vwap(self) -> Optional[float]:
        if self.vwap_den <= 0:
            return None
        return self.vwap_num / self.vwap_den

    def channel(self) -> Tuple[Optional[float], Optional[float]]:
        if len(self.closes) < self.ch_n:
            return None, None
        recent_high = max(list(self.highs)[-self.ch_n:])
        recent_low = min(list(self.lows)[-self.ch_n:])
        return recent_low, recent_high

    def atr_z(self) -> Optional[float]:
        if len(self.closes) < max(100, self.atr_n + 1):
            return None
        ranges = [h - l for h, l in zip(list(self.highs)[-100:], list(self.lows)[-100:])]
        mean_range = sum(ranges) / max(len(ranges), 1)
        if mean_range <= 0 or self.atr is None:
            return None
        return self.atr / mean_range
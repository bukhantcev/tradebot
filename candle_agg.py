# candle_agg.py
from dataclasses import dataclass

@dataclass
class Candle:
    start_ms: int   # начало минуты (unix ms, округлённое вниз)
    open: float
    high: float
    low: float
    close: float
    volume: float

class MinuteAggregator:
    """
    Копит тики в текущую минуту. Когда приходит тик из НОВОЙ минуты —
    возвращает ЗАКРЫТУЮ свечу (Candle) предыдущей минуты.
    """
    def __init__(self):
        self.cur_minute_ms = None
        self.o = self.h = self.l = self.c = None
        self.v = 0.0

    @staticmethod
    def _minute_floor(ts_ms: int) -> int:
        return (ts_ms // 60000) * 60000

    def on_tick(self, price: float, volume: float, ts_ms: int) -> Candle | None:
        m = self._minute_floor(ts_ms)

        # первая точка
        if self.cur_minute_ms is None:
            self.cur_minute_ms = m
            self.o = self.h = self.l = self.c = float(price)
            self.v = float(volume)
            return None

        # новая минута → отдать закрытую свечу предыдущей
        if m > self.cur_minute_ms:
            closed = Candle(
                start_ms=self.cur_minute_ms,
                open=self.o, high=self.h, low=self.l, close=self.c, volume=self.v
            )
            # инициализируем новую минуту текущим тиком
            self.cur_minute_ms = m
            self.o = self.h = self.l = self.c = float(price)
            self.v = float(volume)
            return closed

        # всё ещё та же минута → обновляем
        p = float(price)
        self.c = p
        if p > self.h: self.h = p
        if p < self.l: self.l = p
        self.v += float(volume)
        return None
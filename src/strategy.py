# src/strategy.py
import threading
import time
from typing import Callable, List
from tinkoff.invest import CandleInterval, OrderDirection

from .tinkoff_sandbox import load_candles, quotation_to_float

INTERVAL_LABEL_TO_ENUM = {
    "1m": CandleInterval.CANDLE_INTERVAL_1_MIN,
    "5m": CandleInterval.CANDLE_INTERVAL_5_MIN,
}

class StrategyEngine:
    def __init__(
        self,
        sbx,
        figi: str,
        interval: CandleInterval = CandleInterval.CANDLE_INTERVAL_5_MIN,
        n_enter: int = 55,
        n_exit: int = 20,
        atr_n: int = 14,
        risk_pct: float = 0.01,
        atr_k: float = 2.0,
        notifier: Callable[[str], None] | None = None,
    ):
        self.sbx = sbx
        self.services = sbx.client
        self.figi = figi
        self.interval = interval
        self.n_enter = n_enter
        self.n_exit = n_exit
        self.atr_n = atr_n
        self.risk_pct = risk_pct
        self.atr_k = atr_k
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._in_pos_lots = 0
        self.logs: List[str] = []
        self._notify = notifier or (lambda _: None)

    def log(self, msg: str):
        self.logs.append(msg)
        self.logs = self.logs[-200:]
        self._notify(msg)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="strategy", daemon=True)
        self._thread.start()
        self.log(f"▶️ Стратегия запущена для {self.figi} {self.interval.name} | риск={self.risk_pct*100:.1f}%")

    def stop(self):
        self._stop.set()
        self.log("⏹ Стратегия остановлена")

    def status(self) -> str:
        running = self._thread.is_alive() if self._thread else False
        return f"figi={self.figi} running={running} lots_in_pos={self._in_pos_lots} risk={self.risk_pct*100:.1f}% interval={self._interval_label()}"

    def set_risk(self, pct: float):
        self.risk_pct = pct
        self.log(f"⚙️ Риск обновлён: {pct*100:.1f}%")

    def set_interval_label(self, label: str):
        if label not in INTERVAL_LABEL_TO_ENUM:
            self.log(f"Некорректный интервал: {label}")
            return
        self.interval = INTERVAL_LABEL_TO_ENUM[label]
        self.log(f"⏱ Интервал обновлён: {label}")

    def _interval_label(self) -> str:
        for k, v in INTERVAL_LABEL_TO_ENUM.items():
            if v == self.interval:
                return k
        return "5m"

    def _run(self):
        lot = self.sbx.get_lot_size(self.figi)
        self.log(f"Лот = {lot}")
        last_signal = None
        while not self._stop.is_set():
            try:
                candles = load_candles(self.services, self.figi, hours=96, interval=self.interval)
                if len(candles) < max(self.n_enter, self.n_exit, self.atr_n) + 2:
                    time.sleep(10)
                    continue
                highs = [quotation_to_float(c.high) for c in candles]
                lows = [quotation_to_float(c.low) for c in candles]
                closes = [quotation_to_float(c.close) for c in candles]
                upper = max(highs[-self.n_enter:])
                lower = min(lows[-self.n_exit:])
                last_close = closes[-1]
                atr = self._calc_atr(highs, lows, closes, self.atr_n)
                stop_price = last_close - self.atr_k * atr
                signal = "long" if last_close > upper else ("flat" if last_close < lower else last_signal)
                if (last_signal != "long") and (signal == "long"):
                    qty = self._calc_lots(last_close, stop_price, lot)
                    if qty > 0:
                        self.log(f"🟢 ENTRY long {qty} лот(ов) @~{last_close:.2f}, stop ~{stop_price:.2f}, ATR {atr:.2f}")
                        self.sbx.post_market_order(self.figi, qty, OrderDirection.ORDER_DIRECTION_BUY)
                        self._in_pos_lots = qty
                        last_signal = "long"
                    else:
                        self.log("ℹ️ Пропуск входа: размер позиции по риску = 0")
                        last_signal = "flat"
                elif (last_signal == "long") and (signal == "flat" or last_close < stop_price):
                    if self._in_pos_lots > 0:
                        self.log(f"🔴 EXIT {self._in_pos_lots} лот(ов) @~{last_close:.2f}")
                        self.sbx.post_market_order(self.figi, self._in_pos_lots, OrderDirection.ORDER_DIRECTION_SELL)
                        self._in_pos_lots = 0
                    last_signal = "flat"
                time.sleep(60)
            except Exception as e:
                self.log(f"❗️Ошибка стратегии: {e}")
                time.sleep(5)

    @staticmethod
    def _calc_atr(highs, lows, closes, n):
        trs = []
        for i in range(1, len(closes)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            trs.append(max(hl, hc, lc))
        return sum(trs[-n:]) / n

    def _calc_lots(self, price: float, stop_price: float, lot_size: int) -> int:
        port = self.sbx.get_total_rub()
        risk_rub = port * self.risk_pct
        per_lot_risk = max(price - stop_price, 0.01) * lot_size
        lots = int(risk_rub // per_lot_risk)
        return max(lots, 0)

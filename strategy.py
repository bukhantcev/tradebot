"""
Базовая стратегия + интеграция с OnlineModel.
Сигналы по закрытию 1m свечи. Фильтры:
- тренд (ema_fast vs ema_slow)
- импульс/ROC, спред-прокси
- ML prob в качестве веса (не жёсткий фильтр)
SL/TP: SL=1.5*ATR, TP=2*SL, трейл включается при +1*SL
"""
import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import pandas as pd

from config import RISK_PCT, SYMBOL
from features import load_recent_1m, compute_features, last_feature_row
from ml import OnlineModel

logger = logging.getLogger("STRATEGY")

@dataclass
class Signal:
    side: Optional[str]  # "Buy" | "Sell" | None
    reason: str
    sl: Optional[float]
    tp: Optional[float]
    atr: Optional[float]
    ts_ms: Optional[int]

class StrategyEngine:
    def __init__(self, risk_pct: float = RISK_PCT, symbol: str = SYMBOL):
        self.risk_pct = risk_pct
        self.symbol = symbol
        self.model = OnlineModel()
        self.cooldown_sec = 180
        self._last_trade_time = 0.0
        self._mode = "auto"  # "auto" | "trend" | "range"

    def set_mode(self, mode: str):
        if mode in ("auto","trend","range"):
            self._mode = mode

    async def on_kline_closed(self) -> Signal:
        df = await load_recent_1m(200, symbol=self.symbol)
        if len(df) < 60:
            return Signal(None, "warmup", None, None, None, None)
        dff = compute_features(df)
        f0 = last_feature_row(dff)
        if not f0:
            return Signal(None, "no_features", None, None, None, None)

        # update ML with previous step info (if possible)
        if len(dff) >= 2:
            prev = dff.iloc[-2]
            curr = dff.iloc[-1]
            try:
                self.model.update(
                    {"ts_ms": int(prev["ts_ms"]), "close": float(prev["close"]),
                     "ema_fast": float(prev["ema_fast"]), "ema_slow": float(prev["ema_slow"]),
                     "atr14": float(prev["atr14"]), "roc1": float(prev["roc1"]),
                     "spread_proxy": float(prev["spread_proxy"]), "vol_roll": float(prev["vol_roll"])},
                    float(curr["close"])
                )
            except Exception as e:
                logger.debug(f"[ML] update skip: {e}")

        prob_up = self.model.predict(f0)  # 0..1
        atr = max(f0["atr14"], 1e-8)
        close = f0["close"]
        ema_fast, ema_slow = f0["ema_fast"], f0["ema_slow"]
        trend_up = ema_fast > ema_slow
        roc = f0["roc1"]

        # cooldown
        now = time.time()
        if now - self._last_trade_time < self.cooldown_sec:
            return Signal(None, "cooldown", None, None, atr, f0["ts_ms"])

        # Decide side
        side = None
        reason = []
        if self._mode in ("auto", "trend"):
            if trend_up and prob_up > 0.52 and roc >= -0.001:
                side = "Buy"
                reason.append("trend_up+ml")
            elif (not trend_up) and prob_up < 0.48 and roc <= 0.001:
                side = "Sell"
                reason.append("trend_dn+ml")
        if side is None and self._mode in ("auto", "range"):
            # контртренд: возврат к EMA_slow при переотклонении
            dev = (close - ema_slow) / max(close, 1e-8)
            if dev > 0.004 and prob_up < 0.5:
                side = "Sell"; reason.append("revert_down")
            elif dev < -0.004 and prob_up > 0.5:
                side = "Buy"; reason.append("revert_up")

        if side is None:
            return Signal(None, "no_entry", None, None, atr, f0["ts_ms"])

        # SL/TP
        sl_mult = 1.5
        tp_mult_vs_sl = 2.0
        if side == "Buy":
            sl = close - sl_mult * atr
            tp = close + tp_mult_vs_sl * (close - sl)
        else:
            sl = close + sl_mult * atr
            tp = close - tp_mult_vs_sl * (sl - close)

        self._last_trade_time = now
        return Signal(side, "+".join(reason), sl, tp, atr, f0["ts_ms"])
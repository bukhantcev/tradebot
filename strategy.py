import logging
import time
from dataclasses import dataclass
from typing import Optional

from config import RISK_PCT, SYMBOL
from features import load_recent_1m, compute_features, last_feature_row
from ml import OnlineModel

log = logging.getLogger("STRATEGY")

@dataclass
class Signal:
    side: Optional[str]   # "Buy" | "Sell" | None
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
        self._mode = "auto"

    def set_mode(self, mode: str):
        if mode in ("auto","trend","range"):
            log.debug(f"[MODE] {self._mode} -> {mode}")
            self._mode = mode

    async def on_kline_closed(self) -> Signal:
        df = await load_recent_1m(200, symbol=self.symbol)
        if not df.empty:
            from datetime import datetime, timezone
            last_ts = int(df.iloc[-1]["ts_ms"]) // 1000
            log.debug(f"[BAR] last_closed_1m={datetime.fromtimestamp(last_ts, tz=timezone.utc).isoformat()}")
        log.debug(f"[BAR] loaded rows={len(df)}")
        if len(df) < 60:
            return Signal(None, "warmup", None, None, None, None)

        dff = compute_features(df)
        f0 = last_feature_row(dff)
        log.debug(f"[FEAT] last={f0}")

        # онлайн-апдейт модели на предыдущем шаге
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
                log.debug(f"[ML] update skip: {e}")

        prob_up = self.model.predict(f0)
        atr = max(f0["atr14"], 1e-8)
        close = f0["close"]
        ema_fast, ema_slow = f0["ema_fast"], f0["ema_slow"]
        trend_up = ema_fast > ema_slow
        roc = f0["roc1"]
        log.debug(f"[ML] p_up={prob_up:.3f} trend_up={trend_up} roc={roc:.4f}")

        now = time.time()
        if now - self._last_trade_time < self.cooldown_sec:
            return Signal(None, "cooldown", None, None, atr, f0["ts_ms"])

        side = None
        reasons = []
        if self._mode in ("auto","trend"):
            if trend_up and prob_up > 0.52 and roc >= -0.001:
                side = "Buy"; reasons.append("trend_up+ml")
            elif (not trend_up) and prob_up < 0.48 and roc <= 0.001:
                side = "Sell"; reasons.append("trend_dn+ml")
        if side is None and self._mode in ("auto","range"):
            dev = (close - ema_slow) / max(close, 1e-8)
            if dev > 0.004 and prob_up < 0.5:
                side = "Sell"; reasons.append("revert_down")
            elif dev < -0.004 and prob_up > 0.5:
                side = "Buy"; reasons.append("revert_up")

        if side is None:
            log.debug("[DECIDE] no_entry")
            return Signal(None, "no_entry", None, None, atr, f0["ts_ms"])

        sl_mult = 1.5
        tp_mult_vs_sl = 2.0
        if side == "Buy":
            sl = close - sl_mult * atr
            tp = close + tp_mult_vs_sl * (close - sl)
        else:
            sl = close + sl_mult * atr
            tp = close - tp_mult_vs_sl * (sl - close)

        self._last_trade_time = now
        reason = "+".join(reasons)
        log.info(f"[SIGNAL] side={side} reason={reason} close={close:.2f} atr={atr:.2f} sl={sl:.2f} tp={tp:.2f}")
        return Signal(side, reason, sl, tp, atr, f0["ts_ms"])
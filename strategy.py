# strategy.py
import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any
import asyncio
from config import SYMBOL, RISK_PCT
from features import load_recent_1m, compute_features, last_feature_row
from llm import ask_model

log = logging.getLogger("STRATEGY")


@dataclass
class Signal:
    side: Optional[str]   # "Buy" | "Sell" | None
    reason: str
    sl: Optional[float]
    tp: Optional[float]
    atr: Optional[float]
    ts_ms: Optional[int]
    prev_high: Optional[float] = None
    prev_low: Optional[float] = None


class StrategyEngine:
    """
    Минималистичная стратегия:
    - Берёт закрытые 1m бары из БД
    - Считает базовые фичи
    - Делает запрос к LLM и по ответу формирует сигнал
    - SL/TP — из ATR (настраиваемые множители)
    Логи только по факту: [SIGNAL] и [DECIDE] (сам запрос/ответ ЛЛМ логируются в llm.py).
    """
    def __init__(
        self,
        risk_pct: float = RISK_PCT,
        symbol: str = SYMBOL,
        sl_mult: float = 1.5,
        tp_vs_sl: float = 2.0,
        cooldown_sec: int = 180,
        notifier: Optional[Any] = None,  # опционально: объект с методом notify(str)
    ):
        self.risk_pct = risk_pct
        self.symbol = symbol
        self.sl_mult = sl_mult
        self.tp_vs_sl = tp_vs_sl
        self.cooldown_sec = cooldown_sec
        self._last_trade_time = 0.0
        self._notifier = notifier

    def set_notifier(self, notifier: Any):
        self._notifier = notifier

    async def on_kline_closed(self) -> Signal:
        # 1) История только из закрытых минут
        df = await load_recent_1m(200, symbol=self.symbol)
        prev_high = None
        prev_low = None
        if len(df) < 60:
            # короткий факт-лог — без шума
            log.info("[SKIP] warmup (<60 closed 1m bars)")
            log.debug(f"[SIGNAL][HL] prevH={prev_high} prevL={prev_low}")
            return Signal(None, "warmup", None, None, None, None, prev_high, prev_low)

        # Экстремы предыдущей ЗАКРЫТОЙ 1m свечи
        prev_high = float(df.iloc[-1]["high"])
        prev_low = float(df.iloc[-1]["low"])

        # 2) Признаки
        dff = compute_features(df)
        f0: Dict[str, Any] = last_feature_row(dff)
        if not f0:
            log.debug(f"[SIGNAL][HL] prevH={prev_high} prevL={prev_low}")
            return Signal(None, "no_features", None, None, None, None, prev_high, prev_low)

        # Ключевой лог «сигнал/срез фич» — компактно
        log.info(f"[SIGNAL] c={f0['close']:.2f} emaF={f0['ema_fast']:.2f} emaS={f0['ema_slow']:.2f} atr={f0['atr14']:.2f} prevH={prev_high:.2f} prevL={prev_low:.2f}")
        if self._notifier:
            try:
                await self._notifier.notify(
                    f"📊 Signal\nc={f0['close']:.2f}  emaF={f0['ema_fast']:.2f}  emaS={f0['ema_slow']:.2f}  atr={f0['atr14']:.2f}\nprevH {prev_high:.2f} / prevL {prev_low:.2f}"
                )
            except Exception:
                pass

        # 3) Анти-спам: общий кулдаун между входами
        now = time.time()
        if now - self._last_trade_time < self.cooldown_sec:
            log.debug(f"[SIGNAL][HL] prevH={prev_high} prevL={prev_low}")
            return Signal(None, "cooldown", None, None, float(f0["atr14"]), int(f0["ts_ms"]), prev_high, prev_low)

        # 4) Вызов LLM (запрос/ответ логируются в llm.py как [LLM→]/[LLM←])
        ctx = {
            "symbol": self.symbol,
            "risk_pct": self.risk_pct,
        }
        decision = await ask_model(
            features={
                "ts_ms": f0["ts_ms"],
                "close": f0["close"],
                "ema_fast": f0["ema_fast"],
                "ema_slow": f0["ema_slow"],
                "atr14": f0["atr14"],
                "roc1": f0["roc1"],
                "spread_proxy": f0["spread_proxy"],
                "vol_roll": f0["vol_roll"],
            },
            ctx=ctx,
        )

        action_raw = str(decision.get("decision", "Hold"))
        reason = str(decision.get("reason", "") or "").strip()
        action = action_raw.capitalize()
        if action not in ("Buy", "Sell"):
            log.info(f"[DECIDE] Hold | {reason}")
            if self._notifier:
                try:
                    await self._notifier.notify(f"🤖 LLM: Hold • {reason or 'no reason'}")
                except Exception:
                    pass
            log.debug(f"[SIGNAL][HL] prevH={prev_high} prevL={prev_low}")
            return Signal(None, "hold", None, None, float(f0["atr14"]), int(f0["ts_ms"]), prev_high, prev_low)

        # 5) Построение SL/TP из ATR
        close = float(f0["close"])
        atr = max(float(f0["atr14"]), 1e-8)
        sl_mult = float(self.sl_mult)
        tp_vs_sl = float(self.tp_vs_sl)

        if action == "Buy":
            sl = close - sl_mult * atr
            tp = close + tp_vs_sl * (close - sl)
        else:  # Sell
            sl = close + sl_mult * atr
            tp = close - tp_vs_sl * (sl - close)

        log.info(f"[DECIDE] {action} | sl={sl:.2f} tp={tp:.2f} • {reason}")
        if self._notifier:
            try:
                arrow = "🟢 Buy" if action == "Buy" else "🔴 Sell"
                await self._notifier.notify(f"🤖 LLM: {arrow}\nSL {sl:.2f} / TP {tp:.2f}\n{('💬 ' + reason) if reason else ''}")
            except Exception:
                pass

        # фиксируем кулдаун
        self._last_trade_time = now

        log.debug(f"[SIGNAL][HL] prevH={prev_high} prevL={prev_low}")
        return Signal(
            side=action,
            reason=reason or "llm",
            sl=float(sl),
            tp=float(tp),
            atr=float(atr),
            ts_ms=int(f0["ts_ms"]),
            prev_high=prev_high,
            prev_low=prev_low,
        )
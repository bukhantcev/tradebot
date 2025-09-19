# strategy.py
import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any
import asyncio
import math

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
    prev_open: Optional[float] = None
    prev_close: Optional[float] = None


class StrategyEngine:
    """
    Минималистичная стратегия:
    - Берёт закрытые 1m бары из БД
    - Считает базовые фичи
    - Делает запрос к LLM и по ответу формирует сигнал
    - SL/TP — из ATR и/или тела предыдущей свечи
    Логи только по факту: [SIGNAL] и [DECIDE] (сам запрос/ответ ЛЛМ логируются в llm.py).
    """
    def __init__(
        self,
        risk_pct: float = RISK_PCT,
        symbol: str = SYMBOL,
        tick_size: float = 0.1,
        sl_mult: float = 1.5,
        tp_vs_sl: float = 2.0,
        cooldown_sec: int = 0,
        notifier: Optional[Any] = None,  # опционально: объект с методом notify(str)
    ):
        self.risk_pct = risk_pct
        self.symbol = symbol
        self.tick_size = float(tick_size)
        self.sl_mult = sl_mult
        self.tp_vs_sl = tp_vs_sl
        self.cooldown_sec = cooldown_sec
        self._last_trade_time = 0.0
        self._notifier = notifier

    # --- ticks helpers
    def _to_tick_down(self, price: float) -> float:
        t = self.tick_size or 0.1
        return math.floor(price / t) * t

    def _to_tick_up(self, price: float) -> float:
        t = self.tick_size or 0.1
        return math.ceil(price / t) * t

    def set_notifier(self, notifier: Any):
        self._notifier = notifier

    async def on_kline_closed(self) -> Signal:
        log.debug("[ON_CLOSE][ENTER]")
        # Cooldown (если задан)
        if self.cooldown_sec > 0:
            left = max(0.0, self.cooldown_sec - (time.time() - self._last_trade_time))
            if left > 0:
                log.debug(f"[ON_CLOSE][COOLDOWN] skip {left:.1f}s remain")
                return Signal(None, "cooldown", None, None, None, None, None, None, None, None)

        # 1) История только из закрытых минут
        try:
            df = await load_recent_1m(200, symbol=self.symbol)
        except Exception as e:
            log.exception(f"[ON_CLOSE][LOAD_ERR] {e}")
            return Signal(None, f"load_error:{e}", None, None, None, None)

        if df is None or len(df) < 60:
            log.info("[SKIP] warmup (<60 closed 1m bars)")
            return Signal(None, "warmup", None, None, None, None)

        # Экстремы предыдущей ЗАКРЫТОЙ 1m свечи
        try:
            prev_high = float(df.iloc[-1]["high"])
            prev_low = float(df.iloc[-1]["low"])
            prev_open = float(df.iloc[-1]["open"])
            prev_close = float(df.iloc[-1]["close"])
        except Exception as e:
            log.exception(f"[ON_CLOSE][HL_ERR] {e}")
            return Signal(None, f"prev_hl_error:{e}", None, None, None, None)

        # 2) Признаки
        try:
            dff = compute_features(df)
            f0: Dict[str, Any] = last_feature_row(dff)
        except Exception as e:
            log.exception(f"[FEAT][ERR] {e}")
            return Signal(None, f"features_error:{e}", None, None, None, None, prev_high, prev_low, prev_open, prev_close)

        if not f0:
            return Signal(None, "no_features", None, None, None, None, prev_high, prev_low, prev_open, prev_close)

        # Ключевой лог «сигнал/срез фич» — компактно
        log.info(f"[SIGNAL] c={f0['close']:.2f} emaF={f0['ema_fast']:.2f} emaS={f0['ema_slow']:.2f} atr={f0['atr14']:.2f} prevH={prev_high:.2f} prevL={prev_low:.2f}")
        if self._notifier:
            try:
                await self._notifier.notify(
                    f"📊 Signal\nc={f0['close']:.2f}  emaF={f0['ema_fast']:.2f}  emaS={f0['ema_slow']:.2f}  atr={f0['atr14']:.2f}\nprevH {prev_high:.2f} / prevL {prev_low:.2f}"
                )
            except Exception:
                pass

        # 4) Вызов LLM (подробные DEBUG — собираем запрос, отправляем, парсим)
        ctx = {"symbol": self.symbol, "risk_pct": self.risk_pct}
        feats_dbg = (
            f"ts={int(f0.get('ts_ms', 0))} "
            f"c={float(f0.get('close', 0)):.2f} "
            f"emaF={float(f0.get('ema_fast', 0)):.2f} "
            f"emaS={float(f0.get('ema_slow', 0)):.2f} "
            f"atr14={float(f0.get('atr14', 0)):.2f} "
            f"roc1={float(f0.get('roc1', 0)):.4f} "
            f"spr={float(f0.get('spread_proxy', 0)):.4f} "
            f"vol={float(f0.get('vol_roll', 0)):.5f}"
        )
        log.debug(f"[LLM][PREP] ctx={ctx} | feats=({feats_dbg})")
        log.info("[LLM][CALL] start")
        t0 = time.time()
        try:
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
            dt_ms = (time.time() - t0) * 1000.0
            resp_str = str(decision)
            if len(resp_str) > 300:
                resp_str = resp_str[:300] + "…"
            log.debug(f"[LLM][OK] {dt_ms:.1f} ms | resp={resp_str}")
            log.info(f"[LLM][CALL][OK] {dt_ms:.1f} ms decision={str(decision.get('decision','')).strip() or 'n/a'}")
        except Exception as e:
            dt_ms = (time.time() - t0) * 1000.0
            log.exception(f"[LLM][ERR] {dt_ms:.1f} ms | {e}")
            if self._notifier:
                try:
                    await self._notifier.notify(f"⚠️ ИИ ошибка: {e}")
                except Exception:
                    pass
            return Signal(None, f"llm_error:{e}", None, None, float(f0.get("atr14", 0.0)), int(f0.get("ts_ms", 0)), prev_high, prev_low, prev_open, prev_close)

        action_raw = str(decision.get("decision", "Hold"))
        reason = str(decision.get("reason", "") or "").strip()
        action = action_raw.capitalize()
        if action not in ("Buy", "Sell"):
            log.info(f"[DECIDE] Hold | {reason}")
            if self._notifier:
                try:
                    await self._notifier.notify(f"🤖 ИИ: Hold • {reason or 'no reason'}")
                except Exception:
                    pass
            return Signal(None, "hold", None, None, float(f0["atr14"]), int(f0["ts_ms"]), prev_high, prev_low, prev_open, prev_close)

        # 5) Построение SL/TP относительно тела предыдущей свечи,
        #    + защита, чтобы TP гарантированно был по правильную сторону от LastPrice.
        close = float(f0["close"])
        atr = max(float(f0["atr14"]), 1e-8)
        sl_mult = float(self.sl_mult)

        body_high = max(prev_open, prev_close)
        body_low = min(prev_open, prev_close)
        tick = max(self.tick_size, 1e-9)
        tp_nudges = 4 * tick  # смещаем TP на 4 тика внутрь тела — по просьбе

        if action == "Buy":
            sl = close - sl_mult * atr
            desired_tp = max(body_high - tp_nudges, close + tick)
            tp = self._to_tick_up(desired_tp)
            if tp <= close:
                tp = self._to_tick_up(close + tick)
            log.info(f"[LLM][TPSL][MARK] Buy TP adjusted for LastPrice: sl={sl:.2f} tp={tp:.2f}")
        else:  # Sell
            sl = close + sl_mult * atr
            desired_tp = min(body_low + tp_nudges, close - tick)
            tp = self._to_tick_down(desired_tp)
            if tp >= close:
                tp = self._to_tick_down(close - tick)
            log.info(f"[LLM][TPSL][MARK] Sell TP adjusted for LastPrice: sl={sl:.2f} tp={tp:.2f}")

        log.info(f"[DECIDE] {action} | sl={sl:.2f} tp={tp:.2f} • {reason}")
        if self._notifier:
            try:
                arrow = "🟢 Buy" if action == "Buy" else "🔴 Sell"
                await self._notifier.notify(f"🤖 ИИ: {arrow}\nSL {sl:.2f} / TP {tp:.2f}\n{('💬 ' + reason) if reason else ''}")
            except Exception:
                pass

        # фиксируем кулдаун
        self._last_trade_time = time.time()

        return Signal(
            side=action,
            reason=reason or "llm",
            sl=float(sl),
            tp=float(tp),
            atr=float(atr),
            ts_ms=int(f0["ts_ms"]),
            prev_high=prev_high,
            prev_low=prev_low,
            prev_open=prev_open,
            prev_close=prev_close,
        )
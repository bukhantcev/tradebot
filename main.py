import os
import json
import sqlite3
import math
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
import talib
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Bot
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from pybit.unified_trading import HTTP

# наша аналитическая БД (оставляю твой модуль)
from test_db import AnalyticsDB, RiskConfig

UTC = timezone.utc

# ---------------------- ENV / CONFIG ----------------------
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Bybit API
BYBIT_API_KEY_MAIN = os.getenv("BYBIT_API_KEY_MAIN", "")
BYBIT_API_SECRET_MAIN = os.getenv("BYBIT_API_SECRET_MAIN", "")
BYBIT_API_KEY_TEST = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET_TEST = os.getenv("BYBIT_API_SECRET", "")

# Telegram
if os.getenv("HOST_ROLE", "local") == "local":
    TELEGRAM_BOT_TOKEN = os.getenv("TG_TOKEN_LOCAL", "")
else:
    TELEGRAM_BOT_TOKEN = os.getenv("TG_TOKEN_SERVER", "")
TELEGRAM_CHAT_ID = os.getenv("TG_ADMIN_CHAT_ID", "")

# Торговые параметры
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
BASE_LEVERAGE = int(os.getenv("BASE_LEVERAGE", "10"))
MAX_NOTIONAL_USDT = float(os.getenv("MAX_NOTIONAL_USDT", "10"))

IOC_TICKS_TRIES = int(os.getenv("IOC_TICKS_TRIES", "2"))
SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", "10"))
SLP_BPS_STEPS = [float(x) for x in os.getenv("SLP_BPS_STEPS", "5,10,15,25").split(",")]

MAX_SPREAD_TICKS = float(os.getenv("MAX_SPREAD_TICKS", "3"))
MAX_IOC_SLIPPAGE_BPS = float(os.getenv("MAX_IOC_SLIPPAGE_BPS", "25"))
ATR_PCT_MIN = float(os.getenv("ATR_PCT_MIN", "0.0015"))
ATR_PCT_BOOST = float(os.getenv("ATR_PCT_BOOST", "0.006"))
RISK_USDT = float(os.getenv("RISK_USDT", "0"))

# --- позицион-менеджмент (без кулдауна) ---
PART_TP_FRAC      = float(os.getenv("PART_TP_FRAC", "0.3333"))  # доля позиции на частичный TP
PART_TP_ATR_MULT  = float(os.getenv("PART_TP_ATR_MULT", "0.5")) # цель частичного TP в ATR
TRAIL_ARM_ATR     = float(os.getenv("TRAIL_ARM_ATR", "1.0"))    # когда включать трейлинг (в ATR от входа)
TRAIL_FROM_SL_MIN = float(os.getenv("TRAIL_FROM_SL_MIN", "2"))  # минимум: 2 тика
SR_DIST_ATR_MIN   = float(os.getenv("SR_DIST_ATR_MIN", "0.25")) # мин. дистанция до S/R в ATR

# Настройка TP/SL по инструментам (можно править в .env/коде)
PER_SYMBOL_TPSL = {
    "BTCUSDT": {"mode": "auto"},
    "ETHUSDT": {"mode": "atr", "sl_atr": 1.2, "tp_atr": 2.0, "atr_len": 14},
    "XRPUSDT": {"mode": "abs", "sl_abs": 0.003, "tp_abs": 0.006},
    "SOLUSDT": {"mode": "rr", "sl_ticks": 15, "rr": 1.8},
}

# Инвертировать торговые сигналы (BUY<->SELL)
INVERT_SIGNALS = str(os.getenv("INVERT_SIGNALS", "0")).strip().lower() not in ("0", "", "false", "no", "off")

# ---------------------- TP/SL HELPER ----------------------
def compute_tpsl_for_symbol(symbol: str, side: str, entry: float,
                            atr_value: Optional[float], meta: dict) -> Tuple[Optional[float], Optional[float]]:
    """
    Унифицированный расчёт SL/TP:
      - mode=auto: выбор между ATR / тиками / абсолютом в зависимости от волатильности
      - mode=ticks|atr|abs|rr: жёсткие правила
    """
    conf = PER_SYMBOL_TPSL.get(symbol)
    if not conf:
        return (None, None)
    mode = str(conf.get("mode", "auto")).lower()
    tick_size = (meta or {}).get("tick_size", 0.1) or 0.1
    s = str(side).lower()

    if mode in ("", "auto", None, "auto"):
        atr_pct = (float(atr_value) / entry) if (atr_value and entry) else 0.0
        sl_dist = tp_dist = None
        if atr_pct >= 0.003:
            sl_mult = conf.get("sl_atr", 1.2)
            tp_mult = conf.get("tp_atr", 1.8)
            if atr_value and atr_value > 0:
                sl_dist = sl_mult * atr_value
                tp_dist = tp_mult * atr_value
        elif atr_pct >= 0.0015:
            base_ticks = max(5, int(round((0.8 * (atr_value or 0.0)) / tick_size)))
            rr = conf.get("rr", 1.6)
            sl_dist = base_ticks * tick_size
            tp_dist = int(round(base_ticks * rr)) * tick_size
        else:
            sl_abs_frac = conf.get("sl_abs_frac", 0.0010)
            tp_abs_frac = conf.get("tp_abs_frac", 0.0018)
            sl_dist = max(2 * tick_size, sl_abs_frac * entry)
            tp_dist = max(3 * tick_size, tp_abs_frac * entry)
        if sl_dist is not None and tp_dist is not None:
            return (entry - sl_dist, entry + tp_dist) if s == "buy" else (entry + sl_dist, entry - tp_dist)

    if mode == "ticks":
        sl_ticks = conf.get("sl_ticks")
        tp_ticks = conf.get("tp_ticks")
        if sl_ticks is None or tp_ticks is None:
            return (None, None)
        sl_dist = sl_ticks * tick_size
        tp_dist = tp_ticks * tick_size
        return (entry - sl_dist, entry + tp_dist) if s == "buy" else (entry + sl_dist, entry - tp_dist)

    if mode == "atr":
        sl_atr = conf.get("sl_atr")
        tp_atr = conf.get("tp_atr")
        if sl_atr is None or tp_atr is None or not atr_value or atr_value <= 0:
            return (None, None)
        sl_dist = sl_atr * atr_value
        tp_dist = tp_atr * atr_value
        return (entry - sl_dist, entry + tp_dist) if s == "buy" else (entry + sl_dist, entry - tp_dist)

    if mode == "abs":
        sl_abs = conf.get("sl_abs")
        tp_abs = conf.get("tp_abs")
        if sl_abs is None or tp_abs is None:
            return (None, None)
        return (entry - sl_abs, entry + tp_abs) if s == "buy" else (entry + sl_abs, entry - tp_abs)

    if mode == "rr":
        rr = conf.get("rr")
        if rr is None:
            return (None, None)
        sl_dist = None
        if conf.get("sl_ticks") is not None:
            sl_dist = conf["sl_ticks"] * tick_size
        elif conf.get("sl_abs") is not None:
            sl_dist = conf["sl_abs"]
        elif conf.get("sl_atr") is not None and atr_value and atr_value > 0:
            sl_dist = conf["sl_atr"] * atr_value
        if sl_dist is None:
            return (None, None)
        tp_dist = sl_dist * rr
        return (entry - sl_dist, entry + tp_dist) if s == "buy" else (entry + sl_dist, entry - tp_dist)

    return (None, None)

# ---------------------- TYPES ----------------------
class TradingMode(Enum):
    TESTNET = "testnet"
    MAINNET = "mainnet"

class OrderSide(Enum):
    BUY = "Buy"
    SELL = "Sell"

class BotStatus(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    ERROR = "error"

@dataclass
class TradeSignal:
    side: OrderSide
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    strategy_name: str

@dataclass
class Position:
    symbol: str
    side: str
    size: float
    entry_price: float
    unrealized_pnl: float
    percentage: float

# ---------------------- helpers for inversion ----------------------
def invert_side(side: OrderSide) -> OrderSide:
    return OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY

def invert_tpsl(entry: float, orig_side: OrderSide,
                sl: Optional[float], tp: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    """
    Инвертируем расположение SL/TP вокруг точки входа:
      BUY(sl=entry-X, tp=entry+Y) -> SELL(sl=entry+X, tp=entry-Y)
      SELL(sl=entry+X, tp=entry-Y) -> BUY(sl=entry-X, tp=entry+Y)
    Если какого-то уровня нет — возвращаем None.
    """
    if sl is None and tp is None:
        return (None, None)

    def dist(a, b):
        try:
            return abs(float(a) - float(b))
        except Exception:
            return None

    if orig_side == OrderSide.BUY:
        d_sl = dist(entry, sl) if sl is not None else None
        d_tp = dist(tp, entry) if tp is not None else None
        new_sl = (entry + d_sl) if d_sl is not None else None
        new_tp = (entry - d_tp) if d_tp is not None else None
        return (new_sl, new_tp)
    else:
        d_sl = dist(sl, entry) if sl is not None else None
        d_tp = dist(entry, tp) if tp is not None else None
        new_sl = (entry - d_sl) if d_sl is not None else None
        new_tp = (entry + d_tp) if d_tp is not None else None
        return (new_sl, new_tp)

# ---------------------- ANALYZER / STRATEGIES ----------------------
class TechnicalAnalyzer:
    def analyze_market(self, df: pd.DataFrame) -> Dict:
        """Возвращает индикаторы и бинарные сигналы."""
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values

        rsi = talib.RSI(close, timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(close)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
        ema_fast = talib.EMA(close, timeperiod=9)
        ema_slow = talib.EMA(close, timeperiod=21)
        atr = talib.ATR(high, low, close, timeperiod=14)

        price = float(close[-1])
        atr_val = float(atr[-1]) if atr[-1] == atr[-1] else 0.0  # NaN-safe
        atr_pct = (atr_val / price) if price > 0 else 0.0

        # детектор «супер-спайка» объёма — сильные новости/шипы пропускаем
        avg_vol = float(np.mean(volume[-20:])) if len(volume) >= 20 else float(np.mean(volume))
        super_spike = bool(avg_vol > 0 and volume[-1] > 2.5 * avg_vol)

        signals = {
            "rsi_oversold": bool(rsi[-1] < 30),
            "rsi_overbought": bool(rsi[-1] > 70),
            "macd_bullish": bool(macd[-1] > macd_signal[-1] and macd_hist[-1] > 0),
            "macd_bearish": bool(macd[-1] < macd_signal[-1] and macd_hist[-1] < 0),
            "price_above_ema_fast": bool(price > float(ema_fast[-1])),
            "price_below_ema_fast": bool(price < float(ema_fast[-1])),
            "bb_oversold": bool(price < float(bb_lower[-1])),
            "bb_overbought": bool(price > float(bb_upper[-1])),
            "volume_spike": bool(volume[-1] > float(avg_vol * 1.5)),
            "atr": atr_val,
            "atr_pct_ok": bool(0.0015 <= atr_pct <= 0.0100),
            "super_spike": super_spike,
        }
        # --- простые S/R по локальным экстремумам за N свечей ---
        lookback = min(len(high), 50)
        sr_hi = np.max(high[-lookback:]) if lookback > 0 else float('nan')
        sr_lo = np.min(low[-lookback:])  if lookback > 0 else float('nan')

        dist_up = abs(sr_hi - price) if sr_hi == sr_hi else float('inf')
        dist_dn = abs(price - sr_lo) if sr_lo == sr_lo else float('inf')
        min_dist = min(dist_up, dist_dn)

        # требуем минимум 0.25*ATR до ближайшего уровня
        sr_ok = bool(atr_val > 0 and min_dist >= SR_DIST_ATR_MIN * atr_val)

        signals["sr_ok"] = sr_ok
        signals["sr_hi"] = float(sr_hi) if sr_hi == sr_hi else None
        signals["sr_lo"] = float(sr_lo) if sr_lo == sr_lo else None

        return {
            "price": price,
            "signals": signals,
            "indicators": {
                "rsi": float(rsi[-1]),
                "macd": float(macd[-1]),
                "ema_fast": float(ema_fast[-1]),
                "ema_slow": float(ema_slow[-1]),
                "bb_upper": float(bb_upper[-1]),
                "bb_lower": float(bb_lower[-1]),
                "atr": atr_val,
                "atr_pct": atr_pct,
            },
        }

class ScalpingStrategies:
    def __init__(self, analyzer: TechnicalAnalyzer):
        self.analyzer = analyzer

    def rsi_mean_reversion(self, market: Dict) -> Optional[TradeSignal]:
        s = market["signals"]
        p = market["price"]
        atr = s["atr"]

        # только в зоне «не крайностей» — ловим откат, но не ножи
        rsi = market["indicators"]["rsi"]
        if 38 <= rsi <= 62 and s["volume_spike"] and s["atr_pct_ok"] and not s["super_spike"] and s.get("sr_ok"):
            if s["price_above_ema_fast"] and not s["bb_overbought"]:
                return TradeSignal(
                    side=OrderSide.BUY,
                    entry_price=p,
                    stop_loss=p - 1.0 * atr,
                    take_profit=p + 1.5 * atr,
                    confidence=0.68,
                    strategy_name="rsi_mean_reversion_buy",
                )
            if s["price_below_ema_fast"] and not s["bb_oversold"]:
                return TradeSignal(
                    side=OrderSide.SELL,
                    entry_price=p,
                    stop_loss=p + 1.0 * atr,
                    take_profit=p - 1.5 * atr,
                    confidence=0.68,
                    strategy_name="rsi_mean_reversion_sell",
                )
        return None

    def macd_momentum(self, market: Dict) -> Optional[TradeSignal]:
        s = market["signals"]
        p = market["price"]
        atr = s["atr"]
        if s["atr_pct_ok"] and not s["super_spike"] and s.get("sr_ok"):
            if s["macd_bullish"] and s["price_above_ema_fast"]:
                return TradeSignal(
                    side=OrderSide.BUY,
                    entry_price=p,
                    stop_loss=p - 1.0 * atr,
                    take_profit=p + 1.5 * atr,
                    confidence=0.62,
                    strategy_name="macd_momentum_buy",
                )
            if s["macd_bearish"] and s["price_below_ema_fast"]:
                return TradeSignal(
                    side=OrderSide.SELL,
                    entry_price=p,
                    stop_loss=p + 1.0 * atr,
                    take_profit=p - 1.5 * atr,
                    confidence=0.62,
                    strategy_name="macd_momentum_sell",
                )
        return None

    def bb_breakout(self, market: Dict) -> Optional[TradeSignal]:
        s = market["signals"]
        p = market["price"]
        atr = s["atr"]
        if s["atr_pct_ok"] and s["volume_spike"] and not s["super_spike"] and s.get("sr_ok"):
            if s["bb_overbought"]:
                return TradeSignal(
                    side=OrderSide.BUY,
                    entry_price=p,
                    stop_loss=p - 2.0 * atr,
                    take_profit=p + 1.0 * atr,
                    confidence=0.52,
                    strategy_name="bb_breakout_buy",
                )
            if s["bb_oversold"]:
                return TradeSignal(
                    side=OrderSide.SELL,
                    entry_price=p,
                    stop_loss=p + 2.0 * atr,
                    take_profit=p - 1.0 * atr,
                    confidence=0.52,
                    strategy_name="bb_breakout_sell",
                )
        return None

    def get_best_signal(self, market: Dict) -> Optional[TradeSignal]:
        cands: List[TradeSignal] = []
        for fn in (self.rsi_mean_reversion, self.macd_momentum, self.bb_breakout):
            sig = fn(market)
            if sig:
                cands.append(sig)
        if not cands:
            return None
        best = max(cands, key=lambda x: x.confidence)
        risk = abs(best.entry_price - best.stop_loss)
        profit = abs(best.take_profit - best.entry_price)
        if risk <= 0 or (profit / risk) < 1.2:
            return None
        return best

# ---------- Инвертирующая обёртка ----------
class InversionProxyStrategies:
    """Обёртка стратегий: инвертирует сторону и зеркалит SL/TP."""
    def __init__(self, analyzer: TechnicalAnalyzer, base: ScalpingStrategies):
        self.analyzer = analyzer
        self.base = base

    def _invert_signal(self, sig: Optional[TradeSignal]) -> Optional[TradeSignal]:
        if not sig:
            return None
        new_side = invert_side(sig.side)
        new_sl, new_tp = invert_tpsl(sig.entry_price, sig.side, sig.stop_loss, sig.take_profit)
        return TradeSignal(
            side=new_side,
            entry_price=sig.entry_price,
            stop_loss=new_sl if new_sl is not None else sig.stop_loss,
            take_profit=new_tp if new_tp is not None else sig.take_profit,
            confidence=sig.confidence,
            strategy_name=f"INV[{sig.strategy_name}]",
        )

    def rsi_mean_reversion(self, market: Dict) -> Optional[TradeSignal]:
        return self._invert_signal(self.base.rsi_mean_reversion(market))

    def macd_momentum(self, market: Dict) -> Optional[TradeSignal]:
        return self._invert_signal(self.base.macd_momentum(market))

    def bb_breakout(self, market: Dict) -> Optional[TradeSignal]:
        return self._invert_signal(self.base.bb_breakout(market))

    def get_best_signal(self, market: Dict) -> Optional[TradeSignal]:
        return self._invert_signal(self.base.get_best_signal(market))

# ---------------------- BYBIT TRADER ----------------------
class BybitTrader:
    def __init__(self, api_key: str, api_secret: str, testnet: bool):
        self.client = HTTP(api_key=api_key, api_secret=api_secret, testnet=testnet)
        self.testnet = testnet

    # ---- helpers ----
    def _safe_float(self, v, default=0.0) -> float:
        try:
            if v is None or v == "":
                return float(default)
            return float(v)
        except Exception:
            return float(default)

    def _tick_decimals(self, tick: float) -> int:
        s = f"{tick}"
        return len(s.split(".")[1]) if "." in s else 0

    # ---- meta / state ----
    def get_symbol_meta(self, symbol: str) -> dict:
        """tickSize, qtyStep, minOrderQty, minOrderAmt"""
        try:
            res = self.client.get_instruments_info(category="linear", symbol=symbol)
            if res.get("retCode") == 0 and res.get("result", {}).get("list"):
                it = res["result"]["list"][0]
                lot = it.get("lotSizeFilter", {})
                pf = it.get("priceFilter", {})
                return {
                    "qty_step": self._safe_float(lot.get("qtyStep"), 0.001),
                    "min_qty": self._safe_float(lot.get("minOrderQty"), 0.001),
                    "min_notional": self._safe_float(lot.get("minOrderAmt"), 0.0),
                    "tick_size": self._safe_float(pf.get("tickSize"), 0.1) or 0.1,
                }
        except Exception as e:
            logging.error(f"meta error: {e}")
        return {"qty_step": 0.001, "min_qty": 0.001, "min_notional": 0.0, "tick_size": 0.1}

    def normalize_qty(self, symbol: str, qty: float) -> float:
        meta = self.get_symbol_meta(symbol)
        step = meta["qty_step"] or 0.001
        minq = meta["min_qty"] or step
        import math
        norm = math.floor(float(qty) / float(step)) * float(step)
        if norm < minq:
            norm = minq
        return float(f"{norm:.8f}")

    def normalize_price(self, symbol: str, price: float, side: Optional[str] = None, adjust: bool = False) -> float:
        """Квантование цены по тик-сайзу. adjust=True — сдвиг на 1 тик внутрь в пользу исполнения IOC."""
        meta = self.get_symbol_meta(symbol)
        tick = meta.get("tick_size", 0.1) or 0.1
        dec = self._tick_decimals(tick)

        steps = round(float(price) / tick)
        norm = steps * tick

        if adjust and side:
            if str(side).lower() == "buy":
                norm = (int(round(norm / tick)) + 1) * tick
            else:
                norm = (int(round(norm / tick)) - 1) * tick

        return float(f"{norm:.{dec}f}")

    def get_last_price(self, symbol: str) -> Optional[float]:
        try:
            r = self.client.get_tickers(category="linear", symbol=symbol)
            if r.get("retCode") == 0:
                lst = (r.get("result") or {}).get("list") or []
                if lst:
                    return self._safe_float(lst[0].get("lastPrice"))
        except Exception:
            pass
        return None

    def get_order_fill(self, symbol: str, order_id: str) -> Tuple[bool, Optional[float]]:
        """Проверяем факт исполнения по orderId через историю ордеров."""
        try:
            res = self.client.get_order_history(category="linear", symbol=symbol, orderId=order_id, limit=1)
            if res.get("retCode") != 0:
                return False, None
            rows = res.get("result", {}).get("list") or []
            if not rows:
                return False, None
            row = rows[0]
            cum = self._safe_float(row.get("cumExecQty"))
            avg = self._safe_float(row.get("avgPrice"))
            return (cum > 0, (avg if avg > 0 else None))
        except Exception:
            return False, None

    def get_balance(self) -> Dict:
        try:
            res = self.client.get_wallet_balance(accountType="UNIFIED")
            return res["result"]["list"][0] if res.get("retCode") == 0 else {}
        except Exception as e:
            logging.error(f"balance error: {e}")
            return {}

    def get_position(self, symbol: str) -> Optional[Position]:
        try:
            res = self.client.get_positions(category="linear", symbol=symbol)
            if res.get("retCode") == 0 and res.get("result", {}).get("list"):
                p = res["result"]["list"][0]
                return Position(
                    symbol=p.get("symbol", symbol),
                    side=p.get("side", ""),
                    size=self._safe_float(p.get("size")),
                    entry_price=self._safe_float(p.get("avgPrice")),
                    unrealized_pnl=self._safe_float(p.get("unrealisedPnl")),
                    percentage=self._safe_float(p.get("percentage")),
                )
        except Exception as e:
            logging.error(f"position error: {e}")
        return None

    def get_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        try:
            res = self.client.get_kline(category="linear", symbol=symbol, interval=interval, limit=limit)
            if res.get("retCode") == 0:
                data = res["result"]["list"]
                df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
                df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
                return df.sort_values("timestamp").reset_index(drop=True)
        except Exception as e:
            logging.error(f"klines error: {e}")
        return pd.DataFrame()

    def get_open_orders(self, symbol: str) -> list:
        """Return list of open active orders for symbol (linear)."""
        try:
            res = self.client.get_open_orders(category="linear", symbol=symbol)
            if res.get("retCode") == 0:
                return (res.get("result") or {}).get("list") or []
        except Exception as e:
            logging.error(f"get_open_orders error: {e}")
        return []

    def has_partial_tp(self, symbol: str, side_close: str, target_price: float, tol_ticks: int = 2) -> bool:
        """Detect existing reduceOnly GTC limit close order near target (within tol_ticks)."""
        try:
            orders = self.get_open_orders(symbol)
            if not orders:
                return False
            tick = self.get_symbol_meta(symbol).get("tick_size", 0.1) or 0.1
            for o in orders:
                try:
                    if str(o.get("reduceOnly")).lower() != "true":
                        continue
                    if str(o.get("orderType")).lower() != "limit":
                        continue
                    if str(o.get("side", "")).lower() != str(side_close).lower():
                        continue
                    px = self._safe_float(o.get("price"))
                    if px <= 0:
                        continue
                    if abs(px - float(target_price)) <= tol_ticks * float(tick):
                        return True
                except Exception:
                    continue
        except Exception as e:
            logging.error(f"has_partial_tp error: {e}")
        return False

    def arm_trailing_stop(self, symbol: str, take_profit: Optional[float], trailing_dist: Optional[float]) -> dict:
        """Set trailing stop and optional TP via set_trading_stop. Omits None fields to avoid 'take_profit invalid'."""
        try:
            params = {
                "category": "linear",
                "symbol": symbol,
                "positionIdx": 0,
                "triggerBy": "LastPrice",
            }
            # tpslMode имеет смысл только если мы что-то ставим из TP/Trailing
            if take_profit is not None or trailing_dist is not None:
                params["tpslMode"] = "Full"
            if take_profit is not None:
                # нормализуем TP к тик-сайзу
                tp_px = self.normalize_price(symbol, float(take_profit))
                params["takeProfit"] = str(tp_px)
            if trailing_dist is not None:
                # биржа ждёт положительную дистанцию в цене
                dist = abs(float(trailing_dist))
                params["trailingStop"] = str(dist)
            return self.client.set_trading_stop(**params)
        except Exception as e:
            return {"retCode": -1, "retMsg": str(e)}

    # ---- главная операция открытия позиции ----
    def place_market_order(
        self,
        symbol: str,
        side: str,
        qty: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Dict:
        """
        Алгоритм:
          1) стакан
          2) Limit + IOC с адаптивным slippage и сдвигом по тикам, ТОЛЬКО TP в ордере (SL → через trailing)
          3) подтверждение fill/позиции
          4) set_trading_stop: trailing + TP (ретраи)
        """
        # 1) стакан
        try:
            ob = self.client.get_orderbook(category="linear", symbol=symbol, limit=1)
            if ob.get("retCode") != 0:
                return {"retCode": -1, "retMsg": f"orderbook retCode={ob.get('retCode')}"}
            best_bid = self._safe_float(ob["result"]["b"][0][0])
            best_ask = self._safe_float(ob["result"]["a"][0][0])
        except Exception as e:
            return {"retCode": -1, "retMsg": f"orderbook error: {e}"}

        sside = str(side).lower()

        last_avg_from_order: Optional[float] = None

        def _try_ioc_with(bps: float, tick_shift: int) -> Dict:
            # refresh
            ob2 = self.client.get_orderbook(category="linear", symbol=symbol, limit=1)
            if ob2.get("retCode") != 0:
                return {"retCode": -1, "retMsg": f"orderbook retCode={ob2.get('retCode')}"}
            bb = self._safe_float(ob2["result"]["b"][0][0])
            ba = self._safe_float(ob2["result"]["a"][0][0])

            if sside == "buy":
                ref = ba * (1.0 + bps / 10000.0)
            else:
                ref = bb * (1.0 - bps / 10000.0)

            base = self.normalize_price(symbol, ref, side=side, adjust=True)
            tick = self.get_symbol_meta(symbol).get("tick_size", 0.1) or 0.1
            price = base + tick_shift * tick if sside == "buy" else base - tick_shift * tick
            price = self.normalize_price(symbol, price)

            # предвалидация TP по правилам
            ref_entry = base
            approx_last = ba if sside == "buy" else bb
            pre_sl, pre_tp = self._ensure_tp_sl_valid(
                symbol=symbol,
                side=side,
                entry_price=float(ref_entry),
                stop_loss=stop_loss,
                take_profit=take_profit,
                current_price=float(approx_last),
            )

            # TP должен быть на «правильной стороне» от фактической цены IOC
            safe_tp = None
            if pre_tp is not None:
                tick_meta = self.get_symbol_meta(symbol)
                tick = tick_meta.get("tick_size", 0.1) or 0.1
                if sside == "buy":
                    ceil_ref = max(float(price), float(approx_last), float(ref_entry))
                    if pre_tp <= ceil_ref:
                        pre_tp = ceil_ref + tick
                else:
                    floor_ref = min(float(price), float(approx_last), float(ref_entry))
                    if pre_tp >= floor_ref:
                        pre_tp = floor_ref - tick
                safe_tp = self.normalize_price(symbol, float(pre_tp))

            logging.info(f"[IOC] bps={bps} shift={tick_shift} side={side} qty={qty} price={price} bestBid={bb} bestAsk={ba}")
            try:
                params = dict(
                    category="linear",
                    symbol=symbol,
                    side=side,
                    orderType="Limit",
                    timeInForce="IOC",
                    price=str(price),
                    qty=qty,
                    reduceOnly=False,
                    tpTriggerBy="LastPrice",
                    slTriggerBy="LastPrice",
                    tpslMode="Full",
                    positionIdx=0,
                )
                if safe_tp is not None:
                    params["takeProfit"] = str(safe_tp)
                # stopLoss не ставим в ордер — будем вешать trailing на позицию
                r = self.client.place_order(**params)
                try:
                    order_id = (r.get("result") or {}).get("orderId")
                except Exception:
                    order_id = None
                r = dict(r or {})
                if order_id:
                    r["orderId"] = order_id
                return r
            except Exception as e:
                return {"retCode": -1, "retMsg": str(e)}

        res_open = None
        last_err = None
        positioned = False
        for bps in SLP_BPS_STEPS:
            for t in range(0, IOC_TICKS_TRIES):
                res_open = _try_ioc_with(bps, t)
                if res_open and res_open.get("retCode") == 0:
                    # подтверждаем fill по ордеру
                    ord_id = res_open.get("orderId")
                    filled = False
                    avg_from_order = None
                    if ord_id:
                        for _ in range(6):
                            time.sleep(0.25)
                            filled, avg_from_order = self.get_order_fill(symbol, ord_id)
                            if filled:
                                break
                    # подтверждаем позицию
                    for _ in range(8):
                        time.sleep(0.25)
                        pos = self.get_position(symbol)
                        if pos and pos.size > 0:
                            positioned = True
                            break
                    if not positioned and filled:
                        positioned = True
                        last_avg_from_order = avg_from_order
                    if positioned:
                        break
                last_err = res_open
            if positioned:
                break

        if not positioned:
            return last_err or {"retCode": -2, "retMsg": "ioc not filled"}

        # позиция видна?
        pos = self.get_position(symbol)
        if not pos or pos.size <= 0:
            if last_avg_from_order is None:
                return {**(res_open or {}), "retCode": -2, "retMsg": "position not found after IOC"}
            else:
                logging.warning("[OPEN] order filled по истории, но позиция ещё не видна — пропускаем SL/TP в этом цикле")
                return {**(res_open or {}), "filled": True, "avgPrice": float(last_avg_from_order)}

        # 4) Trailing stop + TP по фактической цене (только если позиция реально видна)
        last_px = self.get_last_price(symbol)
        sl_norm, tp_norm = self._ensure_tp_sl_valid(
            symbol, side, float(pos.entry_price), stop_loss, take_profit, current_price=last_px
        )
        return {**(res_open or {}), "filled": True, "avgPrice": float(pos.entry_price)}

    # ---- валидация SL/TP по правилам биржи ----
    def _ensure_tp_sl_valid(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        current_price: Optional[float] = None,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Приводим SL/TP к правилам Bybit и нормализуем до тика.
          Buy:  SL < min(entry, lastPrice), TP > max(entry, lastPrice)
          Sell: SL > max(entry, lastPrice), TP < min(entry, lastPrice)
        """
        meta = self.get_symbol_meta(symbol)
        tick = meta.get("tick_size", 0.1) or 0.1
        last = float(current_price) if (current_price is not None and current_price > 0) else float(entry_price)

        if entry_price is None or entry_price <= 0:
            return (stop_loss, take_profit)

        s = str(side).lower()
        if s == "buy":
            floor_ref = min(entry_price, last)
            ceil_ref = max(entry_price, last)
            if stop_loss is not None and stop_loss >= floor_ref:
                stop_loss = floor_ref - tick
            if take_profit is not None and take_profit <= ceil_ref:
                take_profit = ceil_ref + tick
        else:  # sell
            floor_ref = min(entry_price, last)
            ceil_ref = max(entry_price, last)
            if stop_loss is not None and stop_loss <= ceil_ref:
                stop_loss = ceil_ref + tick
            if take_profit is not None and take_profit >= floor_ref:
                take_profit = floor_ref - tick

        sl_norm = self.normalize_price(symbol, float(stop_loss), side=None) if stop_loss is not None else None
        tp_norm = self.normalize_price(symbol, float(take_profit), side=None) if take_profit is not None else None
        return (sl_norm, tp_norm)

    def get_recent_filled_reduce_only(self, symbol: str, limit: int = 50):
        """
        Возвращает последний ИСПОЛНЕННЫЙ редьюс-ордер (включая TP/SL/Trailing),
        как квази-"выход позиции". Берём из истории ордеров.
        """
        try:
            res = self.client.get_order_history(category="linear", symbol=symbol, limit=limit)
            if res.get("retCode") != 0:
                return None
            rows = (res.get("result") or {}).get("list") or []
            # фильтруем Filled reduceOnly
            cand = []
            for r in rows:
                try:
                    if str(r.get("reduceOnly")).lower() == "true" and str(r.get("orderStatus")).lower() == "filled":
                        cand.append(r)
                except Exception:
                    continue
            if not cand:
                return None
            # самый "свежий" по updatedTime/createdTime
            cand.sort(key=lambda x: int(x.get("updatedTime") or x.get("createdTime") or 0), reverse=True)
            r = cand[0]
            return {
                "orderId": r.get("orderId"),
                "avgPrice": self._safe_float(r.get("avgPrice")),
                "cumExecQty": self._safe_float(r.get("cumExecQty")),
                "side": r.get("side"),
                "updatedTime": int(r.get("updatedTime") or 0),
            }
        except Exception:
            return None

    def get_server_time(self) -> int:
        """На всякий случай, чтобы штампать close c биржевым временем (мс)."""
        try:
            res = self.client.get_server_time()
            if res.get("retCode") == 0:
                return int((res.get("time") or res.get("result", {}).get("time")) or 0)
        except Exception:
            pass
        return 0

    def place_reduce_only_limit(self, symbol: str, side_close: str, qty: float, price: float) -> dict:
        """Выставляет GTC reduceOnly лимит на закрытие части позиции (частичный TP)."""
        try:
            # 0) Метаданные инструмента
            meta = self.get_symbol_meta(symbol)
            step = float(meta.get("qty_step", 0.001) or 0.001)
            minq = float(meta.get("min_qty", step) or step)
            min_notional = float(meta.get("min_notional", 0.0) or 0.0)

            # 1) Текущая позиция — закрывать можно только до её размера
            pos = self.get_position(symbol)
            if not pos or float(pos.size) <= 0:
                return {"retCode": -11, "retMsg": "no position to reduce"}

            # сторона должна быть обратной
            if str(pos.side).lower() == "buy" and str(side_close).lower() != "sell":
                return {"retCode": -12, "retMsg": "side mismatch: need Sell to close Buy"}
            if str(pos.side).lower() == "sell" and str(side_close).lower() != "buy":
                return {"retCode": -12, "retMsg": "side mismatch: need Buy to close Sell"}

            # 2) Обрезаем желаемое количество до позиции
            max_close = float(pos.size)
            qty_cap = min(float(qty), max_close)
            if qty_cap <= 0:
                return {"retCode": -13, "retMsg": "qty<=0 after cap by position"}

            # 3) Нормализация к шагу
            import math
            qty_norm = math.floor(qty_cap / step) * step
            # 4) Проверка min_qty и min_notional
            if qty_norm < minq:
                return {"retCode": -13, "retMsg": f"qty too small for exchange (qty_norm={qty_norm}, min_qty={minq})"}
            px = self.normalize_price(symbol, price)
            if min_notional > 0 and (px * qty_norm) < min_notional:
                return {"retCode": -13, "retMsg": f"notional too small (px*qty={px*qty_norm} < min_notional={min_notional})"}

            return self.client.place_order(
                category="linear",
                symbol=symbol,
                side=side_close,
                orderType="Limit",
                timeInForce="GTC",
                price=str(px),
                qty=f"{qty_norm:.8f}",
                reduceOnly=True,
                positionIdx=0,
            )
        except Exception as e:
            return {"retCode": -1, "retMsg": str(e)}

# ---------------------- BOT CORE ----------------------
class ScalpingBot:
    def __init__(self):
        self.analyzer = TechnicalAnalyzer()
        base_strats = ScalpingStrategies(self.analyzer)
        self.strategies = InversionProxyStrategies(self.analyzer, base_strats) if INVERT_SIGNALS else base_strats

        # аналитическая БД
        self.adb = AnalyticsDB("analytics.db")

        # клиенты биржи
        self.trader_main = BybitTrader(BYBIT_API_KEY_MAIN, BYBIT_API_SECRET_MAIN, False)
        self.trader_test = BybitTrader(BYBIT_API_KEY_TEST, BYBIT_API_SECRET_TEST, True)

        self.mode = TradingMode.TESTNET
        self.symbol = SYMBOL
        self.status = BotStatus.STOPPED

        self.max_notional_usdt = MAX_NOTIONAL_USDT
        self.risk_cfg = RiskConfig(
            max_notional_usdt=self.max_notional_usdt,
            leverage=BASE_LEVERAGE,
            notes="tradebot",
        )

        # Telegram
        self.telegram_token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID

        # планировщик
        self.scheduler = AsyncIOScheduler()

        # слепок позиции для уведомления о закрытии
        self._last_pos_open = False
        self._last_pos_side = None
        self._last_pos_entry = None
        self._last_pos_size = 0.0

    @property
    def current_trader(self) -> BybitTrader:
        return self.trader_test if self.mode == TradingMode.TESTNET else self.trader_main

    # ----- Разворотный фильтр -----
    def _is_reversal_signal(self, market: Dict, signal: TradeSignal, pos: Position) -> bool:
        """True, если сигнал сильно против текущей позиции (условия для принудительного закрытия)."""
        try:
            if not signal or not pos or pos.size <= 0:
                return False
            s = market.get("signals", {})
            rsi = float(market.get("indicators", {}).get("rsi", 50.0))
            conf = float(getattr(signal, "confidence", 0.0) or 0.0)

            if str(pos.side).lower() == "buy":
                if signal.side == OrderSide.SELL and s.get("macd_bearish") and s.get("price_below_ema_fast") and rsi < 50.0 and conf >= 0.65:
                    return True
            elif str(pos.side).lower() == "sell":
                if signal.side == OrderSide.BUY and s.get("macd_bullish") and s.get("price_above_ema_fast") and rsi > 50.0 and conf >= 0.65:
                    return True
        except Exception:
            return False
        return False

    # ----- UI -----
    def get_keyboard(self) -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "▶️ Старт" if self.status == BotStatus.STOPPED else "⏸️ Стоп",
                        callback_data="toggle_bot",
                    ),
                    InlineKeyboardButton("💰 Баланс", callback_data="balance"),
                ],
                [
                    InlineKeyboardButton("📊 Статус", callback_data="status"),
                    InlineKeyboardButton("📈 Статистика", callback_data="stats"),
                ],
                [
                    InlineKeyboardButton(
                        "🧪 Testnet" if self.mode == TradingMode.MAINNET else "💸 Mainnet",
                        callback_data="toggle_mode",
                    ),
                    InlineKeyboardButton("🛑 Закрыть позицию", callback_data="close_position"),
                ],
                [
                    InlineKeyboardButton("🧹 Отменить все ордера", callback_data="cancel_all"),
                ],
            ]
        )

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🤖 Агрессивный скальпинг-бот запущен!\n\n"
            "⚠️ ВНИМАНИЕ: Очень высокий риск. Тестируйте на TESTNET.",
            reply_markup=self.get_keyboard(),
        )

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        q = update.callback_query
        await q.answer()

        msg = "Окей."
        if q.data == "toggle_bot":
            if self.status == BotStatus.STOPPED:
                self.status = BotStatus.RUNNING
                if not self.scheduler.get_job("trading_job"):
                    self.scheduler.add_job(
                        self.analyze_and_trade,
                        "interval",
                        seconds=20,
                        id="trading_job",
                        coalesce=True,
                        max_instances=1,
                        misfire_grace_time=20,
                    )
                if not self.scheduler.running:
                    self.scheduler.start()
                msg = f"✅ Бот запущен в режиме {self.mode.value} (invert={INVERT_SIGNALS})"
            else:
                self.status = BotStatus.STOPPED
                if self.scheduler.get_job("trading_job"):
                    self.scheduler.remove_job("trading_job")
                msg = "⏸️ Бот остановлен"

        elif q.data == "toggle_mode":
            self.mode = TradingMode.MAINNET if self.mode == TradingMode.TESTNET else TradingMode.TESTNET
            msg = f"🔁 Режим переключен на: {self.mode.value}"

        elif q.data == "balance":
            bal = self.current_trader.get_balance()
            total = float(bal.get("totalAvailableBalance", 0.0))
            msg = f"💰 Баланс ({self.mode.value}): ${total:.2f}"

        elif q.data == "status":
            pos = self.current_trader.get_position(self.symbol)
            if pos and pos.size > 0:
                pnl_emoji = "📈" if pos.unrealized_pnl >= 0 else "📉"
                msg = (
                    f"📊 Статус\nРежим: {self.mode.value}\n"
                    f"Позиция: {pos.side} {pos.size}\n"
                    f"Вход: {pos.entry_price:.2f}\n{pnl_emoji} uPnL: {pos.unrealized_pnl:.2f}"
                )
            else:
                msg = f"📊 Статус\nРежим: {self.mode.value}\nПозиция: закрыта"

        elif q.data == "stats":
            msg = "📈 Сырые логи сигналов/ордеров пишутся в analytics.db (таблицы из test_db.py)."

        elif q.data == "close_position":
            ok, txt = await self._close_position_and_clear()
            msg = txt

        elif q.data == "cancel_all":
            ok, txt = await self._cancel_all_orders()
            msg = txt

        try:
            await q.edit_message_text(msg, reply_markup=self.get_keyboard())
        except Exception:
            await self.send_telegram_message(msg)

    async def _close_position_and_clear(self) -> Tuple[bool, str]:
        """Рыночное закрытие позиции и попытка убрать SL/TP (на всякий)."""
        try:
            # отменяем все активные/условные ордера
            await self._cancel_all_orders()

            # закрываем позицию рыночным редьюсом
            trader = self.current_trader
            pos = trader.get_position(self.symbol)
            if not pos or pos.size <= 0:
                return True, "ℹ️ Позиции нет."

            side_close = "Sell" if pos.side == "Buy" else "Buy"
            qty_str = f"{float(pos.size):.8f}"

            try:
                res = trader.client.place_order(
                    category="linear",
                    symbol=self.symbol,
                    side=side_close,
                    orderType="Market",
                    qty=qty_str,
                    reduceOnly=True,
                )
            except Exception as e:
                return False, f"❌ Ошибка закрытия: {e}"

            if res.get("retCode") != 0:
                return False, f"❌ Биржа ответила retCode={res.get('retCode')}"

            return True, "✅ Позиция закрыта (reduceOnly Market)."
        except Exception as e:
            return False, f"❌ Ошибка: {e}"

    async def _cancel_all_orders(self) -> Tuple[bool, str]:
        """Отменить все активные/условные ордера по символу."""
        trader = self.current_trader
        try:
            res1 = trader.client.cancel_all_orders(category="linear", symbol=self.symbol)
            res2 = trader.client.cancel_all_orders(category="linear", symbol=self.symbol, orderFilter="StopOrder")
            ok1 = (res1.get("retCode") == 0)
            ok2 = (res2.get("retCode") == 0)
            if ok1 or ok2:
                return True, "🧹 Отменены активные и условные ордера."
            return False, f"❌ cancel_all: {res1} / {res2}"
        except Exception as e:
            return False, f"❌ Ошибка отмены: {e}"

    async def send_telegram_message(self, text: str):
        try:
            if not self.telegram_token or not self.chat_id:
                logging.warning(f"[TG] token/chat_id не заданы, сообщение: {text}")
                return
            bot = Bot(token=self.telegram_token)
            await bot.send_message(chat_id=self.chat_id, text=text)
        except Exception as e:
            logging.error(f"TG send error: {e}")

    # ----- CORE -----
    async def analyze_and_trade(self):
        if self.status != BotStatus.RUNNING:
            return

        trader = self.current_trader

        # 0) фиксация закрытия (детектор перехода open->closed)
        try:
            pos_now = trader.get_position(self.symbol)
            now_open = bool(pos_now and pos_now.size > 0)
            if self._last_pos_open and not now_open:
                open_trade = self._get_last_open_trade(self.symbol)
                if open_trade:
                    ro = trader.get_recent_filled_reduce_only(self.symbol)
                    if ro and ro.get("avgPrice"):
                        avg_exit = float(ro["avgPrice"])
                        ts_close = datetime.now(UTC)
                    else:
                        last_px = trader.get_last_price(self.symbol) or 0.0
                        avg_exit = float(last_px)
                        ts_close = datetime.now(UTC)
                    try:
                        self.adb.close_trade(symbol=self.symbol, exit_price=avg_exit, ts=ts_close)
                    except Exception:
                        pass
                    await self.send_telegram_message(f"✅ Позиция закрыта. Выход ~ {avg_exit:.2f}")
            self._last_pos_open = now_open
            if now_open and pos_now:
                self._last_pos_side = pos_now.side
                self._last_pos_entry = pos_now.entry_price
                self._last_pos_size = pos_now.size
        except Exception as e:
            logging.warning(f"[CLOSE-DETECT] {e}")

        # 1) данные рынка
        df = trader.get_klines(self.symbol, interval="5", limit=200)
        if df.empty:
            logging.warning("[DATA] empty klines")
            return

        market = self.analyzer.analyze_market(df)
        price = float(market["price"])
        atr_val = float(market["indicators"]["atr"])

        # 2) сигнал (с возможной инверсией на уровне self.strategies)
        sig = self.strategies.get_best_signal(market)

        # лог в БД
        try:
            self.adb.log_signal(
                symbol=self.symbol,
                ts=datetime.now(UTC),
                price=price,
                meta=json.dumps({"inv": INVERT_SIGNALS, "rsi": market["indicators"]["rsi"]}),
                strategy=(sig.strategy_name if sig else "none"),
                side=(sig.side.value if sig else "none"),
                conf=(sig.confidence if sig else 0.0),
            )
        except Exception:
            pass

        # 3) защита от «ножей» и супер-спайков
        s = market["signals"]
        if s.get("super_spike") or not s.get("atr_pct_ok") or not s.get("sr_ok"):
            return

        # 4) если позиции нет — можем войти по сигналу
        pos = trader.get_position(self.symbol)
        if not pos or pos.size <= 0:
            if not sig:
                return

            # размер позиции от риска
            meta = trader.get_symbol_meta(self.symbol)
            tick = float(meta.get("tick_size", 0.1) or 0.1)

            # если у сигнала нет SL/TP — попробуем вычислить по профилю
            sl_px, tp_px = sig.stop_loss, sig.take_profit
            if sl_px is None or tp_px is None:
                alt_sl, alt_tp = compute_tpsl_for_symbol(self.symbol, sig.side.value, sig.entry_price, atr_val, meta)
                sl_px = sl_px if sl_px is not None else alt_sl
                tp_px = tp_px if tp_px is not None else alt_tp

            # валидация уровня
            sl_px, tp_px = trader._ensure_tp_sl_valid(
                symbol=self.symbol,
                side=sig.side.value,
                entry_price=sig.entry_price,
                stop_loss=sl_px,
                take_profit=tp_px,
                current_price=price,
            )

            # риск на сделку -> количество (через дистанцию до SL)
            risk_dist = abs(sig.entry_price - (sl_px if sl_px is not None else sig.entry_price - max(atr_val, tick)))
            if risk_dist <= 0:
                return
            qty_est = (self.risk_cfg.max_notional_usdt / price) if RISK_USDT <= 0 else (RISK_USDT / risk_dist)
            qty = trader.normalize_qty(self.symbol, max(qty_est, meta.get("min_qty", 0.001)))

            # открытие лимит+IOC с TP (SL позже через trailing)
            res = trader.place_market_order(
                symbol=self.symbol,
                side=sig.side.value,
                qty=f"{qty:.8f}",
                stop_loss=sl_px,
                take_profit=tp_px,
            )
            if res.get("retCode") != 0:
                logging.warning(f"[OPEN FAIL] {res}")
                await self.send_telegram_message(f"❌ Открытие не удалось: {res.get('retMsg')}")
                return

            # зафиксировали факт открытия
            try:
                self.adb.open_trade(
                    symbol=self.symbol,
                    ts=datetime.now(UTC),
                    side=sig.side.value,
                    entry_price=float(res.get("avgPrice") or sig.entry_price),
                    strategy=sig.strategy_name,
                    size=qty,
                )
            except Exception:
                pass

            await self.send_telegram_message(
                f"🚀 Открытие: {sig.side.value} qty={qty:.4f} @~{float(res.get('avgPrice') or sig.entry_price):.2f} "
                f"(strat={sig.strategy_name}, inv={INVERT_SIGNALS})"
            )

            # trailing включим, когда уйдём в плюс на TRAIL_ARM_ATR*ATR; частичный TP поставим заранее
            return

        # 5) если позиция есть — управление: частичный TP и трейлинг
        try:
            # side_close для лимитных редьюсов
            side_close = "Sell" if str(pos.side).lower() == "buy" else "Buy"

            # целевой частичный TP
            part_dist = max(PART_TP_ATR_MULT * atr_val, 2 * trader.get_symbol_meta(self.symbol)["tick_size"])
            target_px = pos.entry_price + part_dist if pos.side == "Buy" else pos.entry_price - part_dist
            target_px = trader.normalize_price(self.symbol, target_px)

            # если нет частичного TP — поставим reduceOnly GTC ~ на 1/3 размера
            if not trader.has_partial_tp(self.symbol, side_close, target_px, tol_ticks=2):
                part_qty = max(PART_TP_FRAC * float(pos.size), trader.get_symbol_meta(self.symbol)["min_qty"])
                place_res = trader.place_reduce_only_limit(self.symbol, side_close, part_qty, target_px)
                if place_res.get("retCode") == 0:
                    logging.info(f"[PART-TP] placed {part_qty} @ {target_px}")
                else:
                    logging.info(f"[PART-TP] skip: {place_res}")

            # трейлинг — когда ушли в плюс на TRAIL_ARM_ATR*ATR
            pnl_move = (price - pos.entry_price) if pos.side == "Buy" else (pos.entry_price - price)
            if pnl_move >= TRAIL_ARM_ATR * atr_val:
                trailing_dist = max(ATR_PCT_MIN * price, TRAIL_FROM_SL_MIN * trader.get_symbol_meta(self.symbol)["tick_size"])
                ar = trader.arm_trailing_stop(self.symbol, take_profit=None, trailing_dist=trailing_dist)
                logging.info(f"[TRAIL] set_trading_stop -> {ar}")
        except Exception as e:
            logging.warning(f"[PM] {e}")

    # ----- Вспомогательные методы для БД -----
    def _get_last_open_trade(self, symbol: str):
        try:
            return self.adb.get_last_open_trade(symbol=symbol)
        except Exception:
            return None

# ---------------------- ENTRYPOINT ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
)

async def _bootstrap(app: Application, bot: ScalpingBot):
    app.add_handler(CommandHandler("start", bot.start_command))
    app.add_handler(CallbackQueryHandler(bot.button_callback))
    await app.initialize()
    await app.start()
    await app.updater.start_polling()

def main():
    bot = ScalpingBot()
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    import asyncio
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(_bootstrap(application, bot))
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            loop.run_until_complete(application.stop())
        except Exception:
            pass

if __name__ == "__main__":
    main()
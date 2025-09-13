# Агрессивный скальпинг-бот для Bybit с Telegram управлением
# ВЫСОКИЙ РИСК — ТЕСТИРОВАТЬ ТОЛЬКО НА TESTNET!

import os
import json
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

# наша аналитическая БД
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
TELEGRAM_BOT_TOKEN = os.getenv("TG_TOKEN_LOCAL", "")
TELEGRAM_CHAT_ID = os.getenv("TG_ADMIN_CHAT_ID", "")

# Торговые параметры
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
BASE_LEVERAGE = int(os.getenv("BASE_LEVERAGE", "10"))
MAX_NOTIONAL_USDT = float(os.getenv("MAX_NOTIONAL_USDT", "10"))  # целевой номинал позиции
IOC_TICKS_TRIES = int(os.getenv("IOC_TICKS_TRIES", "5"))  # сколько тиков добавочно двигать цену для IOC
SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", "10"))     # 10 б.п. = 0.10% буфер к стакану

SLP_BPS_STEPS = [float(x) for x in os.getenv("SLP_BPS_STEPS", "10,25,50,80,120").split(",")]

# Пер-символьная конфигурация TP/SL

PER_SYMBOL_TPSL = {
    "BTCUSDT": {"mode": "auto"},
    "ETHUSDT": {"mode": "atr", "sl_atr": 1.2, "tp_atr": 2.0, "atr_len": 14},
    "XRPUSDT": {"mode": "abs", "sl_abs": 0.003, "tp_abs": 0.006},
    "SOLUSDT": {"mode": "rr", "sl_ticks": 15, "rr": 1.8},
}

# ---------------------- TP/SL HELPER ----------------------
from typing import Optional, Tuple
def compute_tpsl_for_symbol(
    symbol: str,
    side: str,
    entry: float,
    atr_value: Optional[float],
    meta: dict,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute absolute SL/TP for a symbol, based on PER_SYMBOL_TPSL config.
    Returns (sl, tp) as floats (not quantized), or (None, None) if cannot compute.
    """
    conf = PER_SYMBOL_TPSL.get(symbol)
    if not conf:
        return (None, None)
    mode = conf.get("mode", "").lower()
    tick_size = meta.get("tick_size", 0.1) if meta else 0.1
    if not tick_size or tick_size <= 0:
        tick_size = 0.1
    # Direction: Buy: SL below, TP above; Sell: SL above, TP below
    s = str(side).lower()

    # --- AUTO MODE: choose distances by situation (ATR vs price & tick) ---
    if mode in ("", "auto", None):
        # Heuristics:
        #  - High vol: use ATR multipliers
        #  - Medium vol: use ticks scaled from ATR
        #  - Low vol: use absolute fractions of price
        atr_pct = (atr_value / entry) if (atr_value and entry) else 0.0
        # Defaults
        sl_dist = None
        tp_dist = None
        chosen = None
        if atr_pct >= 0.003:  # >= 0.3%
            # High volatility → ATR-based
            sl_mult = conf.get("sl_atr", 1.2)
            tp_mult = conf.get("tp_atr", 1.8)
            if atr_value and atr_value > 0:
                sl_dist = sl_mult * atr_value
                tp_dist = tp_mult * atr_value
                chosen = "atr"
        elif atr_pct >= 0.0015:  # 0.15% .. 0.3% → medium
            # Medium volatility → ticks roughly from ATR
            base_ticks = max(5, int(round((0.8 * (atr_value or 0.0)) / tick_size)))
            rr = conf.get("rr", 1.6)
            sl_dist = base_ticks * tick_size
            tp_dist = int(round(base_ticks * rr)) * tick_size
            chosen = "ticks"
        else:
            # Low volatility → absolute fraction of price
            sl_abs_frac = conf.get("sl_abs_frac", 0.0010)   # 0.10%
            tp_abs_frac = conf.get("tp_abs_frac", 0.0018)   # 0.18%
            sl_dist = max(2 * tick_size, sl_abs_frac * entry)
            tp_dist = max(3 * tick_size, tp_abs_frac * entry)
            chosen = "abs"
        # Build SL/TP by side
        if sl_dist is not None and tp_dist is not None:
            if s == "buy":
                sl = entry - sl_dist
                tp = entry + tp_dist
            else:
                sl = entry + sl_dist
                tp = entry - tp_dist
            return (sl, tp)

    # --- TICKS ---
    if mode == "ticks":
        sl_ticks = conf.get("sl_ticks")
        tp_ticks = conf.get("tp_ticks")
        if sl_ticks is None or tp_ticks is None:
            return (None, None)
        sl_dist = sl_ticks * tick_size
        tp_dist = tp_ticks * tick_size
        if s == "buy":
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist
        return (sl, tp)
    # --- ATR ---
    elif mode == "atr":
        sl_atr = conf.get("sl_atr")
        tp_atr = conf.get("tp_atr")
        if sl_atr is None or tp_atr is None or not atr_value or atr_value <= 0:
            return (None, None)
        sl_dist = sl_atr * atr_value
        tp_dist = tp_atr * atr_value
        if s == "buy":
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist
        return (sl, tp)
    # --- ABS ---
    elif mode == "abs":
        sl_abs = conf.get("sl_abs")
        tp_abs = conf.get("tp_abs")
        if sl_abs is None or tp_abs is None:
            return (None, None)
        if s == "buy":
            sl = entry - sl_abs
            tp = entry + tp_abs
        else:
            sl = entry + sl_abs
            tp = entry - tp_abs
        return (sl, tp)
    # --- RR ---
    elif mode == "rr":
        rr = conf.get("rr")
        if rr is None:
            return (None, None)
        # SL distance: prefer sl_ticks, else sl_abs, else sl_atr
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
        if s == "buy":
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist
        return (sl, tp)
    return (None, None)

# Логирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

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

        signals = {
            "rsi_oversold": bool(rsi[-1] < 30),
            "rsi_overbought": bool(rsi[-1] > 70),
            "macd_bullish": bool(macd[-1] > macd_signal[-1] and macd_hist[-1] > 0),
            "macd_bearish": bool(macd[-1] < macd_signal[-1] and macd_hist[-1] < 0),
            "price_above_ema_fast": bool(price > float(ema_fast[-1])),
            "price_below_ema_fast": bool(price < float(ema_fast[-1])),
            "bb_oversold": bool(price < float(bb_lower[-1])),
            "bb_overbought": bool(price > float(bb_upper[-1])),
            "volume_spike": bool(volume[-1] > float(np.mean(volume[-20:]) * 1.5)),
            "atr": float(atr[-1]),
        }

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
                "atr": float(atr[-1]),
            },
        }


class ScalpingStrategies:
    def __init__(self, analyzer: TechnicalAnalyzer):
        self.analyzer = analyzer

    def rsi_mean_reversion(self, market: Dict) -> Optional[TradeSignal]:
        s = market["signals"]
        p = market["price"]
        atr = s["atr"]

        if s["rsi_oversold"] and s["volume_spike"]:
            return TradeSignal(
                side=OrderSide.BUY,
                entry_price=p,
                stop_loss=p - 1.5 * atr,
                take_profit=p + 2.0 * atr,
                confidence=0.7,
                strategy_name="rsi_mean_reversion_buy",
            )
        if s["rsi_overbought"] and s["volume_spike"]:
            return TradeSignal(
                side=OrderSide.SELL,
                entry_price=p,
                stop_loss=p + 1.5 * atr,
                take_profit=p - 2.0 * atr,
                confidence=0.7,
                strategy_name="rsi_mean_reversion_sell",
            )
        return None

    def macd_momentum(self, market: Dict) -> Optional[TradeSignal]:
        s = market["signals"]
        p = market["price"]
        atr = s["atr"]

        if s["macd_bullish"] and s["price_above_ema_fast"]:
            return TradeSignal(
                side=OrderSide.BUY,
                entry_price=p,
                stop_loss=p - 1.0 * atr,
                take_profit=p + 1.5 * atr,
                confidence=0.6,
                strategy_name="macd_momentum_buy",
            )
        if s["macd_bearish"] and s["price_below_ema_fast"]:
            return TradeSignal(
                side=OrderSide.SELL,
                entry_price=p,
                stop_loss=p + 1.0 * atr,
                take_profit=p - 1.5 * atr,
                confidence=0.6,
                strategy_name="macd_momentum_sell",
            )
        return None

    def bb_breakout(self, market: Dict) -> Optional[TradeSignal]:
        s = market["signals"]
        p = market["price"]
        atr = s["atr"]

        if s["bb_overbought"] and s["volume_spike"]:
            return TradeSignal(
                side=OrderSide.BUY,
                entry_price=p,
                stop_loss=p - 2.0 * atr,
                take_profit=p + 1.0 * atr,
                confidence=0.5,
                strategy_name="bb_breakout_buy",
            )
        if s["bb_oversold"] and s["volume_spike"]:
            return TradeSignal(
                side=OrderSide.SELL,
                entry_price=p,
                stop_loss=p + 2.0 * atr,
                take_profit=p - 1.0 * atr,
                confidence=0.5,
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


# ---------------------- BYBIT TRADER ----------------------
class BybitTrader:
    def get_last_price(self, symbol: str) -> Optional[float]:
        """LastPrice из тикера."""
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
        """
        Проверяем факт исполнения по orderId через историю ордеров.
        Возвращает (filled, avg_price) где filled=True если cumExecQty > 0.
        """
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

    def _tick_decimals(self, tick: float) -> int:
        s = f"{tick}"
        return len(s.split(".")[1]) if "." in s else 0

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

    # ---- public ----
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

    def place_market_order(
        self,
        symbol: str,
        side: str,
        qty: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Dict:
        """
        Открываем позицию максимально безопасно:
          1) стакан
          2) открытие Limit + IOC (без SL/TP), с адаптивным SLP_BPS_STEPS и tick_shift
          3) проверка факта позиции
          4) постановка SL/TP через set_trading_stop (ретраи)
        Возвращает {'retCode':0,'filled':True,'avgPrice':...} при успехе,
        {'retCode':-2,'retMsg':'ioc not filled'} если IOC не исполнился.
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
            # refresh orderbook each attempt
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

            # --- pre-validate TP/SL for in-order submission ---
            # Use candidate IOC price as entry estimate and top-of-book as current reference
            ref_entry = base  # candidate limit price before tick_shift
            # approximate current price: bestAsk for buy, bestBid for sell
            approx_last = ba if sside == "buy" else bb
            pre_sl, pre_tp = self._ensure_tp_sl_valid(
                symbol=symbol,
                side=side,
                entry_price=float(ref_entry),
                stop_loss=stop_loss,
                take_profit=take_profit,
                current_price=float(approx_last),
            )

            logging.info(f"[IOC] bps={bps} shift={tick_shift} side={side} qty={qty} price={price} bestBid={bb} bestAsk={ba}")
            try:
                r = self.client.place_order(
                    category="linear",
                    symbol=symbol,
                    side=side,
                    orderType="Limit",
                    timeInForce="IOC",
                    price=str(price),
                    qty=qty,
                    reduceOnly=False,
                    # --- TP only at creation; SL handled via trailing ---
                    takeProfit=(str(pre_tp) if pre_tp is not None else None),
                    stopLoss=None,
                    tpTriggerBy="LastPrice",
                    slTriggerBy="LastPrice",
                    tpslMode="Full",
                    positionIdx=0,
                )
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
                    # подтвердим исполнение по самому ордеру (частичный IOC тоже возможен)
                    ord_id = res_open.get("orderId")
                    filled = False
                    avg_from_order = None
                    if ord_id:
                        for _ in range(6):
                            time.sleep(0.25)
                            filled, avg_from_order = self.get_order_fill(symbol, ord_id)
                            if filled:
                                break
                    # затем подтверждаем позицию (может появиться с задержкой)
                    for _ in range(8):
                        time.sleep(0.25)
                        pos = self.get_position(symbol)
                        if pos and pos.size > 0:
                            positioned = True
                            break
                    # если позиция ещё не появилась, но есть факт исполнения по ордеру — считаем открытой
                    if not positioned and filled:
                        positioned = True
                        last_avg_from_order = avg_from_order
                    if positioned:
                        break
                last_err = res_open
            if positioned:
                break

        if not positioned:
            # no Market fallback (reduces 30208 noise). Let next tick retry.
            return last_err or {"retCode": -2, "retMsg": "ioc not filled"}

        # refresh pos once more to get avgPrice
        pos = self.get_position(symbol)
        if not pos or pos.size <= 0:
            # Позиция не отобразилась, но мог быть частичный/задержанный IOC; попробуем взять avgPrice из ордера
            if last_avg_from_order is None:
                return {**(res_open or {}), "retCode": -2, "retMsg": "position not found after IOC"}
            else:
                # Нет позиции — SL/TP поставить нельзя, пропускаем на этот цикл
                logging.warning("[OPEN] order filled по истории, но позиция ещё не видна — пропускаем SL/TP в этом цикле")
                return {**(res_open or {}), "filled": True, "avgPrice": float(last_avg_from_order)}

        # 4) Trailing stop + TP по фактической цене (только если позиция реально видна)
        last_px = self.get_last_price(symbol)
        sl_norm, tp_norm = self._ensure_tp_sl_valid(
            symbol, side, float(pos.entry_price), stop_loss, take_profit, current_price=last_px
        )

        # Derive trailing distance from SL relative to entry
        trail_dist = None
        try:
            if sl_norm is not None and pos and pos.entry_price:
                tick = self.get_symbol_meta(symbol).get("tick_size", 0.1) or 0.1
                trail_dist = abs(float(pos.entry_price) - float(sl_norm))
                # ensure >= 2 ticks
                min_trail = 2 * tick
                if trail_dist < min_trail:
                    trail_dist = min_trail
                # normalize to tick decimals
                dec = self._tick_decimals(tick)
                trail_dist = float(f"{trail_dist:.{dec}f}")
        except Exception:
            trail_dist = None

        def _try_set_trailing(tp_v, trailing_dist):
            try:
                return self.client.set_trading_stop(
                    category="linear",
                    symbol=symbol,
                    positionIdx=0,        # one-way; если hedge — маппинг Buy->1, Sell->2
                    tpslMode="Full",
                    triggerBy="LastPrice",
                    takeProfit=(str(tp_v) if tp_v is not None else None),
                    trailingStop=(str(trailing_dist) if trailing_dist is not None else None),
                )
            except Exception as e:
                return {"retCode": -1, "retMsg": str(e)}

        last_err = None
        for attempt in range(6):
            res_tpsl = _try_set_trailing(tp_norm, trail_dist)
            if res_tpsl and res_tpsl.get("retCode") == 0:
                last_err = None
                break

            last_err = res_tpsl
            msg = (res_tpsl or {}).get("retMsg", "")

            # Если биржа ругается на «должен быть выше/ниже base_price», пересчитаем только TP, trailing >= 2*tick
            if "should be higher than base_price" in msg or "should be lower than base_price" in msg:
                tick = self.get_symbol_meta(symbol).get("tick_size", 0.1) or 0.1
                last_px = self.get_last_price(symbol) or float(pos.entry_price)
                if str(side).lower() == "buy":
                    # Buy: TP > max(entry,last)
                    ceil_ref = max(float(pos.entry_price), last_px)
                    if tp_norm is not None and tp_norm <= ceil_ref:
                        tp_norm = ceil_ref + tick
                else:
                    # Sell: TP < min(entry,last)
                    floor_ref = min(float(pos.entry_price), last_px)
                    if tp_norm is not None and tp_norm >= floor_ref:
                        tp_norm = floor_ref - tick
                # нормализуем повторно
                tp_norm = self.normalize_price(symbol, tp_norm) if tp_norm is not None else None
                # ensure trailing distance >= 2*tick
                min_trail = 2 * tick
                if trail_dist is not None and trail_dist < min_trail:
                    dec = self._tick_decimals(tick)
                    trail_dist = float(f"{min_trail:.{dec}f}")
                time.sleep(0.25)
                continue

            # Иные ошибки — маленькая пауза и ещё попытки
            time.sleep(0.25)

        if last_err:
            logging.error(f"[Trailing/TP] set_trading_stop failed: {last_err}")

        logging.info(f"[OPEN] success side={side} avgPrice={float(pos.entry_price)} trail={trail_dist} tp={tp_norm}")
        return {**(res_open or {}), "filled": True, "avgPrice": float(pos.entry_price)}


# ---------------------- BOT CORE ----------------------
class ScalpingBot:
    def _is_reversal_signal(self, market: Dict, signal: TradeSignal, pos: Position) -> bool:
        """Возврат True, если сигнал сильно против текущей позиции (разворот).
        Критерии по умолчанию (можно будет вынести в конфиг):
          • Для Buy-позиции: новый сигнал Sell И macd_bearish И price_below_ema_fast И RSI<50 И confidence>=0.65
          • Для Sell-позиции: новый сигнал Buy  И macd_bullish И price_above_ema_fast И RSI>50 И confidence>=0.65
        """
        try:
            if not signal or not pos or pos.size <= 0:
                return False
            s = market.get("signals", {})
            ind = market.get("indicators", {})
            rsi = float(ind.get("rsi", 50.0))
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
    def __init__(self):
        self.analyzer = TechnicalAnalyzer()
        self.strategies = ScalpingStrategies(self.analyzer)

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
            notes="test.py run",
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
                msg = f"✅ Бот запущен в режиме {self.mode.value}"
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

            # на Unified/linear корректнее делать reduceOnly рыночным лимитом IOC:
            # для простоты используем Market + reduceOnly=False (Bybit сам скроет в позицию)
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

        # 0) уведомление о закрытии позиции (детектор)
        try:
            pos_now = self.current_trader.get_position(self.symbol)
            now_open = bool(pos_now and pos_now.size > 0)
            if self._last_pos_open and not now_open:
                try:
                    await self.send_telegram_message(f"❌ Позиция закрыта (SL/TP/ручное). Символ: {self.symbol}")
                except Exception as e:
                    logging.error(f"TG close notify error: {e}")
            # обновить слепок
            self._last_pos_open = now_open
            self._last_pos_side = pos_now.side if now_open else None
            self._last_pos_entry = float(pos_now.entry_price) if now_open else None
            self._last_pos_size = float(pos_now.size) if now_open else 0.0
        except Exception as e:
            logging.error(f"Close detector error: {e}")

        try:
            # 1) данные рынка (1m, 100 свечей)
            df = self.current_trader.get_klines(self.symbol, "1", 100)
            if df.empty:
                logging.warning("Пустые свечи 1m")
                return

            market = self.analyzer.analyze_market(df)
            regime_1m = "UP" if market["price"] > market["indicators"]["ema_fast"] else "DOWN"

            # фичи для БД
            features = {
                "rsi": float(market["indicators"]["rsi"]),
                "ema_fast": float(market["indicators"]["ema_fast"]),
                "ema_slow": float(market["indicators"]["ema_slow"]),
                "bb_upper": float(market["indicators"]["bb_upper"]),
                "bb_lower": float(market["indicators"]["bb_lower"]),
                "atr": float(market["signals"]["atr"]),
                "volume_spike": bool(market["signals"]["volume_spike"]),
                "macd_bullish": bool(market["signals"]["macd_bullish"]),
                "macd_bearish": bool(market["signals"]["macd_bearish"]),
                "rsi_oversold": bool(market["signals"]["rsi_oversold"]),
                "rsi_overbought": bool(market["signals"]["rsi_overbought"]),
            }

            # 2) сигнал
            signal = self.strategies.get_best_signal(market)

            # лог сигнала (даже если None — для частоты)
            try:
                self.adb.add_signal(
                    ts=datetime.now(UTC),
                    symbol=self.symbol,
                    side=(signal.side.value if signal else "None"),
                    entry_estimate=(float(signal.entry_price) if signal else None),
                    sl_estimate=(float(signal.stop_loss) if signal else None),
                    tp_estimate=(float(signal.take_profit) if signal else None),
                    confidence=(float(signal.confidence) if signal else None),
                    strategy=(signal.strategy_name if signal else "no_signal"),
                    features=features,
                    regime_1m=regime_1m,
                    regime_5m=None,
                    sr_1m=None,
                    sr_5m=None,
                    trade_id=None,
                )
            except Exception as e:
                logging.error(f"[ADB] save signal error: {e}")

            if not signal:
                return

            # 2.5) ранний детектор разворота: если есть позиция и пришёл сильный обратный сигнал — закрываем
            try:
                pos_chk = self.current_trader.get_position(self.symbol)
                if pos_chk and pos_chk.size > 0:
                    if self._is_reversal_signal(market, signal, pos_chk):
                        logging.info("[REVERSAL] Обнаружен сильный противоположный сигнал — закрываем позицию по рынку")
                        ok, txt = await self._close_position_and_clear()
                        try:
                            await self.send_telegram_message("⛔ Разворот: позиция закрыта по рынку (противосигнал)")
                        except Exception:
                            pass
                        # после закрытия даём бирже отразить состояние и выходим из цикла
                        return
            except Exception as e:
                logging.error(f"[REVERSAL] error: {e}")

            # 3) позиция уже открыта?
            pos = self.current_trader.get_position(self.symbol)
            if pos and pos.size > 0:
                logging.info("Уже есть открытая позиция, пропускаем сигнал")
                return

            # 4) размер позиции — по номиналу USDT
            price = float(market["price"])
            bal = self.current_trader.get_balance()
            free = float(bal.get("totalAvailableBalance", 0.0))
            target_notional = min(self.max_notional_usdt, free)
            if target_notional <= 0:
                logging.warning("Недостаточно средств")
                return
            raw_qty = target_notional / max(price, 1e-8)
            norm_qty = self.current_trader.normalize_qty(self.symbol, raw_qty)
            if norm_qty <= 0:
                logging.warning(f"Нельзя нормализовать qty (raw={raw_qty})")
                return
            qty_str = f"{norm_qty:.8f}"

            # --- Per-symbol TP/SL override ---
            atr_val = float(market["signals"].get("atr", 0.0)) if isinstance(market.get("signals", {}).get("atr", 0.0), (int, float)) else float(market["signals"]["atr"])
            meta = self.current_trader.get_symbol_meta(self.symbol)
            calc_sl, calc_tp = compute_tpsl_for_symbol(
                symbol=self.symbol,
                side=signal.side.value,
                entry=price,
                atr_value=atr_val,
                meta=meta,
            )
            sl = float(calc_sl) if calc_sl is not None else float(signal.stop_loss)
            tp = float(calc_tp) if calc_tp is not None else float(signal.take_profit)

            # 5) лог попытки открытия
            try:
                self.adb.add_order_attempt(
                    ts=datetime.now(UTC),
                    intent="open",
                    side=signal.side.value,
                    order_type="Limit-IOC",
                    price=None,
                    qty=float(qty_str),
                    stop_loss=sl,
                    take_profit=tp,
                    ret_code=None,
                    ret_msg=None,
                    payload={"category": "linear", "symbol": self.symbol, "side": signal.side.value,
                             "orderType": "Limit-IOC", "qty": qty_str, "stopLoss": sl, "takeProfit": tp},
                    response={},
                    trade_id=None,
                )
            except Exception as e:
                logging.error(f"[ADB] save order attempt error: {e}")

            # 6) открытие
            try:
                # на тестнете иногда висят стопы/активные ордера пред. сделки — очистим перед входом
                self.current_trader.client.cancel_all_orders(category="linear", symbol=self.symbol)
                self.current_trader.client.cancel_all_orders(category="linear", symbol=self.symbol, orderFilter="StopOrder")
            except Exception:
                pass
            result = self.current_trader.place_market_order(
                symbol=self.symbol,
                side=signal.side.value,
                qty=qty_str,
                stop_loss=sl,
                take_profit=tp,
            )

            # 7) лог ответа
            try:
                self.adb.add_order_attempt(
                    ts=datetime.now(UTC),
                    intent="open",
                    side=signal.side.value,
                    order_type="Limit-IOC",
                    price=None,
                    qty=float(qty_str),
                    stop_loss=sl,
                    take_profit=tp,
                    ret_code=result.get("retCode"),
                    ret_msg=result.get("retMsg"),
                    payload={},
                    response=result,
                    trade_id=None,
                )
            except Exception as e:
                logging.error(f"[ADB] save order response error: {e}")

            if result.get("retCode") == -2:
                logging.warning("[OPEN] Не удалось исполнить IOC — позиция не открыта (перепроверим на следующем цикле)")
                return
            if result.get("retCode") != 0:
                logging.error(f"[ORDER] failed: {result}")
                return

            # подтверждение позиции
            pos = self.current_trader.get_position(self.symbol)
            if not pos or pos.size <= 0:
                real_entry = float(result.get("avgPrice") or 0.0)
                if real_entry > 0:
                    logging.warning("[OPEN] fill подтверждён по ордеру, но позиция ещё не отобразилась — уведомим без SL/TP, попробуем позже.")
                else:
                    logging.warning("[OPEN] retCode=0, но позиция отсутствует — не удалось подтвердить исполнение")
                    return
            else:
                real_entry = float(pos.entry_price)

            # 8) регистрируем trade
            trade_id = None
            try:
                trade_id = self.adb.new_trade(
                    ts_open=datetime.now(UTC),
                    symbol=self.symbol,
                    side=signal.side.value,
                    qty=float(qty_str),
                    avg_entry_price=real_entry,
                    strategy=signal.strategy_name,
                    mode=self.mode.value,
                    risk_cfg=self.risk_cfg,
                )
            except Exception as e:
                logging.error(f"[ADB] new_trade error: {e}")

            # 9) уведомление
            await self.send_telegram_message(
                "🚀 Открыта позиция\n"
                f"Символ: {self.symbol}\n"
                f"Сторона: {signal.side.value}\n"
                f"Размер: {qty_str}\n"
                f"Вход: {real_entry:.2f}\n"
                f"SL: {sl:.2f}\n"
                f"TP: {tp:.2f}\n"
                f"Стратегия: {signal.strategy_name}\n"
                f"Режим: {self.mode.value}"
            )

            # 10) обновляем слепок
            self._last_pos_open = bool(pos and pos.size > 0)
            self._last_pos_side = (pos.side if (pos and pos.size > 0) else signal.side.value)
            self._last_pos_entry = real_entry
            self._last_pos_size = (float(pos.size) if (pos and pos.size > 0) else float(qty_str))

            # 11) снапшот в БД
            try:
                if pos and pos.size > 0:
                    bal = self.current_trader.get_balance()
                    self.adb.add_snapshot(
                        trade_id=trade_id,
                        ts=datetime.now(UTC),
                        last_price=float(market["price"]),
                        position_side=pos.side,
                        position_size=float(pos.size),
                        position_entry=float(pos.entry_price),
                        unrealized_pnl=float(pos.unrealized_pnl),
                        balance_available=self.current_trader._safe_float(bal.get("totalAvailableBalance")),
                        raw_json={"position": pos.__dict__, "balance": bal},
                    )
            except Exception as e:
                logging.error(f"[ADB] snapshot error: {e}")

        except Exception as e:
            logging.error(f"analyze_and_trade error: {e}")
            self.status = BotStatus.ERROR


# ---------------------- MAIN ----------------------
def main():
    logging.info("=== Bot starting === (TESTNET by default)")
    bot = ScalpingBot()

    # Telegram app; run_polling сам управляет лупом — без ошибок закрытия лупа
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", bot.start_command))
    app.add_handler(CallbackQueryHandler(bot.button_callback))

    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
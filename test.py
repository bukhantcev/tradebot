# Агрессивный скальпинг-бот для Bybit с Telegram управлением
# КРАЙНЕ РИСКОВАННАЯ ТОРГОВАЯ СИСТЕМА
# Используйте только с деньгами, которые можете потерять!

import asyncio
import os
import sqlite3
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP
import talib
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from test_db import AnalyticsDB, RiskConfig
UTC = timezone.utc

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Настройки (создайте файл config.py)

BYBIT_API_KEY_MAIN = "your_main_api_key"
BYBIT_API_SECRET_MAIN = "your_main_secret"
BYBIT_API_KEY_TEST = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET_TEST = os.getenv("BYBIT_API_SECRET", "")

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TG_TOKEN_LOCAL", "")
TELEGRAM_CHAT_ID = os.getenv("TG_ADMIN_CHAT_ID", "")

# Trading
SYMBOL = "BTCUSDT"
RISK_PERCENT = 20  # Максимальный риск на сделку в %
MIN_PROFIT_RATIO = 1.5  # Минимальное соотношение прибыль/риск
BASE_LEVERAGE = 10



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


class DatabaseManager:
    def __init__(self, db_path: str = "scalping_bot.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Инициализация базы данных"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    size REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    pnl REAL,
                    strategy TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT DEFAULT 'open',
                    stop_loss REAL,
                    take_profit REAL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    mode TEXT NOT NULL,
                    balance REAL NOT NULL,
                    daily_pnl REAL NOT NULL,
                    total_trades INTEGER NOT NULL,
                    winning_trades INTEGER NOT NULL,
                    losing_trades INTEGER NOT NULL,
                    max_drawdown REAL NOT NULL
                )
            """)

            conn.commit()

    def save_trade(self, trade_data: Dict):
        """Сохранение сделки"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO trades (symbol, side, size, entry_price, exit_price, 
                                  pnl, strategy, mode, status, stop_loss, take_profit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data['symbol'], trade_data['side'], trade_data['size'],
                trade_data['entry_price'], trade_data.get('exit_price'),
                trade_data.get('pnl'), trade_data['strategy'], trade_data['mode'],
                trade_data['status'], trade_data.get('stop_loss'),
                trade_data.get('take_profit')
            ))
            conn.commit()

    def get_daily_stats(self, mode: str) -> Dict:
        """Получение дневной статистики"""
        with sqlite3.connect(self.db_path) as conn:
            today = datetime.now().date()
            cursor = conn.execute("""
                SELECT COUNT(*) as total_trades,
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                       SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                       COALESCE(SUM(pnl), 0) as total_pnl,
                       COALESCE(AVG(pnl), 0) as avg_pnl
                FROM trades 
                WHERE DATE(timestamp) = ? AND mode = ? AND status = 'closed'
            """, (today, mode))

            return dict(cursor.fetchone())


class TechnicalAnalyzer:
    def __init__(self):
        self.indicators = {}

    def analyze_market(self, df: pd.DataFrame) -> Dict:
        """Комплексный анализ рынка"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        # Технические индикаторы
        rsi = talib.RSI(close, timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(close)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
        ema_fast = talib.EMA(close, timeperiod=9)
        ema_slow = talib.EMA(close, timeperiod=21)
        atr = talib.ATR(high, low, close, timeperiod=14)

        current_price = close[-1]

        # Сигналы для скальпинга
        signals = {
            'rsi_oversold': rsi[-1] < 30,
            'rsi_overbought': rsi[-1] > 70,
            'macd_bullish': macd[-1] > macd_signal[-1] and macd_hist[-1] > 0,
            'macd_bearish': macd[-1] < macd_signal[-1] and macd_hist[-1] < 0,
            'price_above_ema_fast': current_price > ema_fast[-1],
            'price_below_ema_fast': current_price < ema_fast[-1],
            'bb_oversold': current_price < bb_lower[-1],
            'bb_overbought': current_price > bb_upper[-1],
            'volume_spike': volume[-1] > np.mean(volume[-20:]) * 1.5,
            'atr': atr[-1]
        }

        return {
            'price': current_price,
            'signals': signals,
            'indicators': {
                'rsi': rsi[-1],
                'macd': macd[-1],
                'ema_fast': ema_fast[-1],
                'ema_slow': ema_slow[-1],
                'bb_upper': bb_upper[-1],
                'bb_lower': bb_lower[-1],
                'atr': atr[-1]
            }
        }


class ScalpingStrategies:
    def __init__(self, analyzer: TechnicalAnalyzer):
        self.analyzer = analyzer

    def rsi_mean_reversion(self, market_data: Dict) -> Optional[TradeSignal]:
        """Стратегия разворота по RSI"""
        signals = market_data['signals']
        price = market_data['price']
        atr = signals['atr']

        if signals['rsi_oversold'] and signals['volume_spike']:
            # Покупка на перепроданности
            entry_price = price
            stop_loss = entry_price - (atr * 1.5)
            take_profit = entry_price + (atr * 2.0)

            return TradeSignal(
                side=OrderSide.BUY,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=0.7,
                strategy_name="rsi_mean_reversion_buy"
            )

        elif signals['rsi_overbought'] and signals['volume_spike']:
            # Продажа на перекупленности
            entry_price = price
            stop_loss = entry_price + (atr * 1.5)
            take_profit = entry_price - (atr * 2.0)

            return TradeSignal(
                side=OrderSide.SELL,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=0.7,
                strategy_name="rsi_mean_reversion_sell"
            )

        return None

    def macd_momentum(self, market_data: Dict) -> Optional[TradeSignal]:
        """Стратегия импульса по MACD"""
        signals = market_data['signals']
        price = market_data['price']
        atr = signals['atr']

        if signals['macd_bullish'] and signals['price_above_ema_fast']:
            entry_price = price
            stop_loss = entry_price - (atr * 1.0)
            take_profit = entry_price + (atr * 1.5)

            return TradeSignal(
                side=OrderSide.BUY,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=0.6,
                strategy_name="macd_momentum_buy"
            )

        elif signals['macd_bearish'] and signals['price_below_ema_fast']:
            entry_price = price
            stop_loss = entry_price + (atr * 1.0)
            take_profit = entry_price - (atr * 1.5)

            return TradeSignal(
                side=OrderSide.SELL,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=0.6,
                strategy_name="macd_momentum_sell"
            )

        return None

    def bollinger_breakout(self, market_data: Dict) -> Optional[TradeSignal]:
        """Стратегия прорыва полос Боллинджера"""
        signals = market_data['signals']
        price = market_data['price']
        atr = signals['atr']

        if signals['bb_overbought'] and signals['volume_spike']:
            # Прорыв верхней полосы
            entry_price = price
            stop_loss = entry_price - (atr * 2.0)
            take_profit = entry_price + (atr * 1.0)

            return TradeSignal(
                side=OrderSide.BUY,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=0.5,
                strategy_name="bb_breakout_buy"
            )

        elif signals['bb_oversold'] and signals['volume_spike']:
            # Прорыв нижней полосы
            entry_price = price
            stop_loss = entry_price + (atr * 2.0)
            take_profit = entry_price - (atr * 1.0)

            return TradeSignal(
                side=OrderSide.SELL,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=0.5,
                strategy_name="bb_breakout_sell"
            )

        return None

    def get_best_signal(self, market_data: Dict) -> Optional[TradeSignal]:
        """Получение лучшего сигнала из всех стратегий"""
        strategies = [
            self.rsi_mean_reversion,
            self.macd_momentum,
            self.bollinger_breakout
        ]

        signals = []
        for strategy in strategies:
            signal = strategy(market_data)
            if signal:
                signals.append(signal)

        if not signals:
            return None

        # Выбираем сигнал с наивысшей уверенностью
        best_signal = max(signals, key=lambda s: s.confidence)

        # Проверяем соотношение риск/прибыль
        risk = abs(best_signal.entry_price - best_signal.stop_loss)
        profit = abs(best_signal.take_profit - best_signal.entry_price)

        if profit / risk < 1.2:  # Минимальное соотношение 1:1.2
            return None

        return best_signal


class BybitTrader:
    def _ensure_tp_sl_valid(self, symbol: str, side: str, entry_price: float, stop_loss: Optional[float], take_profit: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
        """
        Bybit rules:
        - For Buy (long): SL < entry, TP > entry
        - For Sell (short): SL > entry, TP < entry
        Adjust levels to be at least 1 tick away and normalize to tick.
        """
        meta = self.get_symbol_meta(symbol)
        tick = meta.get("tick_size", 0.0) or 0.1

        # guard
        if entry_price is None or entry_price == 0:
            return (stop_loss, take_profit)

        if str(side).lower() == "buy":
            # SL must be below, TP above
            if stop_loss is not None and stop_loss >= entry_price:
                stop_loss = entry_price - tick
            if take_profit is not None and take_profit <= entry_price:
                take_profit = entry_price + tick
        else:
            # sell / short: SL above, TP below
            if stop_loss is not None and stop_loss <= entry_price:
                stop_loss = entry_price + tick
            if take_profit is not None and take_profit >= entry_price:
                take_profit = entry_price - tick

        # normalize to tick
        sl_norm = self.normalize_price(symbol, float(stop_loss), side=None) if stop_loss is not None else None
        tp_norm = self.normalize_price(symbol, float(take_profit), side=None) if take_profit is not None else None
        return (sl_norm, tp_norm)
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.client = HTTP(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet
        )
        self.testnet = testnet

    def _safe_float(self, v, default=0.0):
        try:
            if v is None or v == "":
                return float(default)
            return float(v)
        except Exception:
            return float(default)

    def get_symbol_meta(self, symbol: str) -> dict:
        try:
            res = self.client.get_instruments_info(category="linear", symbol=symbol)
            if res.get("retCode") == 0 and res.get("result", {}).get("list"):
                item = res["result"]["list"][0]
                lot = item.get("lotSizeFilter", {})
                price_filter = item.get("priceFilter", {})
                return {
                    "min_qty": self._safe_float(lot.get("minOrderQty"), 0.0),
                    "qty_step": self._safe_float(lot.get("qtyStep"), 0.0),
                    "min_notional": self._safe_float(lot.get("minOrderAmt"), 0.0),
                    "tick_size": float(self._safe_float(price_filter.get("tickSize"), 0.0)),
                }
        except Exception as e:
            logging.error(f"Ошибка получения meta по символу: {e}")
        return {"min_qty": 0.0, "qty_step": 0.0, "min_notional": 0.0, "tick_size": 0.0}

    def normalize_qty(self, symbol: str, qty: float) -> float:
        meta = self.get_symbol_meta(symbol)
        step = meta["qty_step"] or 0.001
        minq = meta["min_qty"] or step
        import math
        normalized = math.floor(float(qty) / float(step)) * float(step)
        if normalized < minq:
            normalized = minq
        return float(f"{normalized:.8f}")

    def normalize_price(self, symbol: str, price: float, side: Optional[str] = None, adjust: bool = False) -> float:
        meta = self.get_symbol_meta(symbol)
        tick = meta.get("tick_size", 0.0) or 0.1
        if tick <= 0:
            return float(price)

        # decimal places from tick size (e.g., 0.1 -> 1, 0.01 -> 2)
        tick_str = f"{tick}"
        decimals = len(tick_str.split(".")[1]) if "." in tick_str else 0

        # nearest tick
        steps = round(float(price) / tick)
        norm = steps * tick

        if adjust and side:
            if str(side).lower() == "buy":
                # push 1 tick "up" to satisfy IOC on asks
                norm = (int(round(norm / tick)) + 1) * tick
            else:
                # push 1 tick "down" to satisfy IOC on bids
                norm = (int(round(norm / tick)) - 1) * tick

        return float(f"{norm:.{decimals}f}")

    def get_balance(self) -> Dict:
        """Получение баланса"""
        try:
            result = self.client.get_wallet_balance(accountType="UNIFIED")
            return result['result']['list'][0] if result['retCode'] == 0 else {}
        except Exception as e:
            logging.error(f"Ошибка получения баланса: {e}")
            return {}

    def get_position(self, symbol: str) -> Optional[Position]:
        """Получение позиции"""
        try:
            result = self.client.get_positions(category="linear", symbol=symbol)
            if result['retCode'] == 0 and result['result']['list']:
                pos_data = result['result']['list'][0]
                return Position(
                    symbol=pos_data.get('symbol', symbol),
                    side=pos_data.get('side', ''),
                    size=self._safe_float(pos_data.get('size')),
                    entry_price=self._safe_float(pos_data.get('avgPrice')),
                    unrealized_pnl=self._safe_float(pos_data.get('unrealisedPnl')),
                    percentage=self._safe_float(pos_data.get('percentage'))
                )
        except Exception as e:
            logging.error(f"Ошибка получения позиции: {e}")
        return None

    def place_market_order(self, symbol: str, side: str, qty: str,
                           stop_loss: float = None, take_profit: float = None) -> Dict:
        """
        Безопасное открытие:
        1) Получаем стакан.
        2) Открываем через IOC-Limit БЕЗ SL/TP (чтобы не ловить 30208/10001).
        3) После успешного открытия — отдельно ставим SL/TP через set_trading_stop.
        Фолбэк: при отказе двигаем цену ещё на +1..+2 тика "внутрь".
        """
        # 1) референсная цена от стакана
        try:
            ob = self.client.get_orderbook(category="linear", symbol=symbol, limit=1)
            if ob.get("retCode") != 0:
                return {'retCode': -1, 'retMsg': f"orderbook retCode={ob.get('retCode')}"}
            best_bid = self._safe_float(ob["result"]["b"][0][0])
            best_ask = self._safe_float(ob["result"]["a"][0][0])
        except Exception as e:
            return {'retCode': -1, 'retMsg': f"orderbook error: {e}"}

        ref_price = best_ask if str(side).lower() == "buy" else best_bid
        base_price = self.normalize_price(symbol, ref_price, side=side, adjust=True)

        def _open_ioc(adjust_ticks: int) -> Dict:
            meta = self.get_symbol_meta(symbol)
            tick = meta.get("tick_size", 0.1) or 0.1
            p = base_price + adjust_ticks * tick if str(side).lower() == "buy" else base_price - adjust_ticks * tick
            p = self.normalize_price(symbol, p)
            return self.client.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Limit",
                timeInForce="IOC",
                price=str(p),
                qty=qty,
                reduceOnly=False
            )

        # 2) пробуем открыть на 0 / +1 / +2 тика (или - для sell)
        attempts = [0, 1, 2]
        res_open = None
        last_err = None
        for a in attempts:
            try:
                res_open = _open_ioc(a)
                if res_open.get("retCode") == 0:
                    break
                last_err = res_open
            except Exception as e:
                last_err = {'retCode': -1, 'retMsg': str(e)}
                # если 30208 пришёл как текст исключения — идём дальше
                if "30208" in str(e):
                    continue
                else:
                    break

        if not res_open or res_open.get("retCode") != 0:
            return last_err or {'retCode': -1, 'retMsg': 'open failed'}

        # 3) подтверждаем, что позиция реально открылась (IOC мог не исполниться)
        pos = None
        for _ in range(6):  # ~1.8s макс
            time.sleep(0.3)
            pos = self.get_position(symbol)
            if pos and pos.size > 0:
                break
        if not pos or pos.size <= 0:
            # позиция не открылась — ничего не ставим, возвращаем спец-код
            return {**(res_open or {}), 'retCode': -2, 'retMsg': 'ioc not filled'}

        # 4) ставим SL/TP уже от фактической цены входа (с ретраями и явными параметрами)
        sl_norm, tp_norm = self._ensure_tp_sl_valid(symbol, side, float(pos.entry_price), stop_loss, take_profit)
        if sl_norm is not None or tp_norm is not None:
            meta = self.get_symbol_meta(symbol)
            # для one-way позиции используем positionIdx=0; если у тебя hedge-mode, можно маппить по стороне (Buy->1, Sell->2)
            position_idx = 0
            last_err = None
            for attempt in range(5):
                try:
                    # небольшая задержка — даём бирже зафиксировать позицию
                    time.sleep(0.3)
                    res_tpsl = self.client.set_trading_stop(
                        category="linear",
                        symbol=symbol,
                        positionIdx=position_idx,
                        tpslMode="Full",
                        triggerBy="LastPrice",
                        stopLoss=(str(sl_norm) if sl_norm is not None else None),
                        takeProfit=(str(tp_norm) if tp_norm is not None else None),
                    )
                    if res_tpsl and res_tpsl.get("retCode") == 0:
                        break
                    last_err = res_tpsl
                except Exception as e:
                    last_err = {"retCode": -1, "retMsg": str(e)}
                # если биржа говорит "zero position" — возможно, ещё не успела обновиться; пробуем снова
            if last_err and last_err.get("retCode") not in (0, None):
                logging.error(f"Не удалось выставить SL/TP после открытия: {last_err}")

        return {**res_open, 'filled': True, 'avgPrice': float(pos.entry_price)}

    def get_klines(self, symbol: str, interval: str = "1", limit: int = 200) -> pd.DataFrame:
        """Получение свечей"""
        try:
            result = self.client.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit
            )

            if result['retCode'] == 0:
                data = result['result']['list']
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                df = df.astype({
                    'open': float, 'high': float, 'low': float,
                    'close': float, 'volume': float
                })
                return df.sort_values('timestamp').reset_index(drop=True)
        except Exception as e:
            logging.error(f"Ошибка получения свечей: {e}")

        return pd.DataFrame()


class ScalpingBot:
    def __init__(self):
        self.db = DatabaseManager()
        self.analyzer = TechnicalAnalyzer()
        self.strategies = ScalpingStrategies(self.analyzer)

        # Analytics DB
        self.adb = AnalyticsDB("analytics.db")

        # Торговые клиенты
        self.trader_main = BybitTrader(BYBIT_API_KEY_MAIN, BYBIT_API_SECRET_MAIN, False)
        self.trader_test = BybitTrader(BYBIT_API_KEY_TEST, BYBIT_API_SECRET_TEST, True)

        # Состояние бота
        self.status = BotStatus.STOPPED
        self.mode = TradingMode.TESTNET
        self.symbol = SYMBOL
        self.risk_percent = RISK_PERCENT
        self.max_notional_usdt = float(os.getenv("MAX_NOTIONAL_USDT", "10"))
        self.risk_cfg = RiskConfig(max_notional_usdt=self.max_notional_usdt, leverage=BASE_LEVERAGE, notes="test.py run")

        # Telegram
        self.telegram_token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID

        # Планировщик
        self.scheduler = AsyncIOScheduler()

        # Статистика
        self.stats = {
            'trades_today': 0,
            'pnl_today': 0.0,
            'winning_trades': 0,
            'losing_trades': 0
        }

        # Трекинг состояния позиции (для уведомлений о закрытии)
        self._last_pos_open = False
        self._last_pos_side = None
        self._last_pos_entry = None
        self._last_pos_size = 0.0

    @property
    def current_trader(self):
        return self.trader_test if self.mode == TradingMode.TESTNET else self.trader_main

    async def analyze_and_trade(self):
        """Основной цикл анализа и торговли"""
        if self.status != BotStatus.RUNNING:
            return

        # Проверка: если в прошлом циклe позиция была открыта, а теперь закрыта — шлём уведомление о закрытии
        try:
            pos_now = self.current_trader.get_position(self.symbol)
            now_open = bool(pos_now and pos_now.size > 0)
            if self._last_pos_open and not now_open:
                # позиция закрылась (SL/TP/ручное)
                try:
                    await self.send_telegram_message(
                        "❌ Позиция закрыта (SL/TP/ручное). \nСимвол: %s" % self.symbol
                    )
                except Exception as _e:
                    logging.error(f"Ошибка отправки уведомления о закрытии позиции: {_e}")
            # обновляем слепок состояния для следующего тика
            self._last_pos_open = now_open
            self._last_pos_side = pos_now.side if now_open else None
            self._last_pos_entry = float(pos_now.entry_price) if now_open else None
            self._last_pos_size = float(pos_now.size) if now_open else 0.0
        except Exception as _e:
            logging.error(f"Проверка закрытия позиции: {_e}")

        try:
            # Получаем данные рынка
            df = self.current_trader.get_klines(self.symbol, "1", 100)
            if df.empty:
                logging.warning("Не удалось получить данные рынка")
                return

            # Анализируем рынок
            market_data = self.analyzer.analyze_market(df)

            # Precompute features for analytics logging
            features = {
                "rsi": float(market_data["indicators"]["rsi"]),
                "ema_fast": float(market_data["indicators"]["ema_fast"]),
                "ema_slow": float(market_data["indicators"]["ema_slow"]),
                "bb_upper": float(market_data["indicators"]["bb_upper"]),
                "bb_lower": float(market_data["indicators"]["bb_lower"]),
                "atr": float(market_data["signals"]["atr"]),
                "volume_spike": bool(market_data["signals"]["volume_spike"]),
                "macd_bullish": bool(market_data["signals"]["macd_bullish"]),
                "macd_bearish": bool(market_data["signals"]["macd_bearish"]),
                "rsi_oversold": bool(market_data["signals"]["rsi_oversold"]),
                "rsi_overbought": bool(market_data["signals"]["rsi_overbought"]),
            }
            regime_1m = "UP" if market_data["price"] > market_data["indicators"]["ema_fast"] else "DOWN"

            # Получаем сигнал
            signal = self.strategies.get_best_signal(market_data)

            # Логируем сам факт сгенерированного сигнала (или отсутствие) для последующего анализа
            sig_id = None
            if signal:
                try:
                    sig_id = self.adb.add_signal(
                        ts=datetime.now(UTC),
                        symbol=self.symbol,
                        side=signal.side.value,
                        entry_estimate=float(signal.entry_price),
                        sl_estimate=float(signal.stop_loss) if signal.stop_loss else None,
                        tp_estimate=float(signal.take_profit) if signal.take_profit else None,
                        confidence=float(signal.confidence),
                        strategy=signal.strategy_name,
                        features=features,
                        regime_1m=regime_1m,
                        regime_5m=None,
                        sr_1m=None,
                        sr_5m=None,
                        trade_id=None,
                    )
                except Exception as e:
                    logging.error(f"[ADB] не удалось сохранить сигнал: {e}")
            else:
                # Записываем нулевой сигнал (optional) — поможет считать частоту
                try:
                    self.adb.add_signal(
                        ts=datetime.now(UTC),
                        symbol=self.symbol,
                        side="None",
                        entry_estimate=None,
                        sl_estimate=None,
                        tp_estimate=None,
                        confidence=None,
                        strategy="no_signal",
                        features=features,
                        regime_1m=regime_1m,
                        regime_5m=None,
                        sr_1m=None,
                        sr_5m=None,
                        trade_id=None,
                    )
                except Exception as e:
                    logging.error(f"[ADB] не удалось сохранить empty-сигнал: {e}")

            if not signal:
                return

            # Проверяем существующую позицию
            position = self.current_trader.get_position(self.symbol)
            if position and position.size > 0:
                logging.info("Уже есть открытая позиция, пропускаем сигнал")
                return

            current_price = market_data['price']
            # целевой номинал позиции: не больше max_notional_usdt
            target_notional = min(self.max_notional_usdt, float(self.current_trader.get_balance().get('totalAvailableBalance', 0)))
            if target_notional <= 0:
                logging.warning("Недостаточно средств для торговли")
                return
            raw_qty = target_notional / max(current_price, 1e-8)
            norm_qty = self.current_trader.normalize_qty(self.symbol, raw_qty)
            if norm_qty <= 0:
                logging.warning(f"Невозможно нормализовать размер позиции (raw={raw_qty})")
                return
            qty = f"{norm_qty:.8f}"

            sl = float(signal.stop_loss) if signal.stop_loss else None
            tp = float(signal.take_profit) if signal.take_profit else None

            # Логируем попытку открытия до факта
            try:
                self.adb.add_order_attempt(
                    ts=datetime.now(UTC),
                    intent="open",
                    side=signal.side.value,
                    order_type="Market",
                    price=None,
                    qty=float(qty),
                    stop_loss=sl,
                    take_profit=tp,
                    ret_code=None,
                    ret_msg=None,
                    payload={"category":"linear","symbol":self.symbol,"side":signal.side.value,"orderType":"Market","qty":qty,"stopLoss":sl,"takeProfit":tp},
                    response={},
                    trade_id=None,
                )
            except Exception as e:
                logging.error(f"[ADB] не удалось сохранить попытку ордера: {e}")

            result = self.current_trader.place_market_order(
                symbol=self.symbol,
                side=signal.side.value,
                qty=qty,
                stop_loss=sl,
                take_profit=tp
            )

            try:
                self.adb.add_order_attempt(
                    ts=datetime.now(UTC),
                    intent="open",
                    side=signal.side.value,
                    order_type="Market",
                    price=None,
                    qty=float(qty),
                    stop_loss=sl,
                    take_profit=tp,
                    ret_code=result.get('retCode'),
                    ret_msg=result.get('retMsg'),
                    payload={},
                    response=result,
                    trade_id=None,
                )
            except Exception as e:
                logging.error(f"[ADB] не удалось записать ответ по ордеру: {e}")

            # Сначала: спец-код -2 (IOC не исполнился)
            if result.get('retCode') == -2:
                logging.warning("[OPEN] not filled (IOC), позиция не открыта")
                return
            # Затем: ошибка по retCode
            if result.get('retCode') != 0:
                logging.error(f"[ORDER] failed: {result}")
                return

            # подтверждаем, что позиция открыта (на всякий случай)
            pos = self.current_trader.get_position(self.symbol)
            if not pos or pos.size <= 0:
                logging.warning("[OPEN] order retCode=0, но позиция отсутствует — пропускаем уведомление/SLTP")
                return
            real_entry = float(pos.entry_price) if hasattr(pos, 'entry_price') else float(signal.entry_price)

            # Успешно: регистрируем сделку
            trade_id = None
            try:
                trade_id = self.adb.new_trade(
                    ts_open=datetime.now(UTC),
                    symbol=self.symbol,
                    side=signal.side.value,
                    qty=float(qty),
                    avg_entry_price=real_entry,  # теперь используем фактическую цену входа
                    strategy=signal.strategy_name,
                    mode=self.mode.value,
                    risk_cfg=self.risk_cfg,
                )
                if sig_id is not None:
                    self.adb.link_signal_to_trade(sig_id, trade_id)
            except Exception as e:
                logging.error(f"[ADB] не удалось сохранить trade/open: {e}")

            # Отправляем уведомление в Telegram
            await self.send_telegram_message(
                f"🚀 Открыта позиция\n"
                f"Символ: {self.symbol}\n"
                f"Сторона: {signal.side.value}\n"
                f"Размер: {qty}\n"
                f"Цена входа: {real_entry:.2f}\n"
                f"Стоп-лосс: {signal.stop_loss:.2f}\n"
                f"Тейк-профит: {signal.take_profit:.2f}\n"
                f"Стратегия: {signal.strategy_name}\n"
                f"Режим: {self.mode.value}"
            )

            # После успешной отправки — обновляем флаги состояния позиции
            self._last_pos_open = True
            self._last_pos_side = pos.side
            self._last_pos_entry = real_entry
            self._last_pos_size = float(pos.size)

            # Snapshot после открытия (best-effort)
            try:
                bal = self.current_trader.get_balance()
                self.adb.add_snapshot(
                    trade_id=trade_id,
                    ts=datetime.now(UTC),
                    last_price=float(market_data['price']),
                    position_side=(pos.side if pos else None),
                    position_size=(float(pos.size) if pos else None),
                    position_entry=(float(pos.entry_price) if pos else None),
                    unrealized_pnl=(float(pos.unrealized_pnl) if pos else None),
                    balance_available=self.current_trader._safe_float(bal.get('totalAvailableBalance')),
                    raw_json={"position": pos.__dict__ if pos else None, "balance": bal},
                )
            except Exception as e:
                logging.error(f"[ADB] snapshot error: {e}")

            logging.info(f"Открыта позиция: {signal.side.value} {qty} {self.symbol}")

        except Exception as e:
            logging.error(f"Ошибка в analyze_and_trade: {e}")
            self.status = BotStatus.ERROR

    async def send_telegram_message(self, text: str):
        """Отправка сообщения в Telegram"""
        try:
            from telegram import Bot
            bot = Bot(token=self.telegram_token)
            await bot.send_message(chat_id=self.chat_id, text=text)
        except Exception as e:
            logging.error(f"Ошибка отправки Telegram сообщения: {e}")

    def get_keyboard(self):
        """Создание клавиатуры для Telegram"""
        keyboard = [
            [
                InlineKeyboardButton("▶️ Старт" if self.status == BotStatus.STOPPED else "⏸️ Стоп",
                                     callback_data="toggle_bot"),
                InlineKeyboardButton("💰 Баланс", callback_data="balance")
            ],
            [
                InlineKeyboardButton("📊 Статус", callback_data="status"),
                InlineKeyboardButton("📈 Статистика", callback_data="stats")
            ],
            [
                InlineKeyboardButton("🧪 Testnet" if self.mode == TradingMode.MAINNET else "💸 Mainnet",
                                     callback_data="toggle_mode")
            ]
        ]
        return InlineKeyboardMarkup(keyboard)

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /start"""
        await update.message.reply_text(
            "🤖 Агрессивный скальпинг-бот запущен!\n\n"
            "⚠️ ВНИМАНИЕ: Данный бот предназначен для агрессивной торговли "
            "и может привести к значительным убыткам!\n\n"
            "Используйте только средства, которые можете себе позволить потерять.",
            reply_markup=self.get_keyboard()
        )

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка нажатий кнопок"""
        query = update.callback_query
        await query.answer()

        if query.data == "toggle_bot":
            if self.status == BotStatus.STOPPED:
                self.status = BotStatus.RUNNING
                self.scheduler.add_job(
                    self.analyze_and_trade,
                    'interval',
                    seconds=10,  # Анализ каждые 10 секунд для агрессивного скальпинга
                    id='trading_job'
                )
                if not self.scheduler.running:
                    self.scheduler.start()
                message = f"✅ Бот запущен в режиме {self.mode.value}"
            else:
                self.status = BotStatus.STOPPED
                if self.scheduler.get_job('trading_job'):
                    self.scheduler.remove_job('trading_job')
                message = "⏸️ Бот остановлен"

        elif query.data == "balance":
            balance = self.current_trader.get_balance()
            if balance:
                total_balance = balance.get('totalAvailableBalance', 0)
                message = f"💰 Баланс ({self.mode.value}):\n${float(total_balance):.2f}"
            else:
                message = "❌ Не удалось получить баланс"

        elif query.data == "status":
            position = self.current_trader.get_position(self.symbol)
            if position and position.size > 0:
                pnl_emoji = "📈" if position.unrealized_pnl > 0 else "📉"
                message = (
                    f"📊 Статус:\n"
                    f"Режим: {self.mode.value}\n"
                    f"Статус бота: {self.status.value}\n"
                    f"Символ: {self.symbol}\n"
                    f"Позиция: {position.side} {position.size}\n"
                    f"Цена входа: {position.entry_price:.2f}\n"
                    f"{pnl_emoji} PnL: ${position.unrealized_pnl:.2f}"
                )
            else:
                message = (
                    f"📊 Статус:\n"
                    f"Режим: {self.mode.value}\n"
                    f"Статус бота: {self.status.value}\n"
                    f"Символ: {self.symbol}\n"
                    f"Позиция: Закрыта"
                )

        elif query.data == "stats":
            stats = self.db.get_daily_stats(self.mode.value)
            stats = {**{"total_trades":0, "winning_trades":0, "losing_trades":0, "total_pnl":0.0, "avg_pnl":0.0}, **(stats or {})}
            win_rate = 0
            if stats['total_trades'] > 0:
                win_rate = (stats['winning_trades'] / stats['total_trades']) * 100

            message = (
                f"📈 Статистика за сегодня ({self.mode.value}):\n"
                f"Всего сделок: {stats['total_trades']}\n"
                f"Прибыльных: {stats['winning_trades']}\n"
                f"Убыточных: {stats['losing_trades']}\n"
                f"Винрейт: {win_rate:.1f}%\n"
                f"Общий PnL: ${stats['total_pnl']:.2f}\n"
                f"Средний PnL: ${stats['avg_pnl']:.2f}"
            )

        elif query.data == "toggle_mode":
            self.mode = TradingMode.MAINNET if self.mode == TradingMode.TESTNET else TradingMode.TESTNET
            message = f"🔄 Режим изменен на: {self.mode.value}"

        await query.edit_message_text(message, reply_markup=self.get_keyboard())

    def run_telegram_bot(self):
        """Запуск Telegram бота (синхронно, без собственного asyncio.run)"""
        application = Application.builder().token(self.telegram_token).build()

        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CallbackQueryHandler(self.button_callback))

        # Важно: run_polling сам управляет циклом событий. Не оборачивать в asyncio.run!
        application.run_polling()


def main():
    """Главная функция"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('scalping_bot.log'),
            logging.StreamHandler()
        ]
    )

    print("🤖 Инициализация агрессивного скальпинг-бота...")
    print("⚠️  ВНИМАНИЕ: Данный бот крайне рискован!")
    print("💀 Можете потерять весь депозит за считанные минуты!")
    print("🧪 ОБЯЗАТЕЛЬНО ТЕСТИРУЙТЕ НА TESTNET!")

    bot = ScalpingBot()

    # Запуск бота
    try:
        bot.run_telegram_bot()
    except KeyboardInterrupt:
        print("👋 Бот остановлен пользователем")
    except Exception as e:
        logging.error(f"Критическая ошибка: {e}")


if __name__ == "__main__":
    main()
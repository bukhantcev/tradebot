# -*- coding: utf-8 -*-
"""
Adaptive multi-strategy backtester for Bybit klines.

What this script does
---------------------
• Analyzes market *trendiness*, *volatility* and *EMA slope* on a higher timeframe to detect a regime:
    - TREND_UP / TREND_DOWN
    - RANGE
    - BREAKOUT (post-squeeze expansion)
    - SCALP (low-trend, noisy/volatile conditions suited to quick mean-revert trades)

• Automatically CHOOSES a strategy per regime:
    - trend_ma_pullback  → trades pullbacks in trend (EMA10/30 + RSI filter)
    - range_bb           → counter-trend at Bollinger bands, TP at basis
    - breakout_donchian  → classic Donchian breakout
    - scalp_meanrevert   → short swings around EMA on a lower TF with tight ATR stops

• Dynamically SWITCHES timeframe used for signal generation, per regime, for example:
    RANGE → use a *higher* TF (e.g., 1h base → 4h) to avoid chop,
    SCALP → use a *lower* TF (e.g., 15m) to catch micro swings.
  The mapping is configurable via ENV (see REGIME_TF_* variables below).

• Risk management is ATR-based with per-strategy multipliers + position sizing by risk %.

Outputs
-------
- Console summary and logs
- CSV with individual trades at ./best_family_trades.csv

Environment (.env) variables
----------------------------
# Data / run
SYMBOL=BTCUSDT
INTERVAL_MIN=60          # base timeframe in minutes (used as scheduling spine)
DAYS=45                  # how many recent days to fetch
CATEGORY=linear          # Bybit v5 category: linear|inverse|spot
DATA_CSV=                # optional path to CSV with columns ts, open, high, low, close, volume

# Fees & risk
START_BALANCE=1000
RISK_PCT=0.05
COMMISSION_PCT=0.0
SLIPPAGE_USDT=0.0

# Multi-TF fetch (used when ADAPTIVE=1)
ADAPTIVE=1
MULTI_INTERVALS=15,60,240  # list of TFs to download (minutes). Should include INTERVAL_MIN.
# Which TF to use for signals in each regime. Allowed values: lower|base|higher
REGIME_TF_TREND=base
REGIME_TF_RANGE=higher
REGIME_TF_BREAKOUT=base
REGIME_TF_SCALP=lower

# Regime detection tuning (computed on the *higher* TF stream)
ADX_WINDOW=14
ADX_TREND=25
ADX_RANGE=18
REGIME_BB_WINDOW=20
REGIME_BB_STD=2.0
BB_SQUEEZE_PCT=0.06
BREAKOUT_BBW_EXPAND=1.35     # factor vs previous bar BBW to call expansion
SCALP_ADX_MAX=20             # max ADX for scalp regime
SCALP_BBW_MIN=0.07           # minimum BBW to ensure some noise/vol for scalping
EMA_SLOPE_MIN=0.0004         # min absolute EMA slope (pct per bar) to call trend

# Strategy params (shared/defaults)
STOP_MULT_ATR=2.0
TRAIL_MULT_ATR=1.0

# Strategy-specific (scalp)
SCALP_ENTRY_ATR=0.6          # entry threshold: distance from EMA10 in ATR units
SCALP_STOP_ATR=0.9
SCALP_TRAIL_ATR=0.6
SCALP_TP_ATR=0.8

# Strategy-specific (range BB)
RANGE_BB_WINDOW=20
RANGE_BB_STD=2.0

# Strategy-specific (breakout)
DONCHIAN=20

# Logging
LOG_LEVEL=INFO               # DEBUG|INFO|WARNING|ERROR
LOG_FILE=tradebot.log
LOG_JSON=0
LOG_BAR_DETAILS=0
LOG_TRADES_ONLY=0
"""

from __future__ import annotations
import os
from collections import Counter, defaultdict
import math
import csv
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional
import itertools
import json
import logging
from logging.handlers import RotatingFileHandler

import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv

# ----------------------------
# Logging setup
# ----------------------------

def setup_logger():
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_file = os.getenv("LOG_FILE", "tradebot.log")
    as_json = os.getenv("LOG_JSON", "0").strip() in ("1", "true", "y", "yes")
    logger = logging.getLogger("tradebot")
    logger.setLevel(level)
    logger.handlers.clear()

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            base = {
                "ts": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "level": record.levelname,
                "msg": record.getMessage(),
                "name": record.name,
            }
            if hasattr(record, "extra"):
                base.update(record.extra)  # type: ignore
            return json.dumps(base, ensure_ascii=False)

    if as_json:
        fmt = JsonFormatter()
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

        file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    else:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

        file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger

log = setup_logger()

# Flags to control verbosity
LOG_BAR_DETAILS = os.getenv("LOG_BAR_DETAILS", "0").strip() in ("1","true","y","yes")
LOG_TRADES_ONLY = os.getenv("LOG_TRADES_ONLY", "0").strip() in ("1","true","y","yes")

# ----------------------------
# Utilities
# ----------------------------

def to_ms(ts: dt.datetime) -> int:
    return int(ts.timestamp() * 1000)

def now_ms() -> int:
    return to_ms(dt.datetime.utcnow())

# ----------------------------
# Data loading
# ----------------------------

BYBIT_V5_KLINE = "https://api.bybit.com/v5/market/kline"

def fetch_bybit_klines(symbol: str, interval_min: int, start_ms: int, end_ms: int, category: str = "linear") -> pd.DataFrame:
    frames = []
    params = {
        "category": category,
        "symbol": symbol,
        "interval": str(interval_min),
        "start": str(start_ms),
        "end": str(end_ms),
        "limit": "1000",
    }
    log.info(f"[DATA] Fetching Bybit klines {symbol} {interval_min}m {category} …")
    while True:
        try:
            resp = requests.get(BYBIT_V5_KLINE, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.error(f"[DATA] HTTP error: {e}")
            raise
        if data.get("retCode") != 0:
            log.error(f"[DATA] Bybit API error: {data}")
            raise RuntimeError(f"Bybit API error: {data}")
        result = data.get("result", {})
        list_rows = result.get("list", [])
        if not list_rows:
            break
        rows = list(reversed(list_rows))
        frames.append(pd.DataFrame(rows, columns=[
            "start", "open", "high", "low", "close", "volume", "turnover"
        ]))
        last_end = int(rows[-1][0])
        next_start = last_end + interval_min * 60 * 1000
        if next_start >= end_ms:
            break
        params["start"] = str(next_start)
    if not frames:
        raise RuntimeError("No kline data returned from Bybit")
    df = pd.concat(frames, ignore_index=True)
    for c in ["start", "open", "high", "low", "close", "volume", "turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.rename(columns={"start": "ts"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    log.info(f"[DATA] Loaded {len(df)} bars for {symbol} {interval_min}m")
    return df[["ts", "datetime", "open", "high", "low", "close", "volume", "turnover"]]

def load_from_csv(path: str) -> pd.DataFrame:
    log.info(f"[DATA] Loading CSV: {path}")
    df = pd.read_csv(path)
    if "ts" not in df.columns:
        for c in df.columns:
            if str(df[c].dtype).startswith("int"):
                df = df.rename(columns={c: "ts"})
                break
    if "datetime" not in df.columns and "ts" in df.columns:
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True, errors="coerce")
    keep = [c for c in ["ts", "datetime", "open", "high", "low", "close", "volume"] if c in df.columns]
    log.info(f"[DATA] CSV rows: {len(df)}")
    return df[keep].copy()

# ----------------------------
# Indicators
# ----------------------------

def bollinger_bands(close: pd.Series, window: int = 10, num_std: float = 1.5):
    ma = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = ma + num_std * std
    lower = ma - num_std * std
    return ma, upper, lower

def atr(df: pd.DataFrame, window: int = 10) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    plus_dm = (high.diff()).where((high.diff() > low.diff()), 0.0).clip(lower=0)
    minus_dm = (low.diff().abs()).where((low.diff() > high.diff()), 0.0).clip(lower=0)
    tr_components = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1)
    tr = tr_components.max(axis=1)
    atr_ = tr.rolling(window=window, min_periods=window).mean()
    pdi = 100 * (plus_dm.rolling(window=window, min_periods=window).mean() / atr_).replace([np.inf, -np.inf], np.nan)
    mdi = 100 * (minus_dm.rolling(window=window, min_periods=window).mean() / atr_).replace([np.inf, -np.inf], np.nan)
    dx = (abs(pdi - mdi) / (pdi + mdi)).replace([np.inf, -np.inf], np.nan) * 100
    adx_ = dx.rolling(window=window, min_periods=window).mean()
    return adx_

def bollinger_bandwidth(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    ma = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = ma + num_std * std
    lower = ma - num_std * std
    return (upper - lower) / ma

# --- Regime detection ---
class Regime:
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"
    BREAKOUT = "breakout"

def detect_regime(df_higher: pd.DataFrame,
                  adx_window: int = 14,
                  adx_trend: float = 25.0,
                  adx_range: float = 18.0,
                  bb_window: int = 20,
                  bb_std: float = 2.0,
                  bb_squeeze_pct: float = 0.06) -> str:
    df = df_higher.copy()
    df["ADX"] = adx(df, window=adx_window)
    df["EMA_fast"] = ema(df["close"], 10)
    df["EMA_slow"] = ema(df["close"], 30)
    bbw = bollinger_bandwidth(df["close"], window=bb_window, num_std=bb_std)
    df["BBW"] = bbw
    roll = 40
    df["BBW_pct20"] = df["BBW"].rolling(roll, min_periods=min(roll, len(df))).quantile(0.2)
    df["BBW_min40"] = df["BBW"].rolling(roll, min_periods=min(roll, len(df))).min()

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    was_squeeze = prev["BBW"] <= max(prev["BBW_pct20"], bb_squeeze_pct)
    now_expanded = (last["BBW"] >= prev["BBW"] * 1.35) or (last["BBW"] >= max(last["BBW_pct20"], bb_squeeze_pct) * 1.5)

    if was_squeeze and now_expanded and last["ADX"] >= max(adx_range + 2, adx_window * 0):
        return Regime.BREAKOUT
    if last["ADX"] >= adx_trend:
        return Regime.TREND_UP if last["EMA_fast"] >= last["EMA_slow"] else Regime.TREND_DOWN
    if (last["ADX"] <= adx_range) and (last["BBW"] <= max(last["BBW_pct20"], bb_squeeze_pct)):
        return Regime.RANGE
    return Regime.TREND_UP if last["EMA_fast"] >= last["EMA_slow"] else Regime.TREND_DOWN

# --- Strategy signals ---
@dataclass
class Signal:
    action: str  # 'buy', 'sell', 'exit', or 'none'
    price: Optional[float] = None
    reason: str = ""

def signal_trend_ma_pullback(df: pd.DataFrame) -> Signal:
    s = df.copy()
    s["EMA_fast"] = ema(s["close"], 10)
    s["EMA_slow"] = ema(s["close"], 30)
    delta = s["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14, min_periods=14).mean()
    roll_down = down.rolling(14, min_periods=14).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    s["RSI"] = rsi

    row = s.iloc[-1]; prev = s.iloc[-2]
    if row["EMA_fast"] > row["EMA_slow"] and prev["EMA_fast"] > prev["EMA_slow"] and row["RSI"] <= 40:
        return Signal("buy", price=row["close"], reason="trend_up_pullback")
    if row["EMA_fast"] < row["EMA_slow"] and prev["EMA_fast"] < prev["EMA_slow"] and row["RSI"] >= 60:
        return Signal("sell", price=row["close"], reason="trend_down_pullback")
    return Signal("none")

def signal_range_bb(df: pd.DataFrame, bb_window: int = 20, bb_std: float = 2.0) -> Signal:
    s = df.copy()
    ma, upper, lower = bollinger_bands(s["close"], window=bb_window, num_std=bb_std)
    s["bb_ma"] = ma; s["bb_upper"] = upper; s["bb_lower"] = lower
    row = s.iloc[-1]
    if row["close"] <= row["bb_lower"]:
        return Signal("buy", price=row["close"], reason="bb_lower_revert")
    if row["close"] >= row["bb_upper"]:
        return Signal("sell", price=row["close"], reason="bb_upper_revert")
    return Signal("none")

def signal_breakout_donchian(df: pd.DataFrame, ch: int = 20) -> Signal:
    s = df.copy()
    highest = s["high"].rolling(ch, min_periods=ch).max()
    lowest = s["low"].rolling(ch, min_periods=ch).min()
    row = s.iloc[-1]
    hi = highest.iloc[-2]; lo = lowest.iloc[-2]
    if row["close"] > hi:
        return Signal("buy", price=row["close"], reason="donchian_breakout_up")
    if row["close"] < lo:
        return Signal("sell", price=row["close"], reason="donchian_breakout_down")
    return Signal("none")

# ----------------------------
# Backtest core
# ----------------------------

def calc_position_size(equity: float, risk_pct: float, stop_distance: float) -> float:
    if stop_distance <= 0:
        return 0.0
    risk_amt = equity * risk_pct
    return max(risk_amt / stop_distance, 0.0)

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    entry: float
    exit: float
    size: float
    pnl_usdt: float
    equity_after: float
    strategy: str = ""
    regime: str = ""
    tf: str = ""

def backtest(df: pd.DataFrame,
             start_balance: float = 1000.0,
             risk_pct: float = 0.05,
             commission_pct: float = 0.0,
             slippage_usdt: float = 0.0,
             bb_window: int = 10,
             bb_std: float = 1.5,
             atr_window: int = 10,
             stop_mult: float = 2.0,
             trail_mult: float = 1.5) -> (float, List[Trade], float, float, int):

    df = df.copy().reset_index(drop=True)
    df["close"] = df["close"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)

    ma, upper, lower = bollinger_bands(df["close"], window=bb_window, num_std=bb_std)
    atr_series = atr(df, window=atr_window)
    df["bb_ma"] = ma; df["bb_upper"] = upper; df["bb_lower"] = lower; df["atr"] = atr_series

    equity = start_balance
    peak_equity = start_balance
    max_dd = 0.0
    position = None
    entry_index = None
    trades: List[Trade] = []

    def apply_fees(price: float) -> float:
        return price * (1 + commission_pct)
    def apply_fees_sell(price: float) -> float:
        return price * (1 - commission_pct)

    log.info(f"[BT] Start single-strategy BB backtest | start_balance={start_balance} risk_pct={risk_pct} "
             f"bb_window={bb_window} bb_std={bb_std} atr_window={atr_window} stop_mult={stop_mult} trail_mult={trail_mult}")

    for i in range(1, len(df) - 1):
        row = df.iloc[i]; next_row = df.iloc[i + 1]

        # manage trailing
        if position is not None:
            atr_now = row["atr"]
            if not math.isnan(atr_now):
                if position["side"] == "long":
                    new_trail = next_row["open"] - trail_mult * atr_now
                    position["trail"] = max(position["trail"], new_trail)
                    stop_level = max(position["stop"], position["trail"])
                    if next_row["open"] <= stop_level:
                        exit_price = max(next_row["open"] - slippage_usdt, stop_level - slippage_usdt)
                        exit_price = apply_fees_sell(exit_price)
                        pnl = (exit_price - position["entry"]) * position["size"]
                        equity += pnl
                        trades.append(Trade(
                            entry_time=df.iloc[entry_index]["datetime"], exit_time=next_row["datetime"],
                            side="long", entry=position["entry"], exit=exit_price, size=position["size"],
                            pnl_usdt=pnl, equity_after=equity, strategy="bb_countertrend", regime="single",
                            tf=str(int(os.getenv("INTERVAL_MIN", "60"))),
                        ))
                        log.info(f"[EXIT] long STOP {next_row['datetime']} "
                                 f"entry={position['entry']:.2f} exit={exit_price:.2f} pnl={pnl:.2f} eq={equality_fmt(equity)}")
                        position = None; entry_index = None
                else:
                    new_trail = next_row["open"] + trail_mult * atr_now
                    position["trail"] = min(position["trail"], new_trail)
                    stop_level = min(position["stop"], position["trail"])
                    if next_row["open"] >= stop_level:
                        exit_price = min(next_row["open"] + slippage_usdt, stop_level + slippage_usdt)
                        exit_price = apply_fees(exit_price)
                        pnl = (position["entry"] - exit_price) * position["size"]
                        equity += pnl
                        trades.append(Trade(
                            entry_time=df.iloc[entry_index]["datetime"], exit_time=next_row["datetime"],
                            side="short", entry=position["entry"], exit=exit_price, size=position["size"],
                            pnl_usdt=pnl, equity_after=equity, strategy="bb_countertrend", regime="single",
                            tf=str(int(os.getenv("INTERVAL_MIN", "60"))),
                        ))
                        log.info(f"[EXIT] short STOP {next_row['datetime']} "
                                 f"entry={position['entry']:.2f} exit={exit_price:.2f} pnl={pnl:.2f} eq={equality_fmt(equity)}")
                        position = None; entry_index = None

        # entries / TP management
        if position is None:
            if not (math.isnan(row["bb_lower"]) or math.isnan(row["atr"])):
                if row["close"] <= row["bb_lower"]:
                    stop_dist = stop_mult * row["atr"]
                    if stop_dist > 0:
                        size = calc_position_size(equity, risk_pct, stop_dist)
                        entry_price = apply_fees(next_row["open"] + slippage_usdt)
                        position = {
                            "side": "long", "entry": entry_price, "size": size,
                            "stop": entry_price - stop_dist, "trail": entry_price - stop_dist, "tp": row["bb_ma"],
                        }
                        entry_index = i + 1
                        log_trade_open("long", row, next_row, entry_price, size, stop_dist, "bb_lower_touch")
                if position is None and row["close"] >= row["bb_upper"] and not math.isnan(row["atr"]):
                    stop_dist = stop_mult * row["atr"]
                    if stop_dist > 0:
                        size = calc_position_size(equity, risk_pct, stop_dist)
                        entry_price = apply_fees_sell(next_row["open"] - slippage_usdt)
                        position = {
                            "side": "short", "entry": entry_price, "size": size,
                            "stop": entry_price + stop_dist, "trail": entry_price + stop_dist, "tp": row["bb_ma"],
                        }
                        entry_index = i + 1
                        log_trade_open("short", row, next_row, entry_price, size, stop_dist, "bb_upper_touch")
        else:
            if not math.isnan(row["bb_ma"]):
                if position["side"] == "long" and next_row["open"] >= row["bb_ma"]:
                    exit_price = max(row["bb_ma"], next_row["open"]) - slippage_usdt
                    exit_price = apply_fees_sell(exit_price)
                    pnl = (exit_price - position["entry"]) * position["size"]
                    equity += pnl
                    trades.append(Trade(
                        entry_time=df.iloc[entry_index]["datetime"], exit_time=next_row["datetime"],
                        side="long", entry=position["entry"], exit=exit_price, size=position["size"],
                        pnl_usdt=pnl, equity_after=equity, strategy="bb_countertrend", regime="single",
                        tf=str(int(os.getenv("INTERVAL_MIN", "60"))),
                    ))
                    log.info(f"[EXIT] long TP {next_row['datetime']} "
                             f"entry={position['entry']:.2f} exit={exit_price:.2f} pnl={pnl:.2f} eq={equality_fmt(equity)}")
                    position = None; entry_index = None
                elif position["side"] == "short" and next_row["open"] <= row["bb_ma"]:
                    exit_price = min(row["bb_ma"], next_row["open"]) + slippage_usdt
                    exit_price = apply_fees(exit_price)
                    pnl = (position["entry"] - exit_price) * position["size"]
                    equity += pnl
                    trades.append(Trade(
                        entry_time=df.iloc[entry_index]["datetime"], exit_time=next_row["datetime"],
                        side="short", entry=position["entry"], exit=exit_price, size=position["size"],
                        pnl_usdt=pnl, equity_after=equity, strategy="bb_countertrend", regime="single",
                        tf=str(int(os.getenv("INTERVAL_MIN", "60"))),
                    ))
                    log.info(f"[EXIT] short TP {next_row['datetime']} "
                             f"entry={position['entry']:.2f} exit={exit_price:.2f} pnl={pnl:.2f} eq={equality_fmt(equity)}")
                    position = None; entry_index = None

        peak_equity = max(peak_equity, equity)
        dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0.0
        max_dd = min(max_dd, dd)

        if LOG_BAR_DETAILS and not LOG_TRADES_ONLY:
            log.debug(f"[BAR] {row['datetime']} close={row['close']:.2f} bb_ma={row.get('bb_ma', np.nan):.2f} "
                      f"atr={row.get('atr', np.nan):.2f} eq={equality_fmt(equity)} pos={'None' if position is None else position['side']}")

    n_trades = len(trades)
    wins = sum(1 for t in trades if t.pnl_usdt > 0)
    win_rate = (wins / n_trades * 100.0) if n_trades else 0.0
    max_dd_pct = abs(max_dd) * 100.0
    log.info(f"[BT] Done | trades={n_trades} win%={win_rate:.2f} maxDD%={max_dd_pct:.2f} final={equality_fmt(equity)}")
    return equity, trades, max_dd_pct, win_rate, n_trades

def equality_fmt(x: float) -> str:
    return f"{x:.2f} USDT"

def log_trade_open(side: str, row, next_row, entry_price, size, stop_dist, reason: str):
    if LOG_TRADES_ONLY or LOG_BAR_DETAILS:
        log.info(f"[ENTRY] {side} {next_row['datetime']} reason={reason} "
                 f"entry={entry_price:.2f} size={size:.6f} stop_dist={stop_dist:.2f}")

# ----------------------------
# Optimization (grid search)
# ----------------------------

def parse_env_list(name: str, default_list, cast=float):
    s = os.getenv(name, "").strip()
    if not s:
        return list(default_list)
    out = []
    for tok in s.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(cast(tok))
    return out

def optimize_params(
    df: pd.DataFrame,
    start_balance: float,
    commission_pct: float,
    slippage_usdt: float,
    bb_windows: List[int],
    bb_stds: List[float],
    atr_windows: List[int],
    stop_mults: List[float],
    trail_mults: List[float],
    risk_pcts: List[float],
    top_n: int = 10,
):
    results = []
    best = None
    log.info(f"[GRID] Start grid search | candidates={len(bb_windows)*len(bb_stds)*len(atr_windows)*len(stop_mults)*len(trail_mults)*len(risk_pcts)}")
    for bw, bs, aw, sm, tm, rp in itertools.product(
        bb_windows, bb_stds, atr_windows, stop_mults, trail_mults, risk_pcts
    ):
        equity, trades, max_dd_pct, win_rate, n_trades = backtest(
            df,
            start_balance=start_balance,
            risk_pct=rp,
            commission_pct=commission_pct,
            slippage_usdt=slippage_usdt,
            bb_window=bw,
            bb_std=bs,
            atr_window=aw,
            stop_mult=sm,
            trail_mult=tm,
        )
        profit = equity - start_balance
        if n_trades == 0:
            continue
        rec = {
            "final_equity": round(equity, 4),
            "profit_usdt": round(profit, 4),
            "win_rate_pct": round(win_rate, 4),
            "max_dd_pct": round(max_dd_pct, 4),
            "n_trades": int(n_trades),
            "bb_window": int(bw),
            "bb_std": float(bs),
            "atr_window": int(aw),
            "stop_mult": float(sm),
            "trail_mult": float(tm),
            "risk_pct": float(rp),
        }
        results.append(rec)
        if best is None or equity > best["final_equity"]:
            best = rec
    results.sort(key=lambda x: x["final_equity"], reverse=True)

    try:
        out_path = os.path.abspath("best_family_search.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else [
                "final_equity","profit_usdt","win_rate_pct","max_dd_pct","n_trades",
                "bb_window","bb_std","atr_window","stop_mult","trail_mult","risk_pct"
            ])
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        log.info(f"[GRID] Saved results: {out_path}")
    except Exception as e:
        log.warning(f"[GRID] Failed to save grid csv: {e}")

    if best:
        log.info(f"[GRID] Best: final={best['final_equity']} win%={best['win_rate_pct']} dd%={best['max_dd_pct']} "
                 f"params=(bb={best['bb_window']},{best['bb_std']} atr={best['atr_window']} stop={best['stop_mult']} trail={best['trail_mult']} risk={best['risk_pct']})")
        best_params = dict(
            bb_window=best["bb_window"],
            bb_std=best["bb_std"],
            atr_window=best["atr_window"],
            stop_mult=best["stop_mult"],
            trail_mult=best["trail_mult"],
            risk_pct=best["risk_pct"],
        )
        return best_params, results[:top_n]
    else:
        log.warning("[GRID] No valid grid results")
        return None, results[:top_n]

# ----------------------------
# Adaptive multi-strategy backtest
# ----------------------------

@dataclass
class AdaptiveParams:
    adx_window: int = 14
    adx_trend: float = 25.0
    adx_range: float = 18.0
    bb_window: int = 20
    bb_std: float = 2.0
    bb_squeeze_pct: float = 0.06
    risk_pct: float = 0.05
    stop_mult_atr: float = 2.0
    trail_mult_atr: float = 1.0

def ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in ["open","high","low","close"]:
        d[c] = d[c].astype(float)
    return d

def adaptive_backtest(df_map: dict,
                      base_tf: str,
                      higher_tf: str,
                      start_balance: float = 1000.0,
                      commission_pct: float = 0.0,
                      slippage_usdt: float = 0.0,
                      params: AdaptiveParams = AdaptiveParams()) -> (float, List[Trade], dict):
    frames = {k: ensure_types(v.sort_values("ts").reset_index(drop=True)) for k, v in df_map.items()}
    if base_tf not in frames or higher_tf not in frames:
        raise ValueError("Missing required timeframes in df_map")

    df_base = frames[base_tf]
    df_hi = frames[higher_tf]
    df_15 = frames["15"] if "15" in frames else df_base

    equity = start_balance
    trades: List[Trade] = []
    position = None
    peak_equity = start_balance
    max_dd = 0.0
    regime_counts = Counter()

    def apply_fees_buy(px): return px * (1 + commission_pct)
    def apply_fees_sell(px): return px * (1 - commission_pct)

    def pick_df(reg: str) -> pd.DataFrame:
        if reg in (Regime.RANGE, Regime.BREAKOUT):
            return df_15
        return df_base

    atr_cache = {k: atr(v, window=14) for k, v in frames.items()}
    j_hi = 0

    log.info(f"[ADAPT] Start | base_tf={base_tf} higher_tf={higher_tf} risk={params.risk_pct} "
             f"stopATRx={params.stop_mult_atr} trailATRx={params.trail_mult_atr}")

    for i in range(1, len(df_base) - 1):
        t_now = df_base.iloc[i]["ts"]
        while j_hi + 1 < len(df_hi) and df_hi.iloc[j_hi + 1]["ts"] <= t_now:
            j_hi += 1
        df_hi_slice = df_hi.iloc[max(0, j_hi - 400): j_hi + 1]
        if len(df_hi_slice) < 30:
            continue
        regime = detect_regime(df_hi_slice,
                               adx_window=params.adx_window,
                               adx_trend=params.adx_trend,
                               adx_range=params.adx_range,
                               bb_window=params.bb_window,
                               bb_std=params.bb_std,
                               bb_squeeze_pct=params.bb_squeeze_pct)
        regime_counts[regime] += 1
        df_act = pick_df(regime)
        idx = df_act.index[df_act["ts"] == t_now]
        if len(idx) == 0:
            continue
        ai = int(idx[0])
        if ai >= len(df_act) - 1:
            break
        row = df_act.iloc[ai]; next_row = df_act.iloc[ai + 1]

        if LOG_BAR_DETAILS and not LOG_TRADES_ONLY:
            log.debug(f"[ADAPT][BAR] t={row['datetime']} regime={regime} close={row['close']:.2f} eq={equality_fmt(equity)} "
                      f"pos={'None' if position is None else position['side']}")

        if position is not None:
            tf_key = position["tf"]
            atr_now = atr_cache[tf_key].iloc[ai] if ai < len(atr_cache[tf_key]) else np.nan
            if not math.isnan(atr_now):
                if position["side"] == "long":
                    new_trail = next_row["open"] - params.trail_mult_atr * atr_now
                    position["trail"] = max(position["trail"], new_trail)
                    stop_level = max(position["stop"], position["trail"])
                    if next_row["open"] <= stop_level:
                        exit_price = max(next_row["open"], stop_level) - slippage_usdt
                        exit_price = apply_fees_sell(exit_price)
                        pnl = (exit_price - position["entry"]) * position["size"]
                        equity += pnl
                        trades.append(Trade(
                            entry_time=position["entry_time"], exit_time=next_row["datetime"], side="long",
                            entry=position["entry"], exit=exit_price, size=position["size"], pnl_usdt=pnl,
                            equity_after=equity, strategy=position["strategy"], regime=position["regime"], tf=position["tf"],
                        ))
                        log.info(f"[ADAPT][EXIT] long STOP t={next_row['datetime']} "
                                 f"regime={position['regime']} strat={position['strategy']} tf={position['tf']} "
                                 f"entry={position['entry']:.2f} exit={exit_price:.2f} pnl={pnl:.2f} eq={equality_fmt(equity)}")
                        position = None
                else:
                    new_trail = next_row["open"] + params.trail_mult_atr * atr_now
                    position["trail"] = min(position["trail"], new_trail)
                    stop_level = min(position["stop"], position["trail"])
                    if next_row["open"] >= stop_level:
                        exit_price = min(next_row["open"], stop_level) + slippage_usdt
                        exit_price = apply_fees_buy(exit_price)
                        pnl = (position["entry"] - exit_price) * position["size"]
                        equity += pnl
                        trades.append(Trade(
                            entry_time=position["entry_time"], exit_time=next_row["datetime"], side="short",
                            entry=position["entry"], exit=exit_price, size=position["size"], pnl_usdt=pnl,
                            equity_after=equity, strategy=position["strategy"], regime=position["regime"], tf=position["tf"],
                        ))
                        log.info(f"[ADAPT][EXIT] short STOP t={next_row['datetime']} "
                                 f"regime={position['regime']} strat={position['strategy']} tf={position['tf']} "
                                 f"entry={position['entry']:.2f} exit={exit_price:.2f} pnl={pnl:.2f} eq={equality_fmt(equity)}")
                        position = None

        if position is None:
            window = 60
            if ai < window:
                continue
            slice_df = df_act.iloc[ai - window: ai + 1]

            sig = Signal("none")
            if regime in (Regime.TREND_UP, Regime.TREND_DOWN):
                sig = signal_trend_ma_pullback(slice_df)
            elif regime == Regime.RANGE:
                sig = signal_range_bb(slice_df)
            elif regime == Regime.BREAKOUT:
                sig = signal_breakout_donchian(slice_df)

            if sig.action in ("buy", "sell"):
                atr_key = (base_tf if regime.startswith("trend") else ("15" if "15" in frames else base_tf))
                atr_now = atr_cache[atr_key].iloc[ai]
                if math.isnan(atr_now) or atr_now <= 0:
                    continue
                entry_price = (next_row["open"] + slippage_usdt) if sig.action == "buy" else (next_row["open"] - slippage_usdt)
                entry_price = (entry_price * (1 + commission_pct)) if sig.action == "buy" else (entry_price * (1 - commission_pct))
                side = "long" if sig.action == "buy" else "short"
                stop_distance = params.stop_mult_atr * atr_now
                size = calc_position_size(equity, params.risk_pct, stop_distance)
                if size <= 0:
                    continue
                strategy_name = "trend_ma_pullback" if regime in (Regime.TREND_UP, Regime.TREND_DOWN) else ("range_bb" if regime == Regime.RANGE else "breakout_donchian")
                tf_val = "15" if regime in (Regime.RANGE, Regime.BREAKOUT) and "15" in frames else base_tf
                position = {
                    "side": side, "entry": entry_price, "size": size,
                    "stop": entry_price - stop_distance if side == "long" else entry_price + stop_distance,
                    "trail": entry_price - stop_distance if side == "long" else entry_price + stop_distance,
                    "tf": tf_val, "entry_time": next_row["datetime"], "regime": regime, "strategy": strategy_name,
                }
                log.info(f"[ADAPT][ENTRY] {side} t={next_row['datetime']} regime={regime} strat={strategy_name} tf={tf_val} "
                         f"reason={sig.reason} entry={entry_price:.2f} size={size:.6f} stop_dist={stop_distance:.2f} eq={equality_fmt(equity)}")

        peak_equity = max(peak_equity, equity)
        dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0.0
        max_dd = min(max_dd, dd)

    metrics = {
        "max_dd_pct": abs(max_dd) * 100.0,
        "n_trades": len(trades),
        "final_equity": equity,
        "regime_counts": dict(regime_counts),
    }
    log.info(f"[ADAPT] Done | trades={metrics['n_trades']} maxDD%={metrics['max_dd_pct']:.2f} final={equality_fmt(equity)} "
             f"| regimes={metrics['regime_counts']}")
    return equity, trades, metrics

# ----------------------------
# Summaries
# ----------------------------

def summarize_by_key(trades: List[Trade], key: str) -> List[dict]:
    groups = defaultdict(list)
    for t in trades:
        k = getattr(t, key, "")
        groups[k].append(t)
    rows = []
    for k, arr in groups.items():
        n = len(arr)
        wins = sum(1 for x in arr if x.pnl_usdt > 0)
        total = sum(x.pnl_usdt for x in arr)
        rows.append({
            key: k, "n_trades": n, "wins": wins,
            "win_rate_pct": round((wins / n * 100.0) if n else 0.0, 2),
            "total_pnl": round(total, 2), "avg_pnl": round((total / n) if n else 0.0, 2),
        })
    rows.sort(key=lambda r: r["total_pnl"], reverse=True)
    return rows

def print_summary(title, rows, key):
    print(f"\n-- {title} --")
    if not rows:
        print("(no trades)")
        return
    print(f"{key:>16} | {'n':>3} | {'wins':>4} | {'win%':>6} | {'total':>12} | {'avg':>9}")
    print("-" * 64)
    for r in rows:
        print(f"{str(r[key]):>16} | {r['n_trades']:>3} | {r['wins']:>4} | {r['win_rate_pct']:>6.2f} | {r['total_pnl']:>12.2f} | {r['avg_pnl']:>9.2f}")

# ----------------------------
# Main
# ----------------------------

def main():
    load_dotenv()
    symbol = os.getenv("SYMBOL", "BTCUSDT")
    interval_min = int(os.getenv("INTERVAL_MIN", "60"))
    days = int(os.getenv("DAYS", "30"))
    start_balance = float(os.getenv("START_BALANCE", "1000"))
    risk_pct = float(os.getenv("RISK_PCT", "0.05"))
    commission_pct = float(os.getenv("COMMISSION_PCT", "0.0"))
    slippage_usdt = float(os.getenv("SLIPPAGE_USDT", "0.0"))
    data_csv = os.getenv("DATA_CSV", "").strip()

    adaptive = os.getenv("ADAPTIVE", "1").strip().lower() in ("1","true","yes","y")
    multi_intervals = os.getenv("MULTI_INTERVALS", "15,60,240").strip()

    log.info(f"[SETUP] symbol={symbol} interval={interval_min}m days={days} start_balance={start_balance} "
             f"risk={risk_pct} fee%={commission_pct} slip={slippage_usdt} adaptive={adaptive} tfs={multi_intervals}")

    if data_csv and os.path.exists(data_csv):
        print(f"Loading data from CSV: {data_csv}")
        base_df = load_from_csv(data_csv)
        dfs = {str(interval_min): base_df}
    else:
        end = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
        start = end - dt.timedelta(days=days + 1)
        if adaptive:
            tf_list = [int(x.strip()) for x in multi_intervals.split(',') if x.strip()]
        else:
            tf_list = [interval_min]
        dfs = {}
        for tf in sorted(set(tf_list)):
            print(f"Fetching Bybit klines for {symbol}, {tf}m, {days} days …")
            dfs[str(tf)] = fetch_bybit_klines(symbol, tf, to_ms(start), to_ms(end))
        for k in list(dfs.keys()):
            log.info(f"[DATA] TF {k}m bars: {len(dfs[k])}")

    for k in list(dfs.keys()):
        dfs[k] = dfs[k].sort_values("ts").reset_index(drop=True)
        for c in ["open","high","low","close","volume"]:
            if c not in dfs[k].columns:
                raise ValueError(f"Column '{c}' missing in data frame for TF {k}")

    if adaptive and len(dfs) >= 2:
        base_tf = str(interval_min if str(interval_min) in dfs else max(dfs.keys(), key=lambda x: int(x)))
        higher_tf = str(max([int(x) for x in dfs.keys()]))
        equity, trades, metrics = adaptive_backtest(
            dfs,
            base_tf=base_tf,
            higher_tf=higher_tf,
            start_balance=start_balance,
            commission_pct=commission_pct,
            slippage_usdt=slippage_usdt,
            params=AdaptiveParams(
                risk_pct=float(os.getenv("RISK_PCT", risk_pct)),
                stop_mult_atr=float(os.getenv("STOP_MULT_ATR", 2.0)),
                trail_mult_atr=float(os.getenv("TRAIL_MULT_ATR", 1.0)),
                adx_window=int(os.getenv("ADX_WINDOW", 14)),
                adx_trend=float(os.getenv("ADX_TREND", 25)),
                adx_range=float(os.getenv("ADX_RANGE", 18)),
                bb_window=int(os.getenv("REGIME_BB_WINDOW", 20)),
                bb_std=float(os.getenv("REGIME_BB_STD", 2.0)),
                bb_squeeze_pct=float(os.getenv("BB_SQUEEZE_PCT", 0.06)),
            ),
        )
        max_dd_pct = metrics["max_dd_pct"]
        win_rate = (sum(1 for t in trades if t.pnl_usdt > 0) / len(trades) * 100.0) if trades else 0.0
        n_trades = metrics["n_trades"]
        rc = metrics.get("regime_counts", {})
        if rc:
            print("Regime usage (bars counted on higher TF decisions):")
            for k in [Regime.TREND_UP, Regime.TREND_DOWN, Regime.RANGE, Regime.BREAKOUT]:
                if k in rc:
                    print(f"  {k:>12}: {rc[k]}")
        chosen_params_note = f"(ADAPTIVE on | TFs={sorted(list(dfs.keys()))} | base_tf={base_tf} | higher_tf={higher_tf})"
    else:
        optimize = os.getenv("OPTIMIZE", "1").strip().lower() in ("1", "true", "yes", "y")
        base_df = list(dfs.values())[0]
        if optimize:
            bb_windows = [int(x) for x in parse_env_list("BB_WINDOWS", [10, 14, 20], int)]
            bb_stds = parse_env_list("BB_STDS", [1.0, 1.5, 2.0], float)
            atr_windows = [int(x) for x in parse_env_list("ATR_WINDOWS", [10, 14, 20], int)]
            stop_mults = parse_env_list("STOP_MULTS", [1.5, 2.0, 2.5], float)
            trail_mults = parse_env_list("TRAIL_MULTS", [1.0, 1.5, 2.0], float)
            risk_pcts = parse_env_list("RISK_PCTS", [0.02, 0.03, 0.05], float)
            best_params, top_list = optimize_params(
                base_df, start_balance, commission_pct, slippage_usdt,
                bb_windows, bb_stds, atr_windows, stop_mults, trail_mults, risk_pcts,
            )
            if best_params is None:
                equity, trades, max_dd_pct, win_rate, n_trades = backtest(
                    base_df,
                    start_balance=start_balance,
                    risk_pct=risk_pct,
                    commission_pct=commission_pct,
                    slippage_usdt=slippage_usdt,
                )
                chosen_params_note = "(no valid grid results — used defaults)"
            else:
                equity, trades, max_dd_pct, win_rate, n_trades = backtest(
                    base_df,
                    start_balance=start_balance,
                    risk_pct=best_params["risk_pct"],
                    commission_pct=commission_pct,
                    slippage_usdt=slippage_usdt,
                    bb_window=best_params["bb_window"],
                    bb_std=best_params["bb_std"],
                    atr_window=best_params["atr_window"],
                    stop_mult=best_params["stop_mult"],
                    trail_mult=best_params["trail_mult"],
                )
                chosen_params_note = (
                    f"(bb_window={best_params['bb_window']}, bb_std={best_params['bb_std']}, "
                    f"atr_window={best_params['atr_window']}, stop_mult={best_params['stop_mult']}, "
                    f"trail_mult={best_params['trail_mult']}, risk_pct={best_params['risk_pct']})"
                )
        else:
            equity, trades, max_dd_pct, win_rate, n_trades = backtest(
                base_df,
                start_balance=start_balance,
                risk_pct=risk_pct,
                commission_pct=commission_pct,
                slippage_usdt=slippage_usdt,
            )
            chosen_params_note = "(defaults from ENV)"

    # Save trades CSV
    out_path = os.path.abspath("best_family_trades.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "entry_time", "exit_time", "side", "entry", "exit", "size", "pnl_usdt", "equity_after",
            "strategy", "regime", "tf"
        ])
        writer.writeheader()
        for t in trades:
            writer.writerow({
                "entry_time": t.entry_time, "exit_time": t.exit_time, "side": t.side,
                "entry": round(t.entry, 2), "exit": round(t.exit, 2), "size": round(t.size, 8),
                "pnl_usdt": round(t.pnl_usdt, 2), "equity_after": round(t.equity_after, 2),
                "strategy": getattr(t, "strategy", ""), "regime": getattr(t, "regime", ""), "tf": getattr(t, "tf", ""),
            })
    log.info(f"[IO] Trades CSV saved: {out_path}")

    profit = equity - start_balance
    profit_pct = profit / start_balance * 100.0

    by_strategy = summarize_by_key(trades, "strategy")
    by_regime = summarize_by_key(trades, "regime")
    by_tf = summarize_by_key(trades, "tf")

    def dump_log_table(name, rows, key):
        for r in rows:
            log.info(f"[SUM] {name} {r[key]} n={r['n_trades']} wins={r['wins']} win%={r['win_rate_pct']:.2f} "
                     f"total={r['total_pnl']:.2f} avg={r['avg_pnl']:.2f}")

    dump_log_table("strategy", by_strategy, "strategy")
    dump_log_table("regime", by_regime, "regime")
    dump_log_table("tf", by_tf, "tf")

    print("\n=== Backtest Summary ===")
    print_summary('P&L by strategy', by_strategy, 'strategy')
    print_summary('P&L by regime', by_regime, 'regime')
    print_summary('P&L by timeframe', by_tf, 'tf')

    if adaptive and len(dfs) >= 2:
        total_bars = ", ".join([f"{k}m:{len(v)}" for k, v in sorted(dfs.items(), key=lambda x: int(x[0]))])
        print(f"Symbol: {symbol} | Multi-TF Bars: {total_bars}")
    else:
        base_df = list(dfs.values())[0]
        print(f"Symbol: {symbol} | Interval: {list(dfs.keys())[0]}m | Bars: {len(base_df)}")
    print(f"Start balance: {start_balance:.2f} USDT")
    print(f"Final equity:  {equity:.2f} USDT {chosen_params_note}")
    print(f"Profit:        {profit:+.2f} USDT ({profit_pct:+.2f}%)")
    print(f"Trades:        {n_trades} | Win rate: {win_rate:.2f}% | Max DD: {max_dd_pct:.2f}%")
    print(f"Trades CSV:    {out_path}")
    if adaptive and len(dfs) >= 2:
        print("Adaptive multi-strategy run. ENV: ADAPTIVE=1, MULTI_INTERVALS=15,60,240")
    else:
        print("Single-strategy Bollinger mean-reversion run. Set ADAPTIVE=1 for regime switching.")

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
LIVE bot for Bybit TESTNET executing the same bar-based counter-trend Bollinger strategy
from the backtester, with ATR-based risk management and bar-open execution.

Key points:
- Uses Bybit v5 TESTNET REST API for order execution (category=linear)
- Stores and maintains kline history in a local SQLite database (sqlite3)
- Computes indicators from stored candles; acts strictly on bar boundaries
- Entries on next bar open after a signal; exits via TP (middle band) or bar-open stops
- Trailing stop updates each bar by ATR * TRAIL_MULT; enforcement at next bar open
- Position sizing: risk % of equity (wallet balance) / initial stop distance
- Logging mirrors the backtester (console + file, optional JSON); .env-driven

Environment (.env) variables (defaults in parentheses):
    # Market & timeframe
    SYMBOL=BTCUSDT
    INTERVAL_MIN=60                # 1,3,5,15,60,240,1440 ... — Bybit v5 minutes
    DAYS=30                        # how much history to keep/fetch for indicators
    CATEGORY=linear                # Bybit v5 category

    # Account / API (TESTNET)
    API_KEY=<your_key>
    API_SECRET=<your_secret>
    BASE_URL=https://api-testnet.bybit.com
    RECV_WINDOW=5000

    # Risk & fees
    START_BALANCE=1000             # used as fallback if wallet call fails
    RISK_PCT=0.05                  # 5% of equity per trade
    COMMISSION_PCT=0.0             # additional commission modeling (optional)
    SLIPPAGE_USDT=0.0              # slippage cushion per side (applied to entry/exit)

    # Strategy params (identical to backtester defaults)
    BB_WINDOW=10
    BB_STD=1.5
    ATR_WINDOW=10
    STOP_MULT=2.0
    TRAIL_MULT=1.5

    # Storage
    DB_PATH=candles.db

    # Looping
    POLL_SECONDS=5                 # check for new bars / progress every N seconds

    # Logging
    LOG_LEVEL=INFO                 # DEBUG|INFO|WARNING|ERROR
    LOG_FILE=tradebot.log
    LOG_JSON=0
    LOG_BAR_DETAILS=0
    LOG_TRADES_ONLY=0
"""

from __future__ import annotations
import os
import json
import hmac
import hashlib
import time
import math
import sqlite3
import logging
from logging.handlers import RotatingFileHandler
import datetime as dt
from dataclasses import dataclass
from typing import Optional, List, Dict
import threading
import urllib.parse

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Command input loop (CLI)
# ---------------------------------------------------------------------

def input_loop(client):
    while True:
        try:
            cmd = input().strip().lower()
            if cmd in ("bal", "balance", "eq"):
                try:
                    eq = client.wallet_balance("USDT")
                    print(f"[COMMAND] Balance: {eq:.2f} USDT")
                except Exception as e:
                    print(f"[COMMAND] Balance check failed: {e}")
            elif cmd in ("auth", "check", "diag"):
                try:
                    print("[COMMAND] Auth/Env diagnosis…")
                    symbol = os.getenv("SYMBOL", "BTCUSDT")
                    category = os.getenv("CATEGORY", "linear")
                    print(client.diagnose_auth(symbol, category))
                except Exception as e:
                    print(f"[COMMAND] Diagnose failed: {e}")
            elif cmd in ("pos", "positions"):
                try:
                    symbol = os.getenv("SYMBOL", "BTCUSDT")
                    category = os.getenv("CATEGORY", "linear")
                    ps = client.get_positions(category, symbol)
                    print(f"[COMMAND] Positions for {symbol}: {len(ps)}")
                    for p in ps:
                        print(json.dumps(p, ensure_ascii=False))
                except Exception as e:
                    print(f"[COMMAND] Positions failed: {e}")
        except EOFError:
            break

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def setup_logger():
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_file = os.getenv("LOG_FILE", "tradebot.log")
    as_json = os.getenv("LOG_JSON", "0").strip().lower() in ("1","true","y","yes")

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
        sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
        fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt); logger.addHandler(fh)
    else:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
        fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt); logger.addHandler(fh)
    return logger

log = setup_logger()
LOG_BAR_DETAILS = os.getenv("LOG_BAR_DETAILS", "0").strip().lower() in ("1","true","y","yes")
LOG_TRADES_ONLY = os.getenv("LOG_TRADES_ONLY", "0").strip().lower() in ("1","true","y","yes")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def now_ms() -> int:
    return int(dt.datetime.utcnow().timestamp() * 1000)

def to_ms(ts: dt.datetime) -> int:
    return int(ts.timestamp() * 1000)

def equality_fmt(x: float) -> str:
    return f"{x:.2f} USDT"

# ---------------------------------------------------------------------
# SQLite storage (candles)
# ---------------------------------------------------------------------

class CandleStore:
    def __init__(self, db_path: str = "candles.db"):
        self.db_path = db_path
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS candles (
                  symbol TEXT NOT NULL,
                  interval_min INTEGER NOT NULL,
                  ts INTEGER NOT NULL,
                  open REAL NOT NULL,
                  high REAL NOT NULL,
                  low REAL NOT NULL,
                  close REAL NOT NULL,
                  volume REAL,
                  turnover REAL,
                  PRIMARY KEY(symbol, interval_min, ts)
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_candles_ts ON candles(symbol, interval_min, ts)")

    def upsert_klines(self, symbol: str, interval_min: int, rows: List[Dict]):
        with self._conn() as con:
            con.executemany(
                """
                INSERT INTO candles(symbol, interval_min, ts, open, high, low, close, volume, turnover)
                VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(symbol, interval_min, ts) DO UPDATE SET
                  open=excluded.open,
                  high=excluded.high,
                  low=excluded.low,
                  close=excluded.close,
                  volume=excluded.volume,
                  turnover=excluded.turnover
                """,
                [
                    (
                        symbol,
                        interval_min,
                        int(r["ts"]),
                        float(r["open"]),
                        float(r["high"]),
                        float(r["low"]),
                        float(r["close"]),
                        float(r.get("volume", 0) or 0.0),
                        float(r.get("turnover", 0) or 0.0),
                    )
                    for r in rows
                ],
            )

    def load_frame(self, symbol: str, interval_min: int, since_ms: Optional[int] = None) -> pd.DataFrame:
        q = "SELECT ts, open, high, low, close, volume, turnover FROM candles WHERE symbol=? AND interval_min=?"
        args = [symbol, interval_min]
        if since_ms is not None:
            q += " AND ts >= ?"; args.append(int(since_ms))
        q += " ORDER BY ts ASC"
        with self._conn() as con:
            df = pd.read_sql_query(q, con, params=args)
        if len(df) == 0:
            return pd.DataFrame(columns=["ts","datetime","open","high","low","close","volume","turnover"])\
                     .astype({"ts":int,"open":float,"high":float,"low":float,"close":float,"volume":float,"turnover":float})
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df[["ts","datetime","open","high","low","close","volume","turnover"]]

# ---------------------------------------------------------------------
# Indicators (same as backtester)
# ---------------------------------------------------------------------

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
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()

# ---------------------------------------------------------------------
# Bybit v5 REST client (TESTNET)
# ---------------------------------------------------------------------

class BybitClient:
    def __init__(self, api_key: str, api_secret: str, base_url: str, recv_window: int = 5000):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.recv_window = str(int(recv_window))
        self.session = requests.Session()
        # approximate server time offset
        try:
            st = self._server_time_ms()
            self.time_offset_ms = st - int(time.time() * 1000)
        except Exception:
            self.time_offset_ms = 0

    # --- signing helpers ---
    def _sign(self, payload: str) -> str:
        return hmac.new(self.api_secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

    def _ts(self) -> str:
        return str(int(time.time() * 1000) + int(getattr(self, "time_offset_ms", 0)))

    def _headers(self, body_or_qs: Optional[str] = "") -> Dict[str, str]:
        ts = self._ts()
        payload = ts + self.api_key + self.recv_window + (body_or_qs or "")
        sign = self._sign(payload)
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": self.recv_window,
            "X-BAPI-SIGN": sign,
            "X-BAPI-SIGN-TYPE": "2",
            "Content-Type": "application/json",
        }

    def _query_string(self, params: Dict[str, str]) -> str:
        """Create a URL-encoded, *sorted* query string exactly as sent on the wire.
        This must match the request's actual query for Bybit v5 signature type 2.
        """
        if not params:
            return ""
        # sort by key then value, then urlencode with safe defaults
        items = sorted([(str(k), str(v)) for k, v in params.items() if v is not None])
        return urllib.parse.urlencode(items, doseq=True, quote_via=urllib.parse.quote)

    def _signed_get(self, path: str, params: Dict[str, str]) -> dict:
        # Build query string once and use it verbatim for signing and the request URL
        ts = self._ts()
        qs = self._query_string(params)
        payload = ts + self.api_key + self.recv_window + qs
        sign = self._sign(payload)
        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": self.recv_window,
            "X-BAPI-SIGN": sign,
            "X-BAPI-SIGN-TYPE": "2",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}{path}"
        if qs:
            url = f"{url}?{qs}"
        r = self.session.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        return r.json()

    def _server_time_ms(self) -> int:
        # v5 preferred
        try:
            r = self.session.get(f"{self.base_url}/v5/market/time", timeout=10)
            r.raise_for_status()
            data = r.json()
            if data.get("retCode") == 0:
                return int(data.get("result", {}).get("timeSecond", int(time.time()))) * 1000
        except Exception:
            pass
        # legacy fallback
        try:
            r = self.session.get(f"{self.base_url}/v3/public/time", timeout=10)
            r.raise_for_status()
            data = r.json()
            ts = data.get("time") or data.get("serverTime")
            return int(ts)
        except Exception:
            return int(time.time() * 1000)

    # --- public ---
    def get_klines(self, symbol: str, interval_min: int, start_ms: int, end_ms: int, category: str = "linear") -> List[Dict]:
        url = f"{self.base_url}/v5/market/kline"
        params = {
            "category": category,
            "symbol": symbol,
            "interval": str(interval_min),
            "start": str(int(start_ms)),
            "end": str(int(end_ms)),
            "limit": "1000",
        }
        out = []
        while True:
            r = self.session.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            if data.get("retCode") != 0:
                raise RuntimeError(f"Bybit kline error: {data}")
            rows = data.get("result", {}).get("list", [])
            if not rows:
                break
            rows = list(reversed(rows))  # newest first -> chronological
            for x in rows:
                out.append({
                    "ts": int(x[0]),
                    "open": float(x[1]),
                    "high": float(x[2]),
                    "low": float(x[3]),
                    "close": float(x[4]),
                    "volume": float(x[5]),
                    "turnover": float(x[6]),
                })
            last = rows[-1]
            next_start = int(last[0]) + interval_min * 60 * 1000
            if next_start >= end_ms:
                break
            params["start"] = str(next_start)
        return out

    def get_instrument(self, symbol: str, category: str = "linear") -> Dict:
        url = f"{self.base_url}/v5/market/instruments-info"
        params = {"category": category, "symbol": symbol}
        r = self.session.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("retCode") != 0:
            raise RuntimeError(f"Instruments error: {data}")
        items = data.get("result", {}).get("list", [])
        if not items:
            raise RuntimeError("No instrument info returned")
        return items[0]

    # --- private helpers / checks ---
    def query_api_key(self) -> dict:
        data = self._signed_get("/v5/user/query-api", {})
        if data.get("retCode") != 0:
            raise RuntimeError(f"query_api_key error: {data}")
        return data.get("result", {})

    def wallet_balance(self, coin: str = "USDT", accountType: str | None = None) -> float:
        types_to_try = [accountType] if accountType else ["UNIFIED", "CONTRACT"]
        last_err = None
        for acct in types_to_try:
            try:
                params = {"accountType": acct, "coin": coin}
                data = self._signed_get("/v5/account/wallet-balance", params)
                if data.get("retCode") != 0:
                    last_err = RuntimeError(str(data))
                    continue
                result = data.get("result", {})
                lst = result.get("list", [])
                if not lst:
                    continue
                return float(lst[0].get("totalEquity", 0))
            except Exception as e:
                last_err = e
        raise last_err if last_err else RuntimeError("wallet_balance failed")

    def get_positions(self, category: str, symbol: str) -> List[Dict]:
        data = self._signed_get("/v5/position/list", {"category": category, "symbol": symbol})
        if data.get("retCode") != 0:
            raise RuntimeError(f"positions error: {data}")
        return data.get("result", {}).get("list", [])

    def place_order(self, *, category: str, symbol: str, side: str, qty: str,
                    orderType: str = "Market", timeInForce: str = "IOC", reduceOnly: bool = False) -> Dict:
        url = f"{self.base_url}/v5/order/create"
        body = {
            "category": category,
            "symbol": symbol,
            "side": side,  # "Buy" or "Sell"
            "orderType": orderType,
            "qty": str(qty),
            "timeInForce": timeInForce,
            "reduceOnly": reduceOnly,
        }
        body_json = json.dumps(body, separators=(",", ":"))
        headers = self._headers(body_json)
        r = self.session.post(url, data=body_json, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("retCode") != 0:
            raise RuntimeError(f"order error: {data}")
        return data.get("result", {})

    def diagnose_auth(self, symbol: str, category: str = "linear") -> str:
        lines = []
        lines.append(f"BASE_URL={self.base_url}")
        try:
            st = self._server_time_ms()
            delta = st - int(time.time() * 1000)
            lines.append(f"Server time OK. skew={delta}ms")
        except Exception as e:
            lines.append(f"Server time fetch failed: {e}")
        try:
            info = self.query_api_key()
            perms = info.get("permissions") or info
            lines.append(f"API key OK. perms={perms}")
        except Exception as e:
            lines.append(f"API key query failed: {e}")
        for acct in ("UNIFIED", "CONTRACT"):
            try:
                eq = self.wallet_balance("USDT", accountType=acct)
                lines.append(f"wallet_balance[{acct}]: {eq} USDT")
            except Exception as e:
                lines.append(f"wallet_balance[{acct}] failed: {e}")
        try:
            pos = self.get_positions(category, symbol)
            lines.append(f"positions: {len(pos)}")
        except Exception as e:
            lines.append(f"positions failed: {e}")
        return "\n".join(lines)

# ---------------------------------------------------------------------
# Strategy state / trade record
# ---------------------------------------------------------------------

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    side: str            # 'long' or 'short'
    entry: float
    exit: Optional[float]
    size: float          # asset units
    strategy: str = "bb_countertrend"
    regime: str = "single"
    tf: str = ""

# ---------------------------------------------------------------------
# Utility: rounding to exchange step
# ---------------------------------------------------------------------

def round_qty(qty: float, qty_step: float, min_qty: float) -> float:
    if qty_step <= 0:
        return max(qty, min_qty)
    steps = math.floor(qty / qty_step)
    rounded = steps * qty_step
    if rounded < min_qty:
        return 0.0
    return float(f"{rounded:.12f}")

# ---------------------------------------------------------------------
# Core bot loop
# ---------------------------------------------------------------------

def run_bot():
    load_dotenv()
    print("[MAIN] Strategy starting…", flush=True)

    # ENV
    symbol = os.getenv("SYMBOL", "BTCUSDT")
    interval_min = int(os.getenv("INTERVAL_MIN", "60"))
    days = int(os.getenv("DAYS", "30"))
    category = os.getenv("CATEGORY", "linear")

    api_key = os.getenv("API_KEY", "").strip()
    api_secret = os.getenv("API_SECRET", "").strip()
    base_url = os.getenv("BASE_URL", "https://api-testnet.bybit.com").strip()
    recv_window = int(os.getenv("RECV_WINDOW", "5000"))

    start_balance_fallback = float(os.getenv("START_BALANCE", "1000"))
    risk_pct = float(os.getenv("RISK_PCT", "0.05"))
    commission_pct = float(os.getenv("COMMISSION_PCT", "0.0"))
    slippage_usdt = float(os.getenv("SLIPPAGE_USDT", "0.0"))

    bb_window = int(os.getenv("BB_WINDOW", "10"))
    bb_std = float(os.getenv("BB_STD", "1.5"))
    atr_window = int(os.getenv("ATR_WINDOW", "10"))
    stop_mult = float(os.getenv("STOP_MULT", "2.0"))
    trail_mult = float(os.getenv("TRAIL_MULT", "1.5"))

    poll_seconds = int(os.getenv("POLL_SECONDS", "5"))
    db_path = os.getenv("DB_PATH", "candles.db")

    if not api_key or not api_secret:
        log.error("[SETUP] Missing API_KEY or API_SECRET in environment.")
        raise SystemExit(1)

    log.info(
        f"[SETUP] LIVE TESTNET | symbol={symbol} tf={interval_min}m days={days} risk={risk_pct} "
        f"bb={bb_window},{bb_std} atr={atr_window} stop={stop_mult} trail={trail_mult}"
    )
    print(f"[MAIN] SETUP OK | symbol={symbol} tf={interval_min}m days={days} risk={risk_pct}", flush=True)

    store = CandleStore(db_path)
    client = BybitClient(api_key, api_secret, base_url, recv_window)

    # --- One-time API key self-check (more verbose) ---
    log.info(f"[CHECK] Using BASE_URL={base_url} | key={api_key[:4]}…{api_key[-4:] if len(api_key)>8 else ''}")
    try:
        test_eq = client.wallet_balance("USDT")
        log.info(f"[CHECK] API auth OK. Wallet equity (auto acct type): {test_eq:.2f} USDT")
        print(f"[MAIN] API auth OK | equity={test_eq:.2f} USDT", flush=True)
    except Exception as e:
        diag = client.diagnose_auth(symbol, category)
        log.error("[CHECK] API auth FAILED. Details:\n" + diag)
        print("[MAIN] API auth FAILED (see logs for details)", flush=True)

    # CLI always enabled via ENABLE_CLI toggle
    ENABLE_CLI = True
    if ENABLE_CLI:
        try:
            threading.Thread(target=input_loop, args=(client,), daemon=True).start()
        except Exception:
            pass

    # Instruments info (qty step)
    ins = client.get_instrument(symbol, category)
    lot = ins.get("lotSizeFilter", {})
    qty_step = float(lot.get("qtyStep", 0.001))
    min_qty = float(lot.get("minOrderQty", 0.001))
    log.info(f"[INFO] qty_step={qty_step} min_qty={min_qty}")
    print(f"[MAIN] Instrument loaded | qty_step={qty_step} min_qty={min_qty}", flush=True)

    # Bootstrap history into DB
    end = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
    start = end - dt.timedelta(days=days + 1)
    log.info("[DATA] Bootstrapping history from Bybit -> SQLite …")
    rows = client.get_klines(symbol, interval_min, to_ms(start), to_ms(end), category)
    if rows:
        store.upsert_klines(symbol, interval_min, rows)
        log.info(f"[DATA] History upserted: {len(rows)} bars")
        print(f"[MAIN] History loaded | bars={len(rows)}", flush=True)
    else:
        log.warning("[DATA] No initial klines loaded from API")

    # Position/strategy runtime state
    position = None  # dict: side, entry, size, stop, trail, tp, entry_bar_ts
    last_bar_ts = None

    while True:
        try:
            # 1) Pull latest klines segment (only recent window) and upsert
            now = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
            fetch_start = to_ms(now - dt.timedelta(days=3))  # small rolling window
            fetch_end = to_ms(now)
            new_rows = client.get_klines(symbol, interval_min, fetch_start, fetch_end, category)
            if new_rows:
                store.upsert_klines(symbol, interval_min, new_rows)

            # 2) Load frame for indicators (full DAYS window)
            frame_since = to_ms(now - dt.timedelta(days=days))
            df = store.load_frame(symbol, interval_min, frame_since)
            if len(df) < max(bb_window, atr_window) + 3:
                time.sleep(poll_seconds); continue

            # 3) Build indicators
            df = df.copy().reset_index(drop=True)
            for c in ("open","high","low","close"):
                df[c] = df[c].astype(float)
            ma, upper, lower = bollinger_bands(df["close"], bb_window, bb_std)
            atr_s = atr(df, atr_window)
            df["bb_ma"] = ma; df["bb_upper"] = upper; df["bb_lower"] = lower; df["atr"] = atr_s

            # 4) Detect bar change
            last_closed_idx = len(df) - 2  # last fully closed bar index
            last_closed = df.iloc[last_closed_idx]
            this_open_bar = df.iloc[-1]    # current forming bar

            if last_bar_ts is None:
                last_bar_ts = int(this_open_bar["ts"])  # initialize on first loop

            # If a NEW bar just started (open bar ts changed), then we execute the actions
            if int(this_open_bar["ts"]) != last_bar_ts:
                # A new bar opened at this_open_bar.open; we execute actions based on last_closed signal/state
                new_bar_open = float(this_open_bar["open"])  # execution price baseline
                print(f"[MAIN] New {interval_min}m bar: {this_open_bar['datetime']} open={new_bar_open:.2f}", flush=True)

                # Fetch equity (wallet balance) to size trades
                try:
                    equity = float(client.wallet_balance("USDT"))
                except Exception as e:
                    log.warning(f"[API] wallet_balance failed (check TESTNET keys/base URL, IP whitelist, permissions). Fallback START_BALANCE. Err={e}")
                    equity = start_balance_fallback

                # Update trailing stop and check exits at bar open
                if position is not None:
                    atr_now = float(last_closed["atr"]) if not math.isnan(last_closed["atr"]) else None
                    if atr_now and atr_now > 0:
                        if position["side"] == "long":
                            new_trail = new_bar_open - trail_mult * atr_now
                            position["trail"] = max(position["trail"], new_trail)
                            stop_level = max(position["stop"], position["trail"])
                            if new_bar_open <= stop_level:
                                qty = round_qty(position["size"], qty_step, min_qty)
                                if qty > 0:
                                    client.place_order(category=category, symbol=symbol, side="Sell", qty=str(qty), reduceOnly=True)
                                log.info(f"[EXIT] long STOP t={this_open_bar['datetime']} entry={position['entry']:.2f} open={new_bar_open:.2f} stop={stop_level:.2f}")
                                print(f"[MAIN] EXIT long STOP | t={this_open_bar['datetime']} entry={position['entry']:.2f} open={new_bar_open:.2f} stop={stop_level:.2f}", flush=True)
                                position = None
                        else:
                            new_trail = new_bar_open + trail_mult * atr_now
                            position["trail"] = min(position["trail"], new_trail)
                            stop_level = min(position["stop"], position["trail"])
                            if new_bar_open >= stop_level:
                                qty = round_qty(position["size"], qty_step, min_qty)
                                if qty > 0:
                                    client.place_order(category=category, symbol=symbol, side="Buy", qty=str(qty), reduceOnly=True)
                                log.info(f"[EXIT] short STOP t={this_open_bar['datetime']} entry={position['entry']:.2f} open={new_bar_open:.2f} stop={stop_level:.2f}")
                                print(f"[MAIN] EXIT short STOP | t={this_open_bar['datetime']} entry={position['entry']:.2f} open={new_bar_open:.2f} stop={stop_level:.2f}", flush=True)
                                position = None

                # If still in position, check TP at middle band (basis) from signal bar
                if position is not None:
                    tp = float(position.get("tp", float("nan")))
                    if not math.isnan(tp):
                        if position["side"] == "long" and new_bar_open >= tp:
                            qty = round_qty(position["size"], qty_step, min_qty)
                            if qty > 0:
                                client.place_order(category=category, symbol=symbol, side="Sell", qty=str(qty), reduceOnly=True)
                            log.info(f"[EXIT] long TP t={this_open_bar['datetime']} entry={position['entry']:.2f} tp={tp:.2f} open={new_bar_open:.2f}")
                            print(f"[MAIN] EXIT long TP | t={this_open_bar['datetime']} entry={position['entry']:.2f} tp={tp:.2f} open={new_bar_open:.2f}", flush=True)
                            position = None
                        elif position["side"] == "short" and new_bar_open <= tp:
                            qty = round_qty(position["size"], qty_step, min_qty)
                            if qty > 0:
                                client.place_order(category=category, symbol=symbol, side="Buy", qty=str(qty), reduceOnly=True)
                            log.info(f"[EXIT] short TP t={this_open_bar['datetime']} entry={position['entry']:.2f} tp={tp:.2f} open={new_bar_open:.2f}")
                            print(f"[MAIN] EXIT short TP | t={this_open_bar['datetime']} entry={position['entry']:.2f} tp={tp:.2f} open={new_bar_open:.2f}", flush=True)
                            position = None

                # If flat, evaluate entry signals from last_closed and execute at new bar open
                if position is None:
                    lower_band = float(last_closed["bb_lower"])
                    upper_band = float(last_closed["bb_upper"])
                    mid = float(last_closed["bb_ma"])
                    atr_now = float(last_closed["atr"]) if not math.isnan(last_closed["atr"]) else 0.0

                    if atr_now > 0 and not any(map(math.isnan, [lower_band, upper_band, mid])):
                        if last_closed["close"] <= lower_band:
                            stop_dist = stop_mult * atr_now
                            size = (equity * risk_pct) / stop_dist if stop_dist > 0 else 0.0
                            entry_price = new_bar_open + slippage_usdt
                            qty = round_qty(size, qty_step, min_qty)
                            if qty > 0:
                                client.place_order(category=category, symbol=symbol, side="Buy", qty=str(qty), reduceOnly=False)
                                position = {
                                    "side": "long",
                                    "entry": entry_price * (1 + commission_pct),
                                    "size": qty,
                                    "stop": (entry_price * (1 + commission_pct)) - stop_dist,
                                    "trail": (entry_price * (1 + commission_pct)) - stop_dist,
                                    "tp": mid,
                                    "entry_bar_ts": int(this_open_bar["ts"]),
                                }
                                log.info(f"[ENTRY] long t={this_open_bar['datetime']} reason=bb_lower_touch entry={entry_price:.2f} size={qty:.6f} stop_dist={stop_dist:.2f} eq={equality_fmt(equity)}")
                                print(f"[MAIN] ENTRY long | t={this_open_bar['datetime']} entry={entry_price:.2f} qty={qty:.6f} stop_dist={stop_dist:.2f}", flush=True)

                        elif last_closed["close"] >= upper_band:
                            stop_dist = stop_mult * atr_now
                            size = (equity * risk_pct) / stop_dist if stop_dist > 0 else 0.0
                            entry_price = new_bar_open - slippage_usdt
                            qty = round_qty(size, qty_step, min_qty)
                            if qty > 0:
                                client.place_order(category=category, symbol=symbol, side="Sell", qty=str(qty), reduceOnly=False)
                                position = {
                                    "side": "short",
                                    "entry": entry_price * (1 - commission_pct),
                                    "size": qty,
                                    "stop": (entry_price * (1 - commission_pct)) + stop_dist,
                                    "trail": (entry_price * (1 - commission_pct)) + stop_dist,
                                    "tp": mid,
                                    "entry_bar_ts": int(this_open_bar["ts"]),
                                }
                                log.info(f"[ENTRY] short t={this_open_bar['datetime']} reason=bb_upper_touch entry={entry_price:.2f} size={qty:.6f} stop_dist={stop_dist:.2f} eq={equality_fmt(equity)}")
                                print(f"[MAIN] ENTRY short | t={this_open_bar['datetime']} entry={entry_price:.2f} qty={qty:.6f} stop_dist={stop_dist:.2f}", flush=True)

                # mark current open bar as observed
                last_bar_ts = int(this_open_bar["ts"])

            # Optional very verbose per-bar log
            if LOG_BAR_DETAILS and not LOG_TRADES_ONLY:
                log.debug(f"[BAR] t={last_closed['datetime']} close={last_closed['close']:.2f} "
                          f"bb_ma={last_closed['bb_ma']:.2f} atr={last_closed['atr']:.2f} "
                          f"pos={'None' if position is None else position['side']}")

        except Exception as e:
            log.error(f"[LOOP] Error: {e}")
        time.sleep(poll_seconds)


if __name__ == "__main__":
    run_bot()
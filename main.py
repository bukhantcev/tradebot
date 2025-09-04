
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
from typing import Optional, List, Dict, Tuple
import threading
import urllib.parse

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# CLI (stdin commands)
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
# SQLite storage (multi-interval candles)
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
        if not rows:
            return
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
# Indicators
# ---------------------------------------------------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0):
    ma = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = ma + num_std * std
    lower = ma - num_std * std
    return ma, upper, lower

def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()

def add_indicators(df: pd.DataFrame, bb_window: int, bb_std: float, atr_window: int, ema_fast: int, ema_slow: int) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = ema(out["close"], ema_fast)
    out["ema_slow"] = ema(out["close"], ema_slow)
    ma, up, lo = bollinger_bands(out["close"], bb_window, bb_std)
    out["bb_ma"], out["bb_upper"], out["bb_lower"] = ma, up, lo
    out["atr"] = atr(out, atr_window)
    # bandwidth in % of price
    out["bb_bw_pct"] = (out["bb_upper"] - out["bb_lower"]) / out["close"]
    return out

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

    def available_balance(self, coin: str = "USDT", accountType: str | None = None) -> float:
        """Return available balance (not total equity). Uses totalAvailableBalance if present, else availableBalance, else 0."""
        types_to_try = [accountType] if accountType else ["UNIFIED", "CONTRACT"]
        last_err = None
        for acct in types_to_try:
            try:
                params = {"accountType": acct, "coin": coin}
                data = self._signed_get("/v5/account/wallet-balance", params)
                if data.get("retCode") != 0:
                    last_err = RuntimeError(str(data)); continue
                lst = (data.get("result", {}) or {}).get("list", [])
                if not lst:
                    continue
                item = lst[0]
                # Bybit v5 returns strings
                val = item.get("totalAvailableBalance") or item.get("availableBalance") or "0"
                try:
                    return float(val)
                except Exception:
                    return 0.0
            except Exception as e:
                last_err = e
        raise last_err if last_err else RuntimeError("available_balance failed")

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
# Strategy state
# ---------------------------------------------------------------------

@dataclass
class PositionState:
    side: str              # "long" / "short"
    entry: float
    size: float
    stop: float
    trail: float
    tp: Optional[float]
    strategy: str
    regime: str
    tf: str
    entry_bar_ts: int

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
# Utility
# ---------------------------------------------------------------------

def parse_intervals_csv(s: str) -> List[int]:
    try:
        arr = [int(x.strip()) for x in s.split(",") if x.strip()]
        arr = sorted(set(arr))
        return arr
    except Exception:
        return [15, 60, 240]

# ---------------------------------------------------------------------
# Adaptive controller
# ---------------------------------------------------------------------

def detect_regime(higher: pd.DataFrame, atr_pct_thr: float, bb_bw_thr: float, slope_thr: float) -> str:
    """Return one of: trend_up, trend_down, range, breakout."""
    if len(higher) < 60:
        return "range"
    df = higher.copy()
    price = float(df["close"].iloc[-2])
    ema_slow = float(df["ema_slow"].iloc[-2])
    ema_slow_prev = float(df["ema_slow"].iloc[-3])
    slope = ema_slow - ema_slow_prev
    atr_val = float(df["atr"].iloc[-2])
    atr_pct = atr_val / price if price > 0 else 0.0
    bb_bw = float(df["bb_bw_pct"].iloc[-2])

    # auto slope threshold if not provided
    if slope_thr <= 0:
        slope_thr = atr_pct * price * 0.10  # 10% of ATR as slope gate (heuristic)

    if bb_bw < bb_bw_thr and atr_pct < atr_pct_thr:
        return "range"
    if slope > slope_thr and price > ema_slow:
        return "trend_up"
    if slope < -slope_thr and price < ema_slow:
        return "trend_down"

    # else, potential breakout if bandwidth expanding and price outside bands
    up = float(df["bb_upper"].iloc[-2]); lo = float(df["bb_lower"].iloc[-2])
    prev_bw = float(df["bb_bw_pct"].iloc[-3]) if not math.isnan(df["bb_bw_pct"].iloc[-3]) else bb_bw
    if bb_bw > prev_bw * 1.2 and (price > up or price < lo):
        return "breakout"
    return "range"

def choose_exec_tf(regime: str, intervals: List[int]) -> int:
    intervals = sorted(intervals)
    if len(intervals) == 0:
        return 60
    if regime == "range":
        return intervals[0]                 # fastest TF
    elif regime in ("trend_up", "trend_down"):
        return intervals[min(1, len(intervals)-1)]  # base TF
    else:
        return intervals[min(1, len(intervals)-1)]

def generate_signal(base_df: pd.DataFrame, regime: str, params: Dict) -> Tuple[Optional[str], str, Optional[float]]:
    """
    Returns (side, strategy_name, tp_level)
    side in {"long","short", None}
    """
    last = base_df.iloc[-2]
    open_bar = base_df.iloc[-1]  # current forming bar (exec @ open)
    atr_now = float(last["atr"]) if not math.isnan(last["atr"]) else 0.0
    if atr_now <= 0:
        return None, "", None

    if regime in ("trend_up", "trend_down"):
        # Trend pullback: enter in direction of ema_slow slope when price pulls to ema_fast ± k*ATR
        k = params.get("PULL_K_ATR", 0.5)
        ema_fast = float(last["ema_fast"]); ema_slow = float(last["ema_slow"])
        slope = ema_slow - float(base_df["ema_slow"].iloc[-3])
        if regime == "trend_up" and slope >= 0:
            trigger = ema_fast - k * atr_now
            if float(last["close"]) <= trigger:
                return "long", "trend_ma_pullback", float(last["ema_fast"])
        if regime == "trend_down" and slope <= 0:
            trigger = ema_fast + k * atr_now
            if float(last["close"]) >= trigger:
                return "short", "trend_ma_pullback", float(last["ema_fast"])
        return None, "", None

    if regime == "range":
        # Counter-trend Bollinger: touch outer band -> revert to mid
        lower = float(last["bb_lower"]); upper = float(last["bb_upper"]); mid = float(last["bb_ma"])
        if not any(map(math.isnan, [lower, upper, mid])):
            if float(last["close"]) <= lower:
                return "long", "range_bb", mid
            if float(last["close"]) >= upper:
                return "short", "range_bb", mid
        return None, "", None

    if regime == "breakout":
        # Simple HH/LL breakout over N bars with ATR gate
        N = params.get("BRK_N", 20)
        recent = base_df.iloc[-(N+1):-1]
        hi = float(recent["high"].max()); lo = float(recent["low"].min())
        c = float(last["close"])
        if c > hi and atr_now / c > 0.004:
            return "long", "breakout", None
        if c < lo and atr_now / c > 0.004:
            return "short", "breakout", None
        return None, "", None

    return None, "", None

# ---------------------------------------------------------------------
# Core bot loop
# ---------------------------------------------------------------------

def run_bot():
    load_dotenv()
    print("[MAIN] Strategy starting…", flush=True)

    # ENV
    symbol = os.getenv("SYMBOL", "BTCUSDT")
    category = os.getenv("CATEGORY", "linear")
    adaptive = os.getenv("ADAPTIVE", "1").strip().lower() in ("1","true","yes","y")
    multi_intervals = parse_intervals_csv(os.getenv("MULTI_INTERVALS", "15,60,240"))
    interval_min = int(os.getenv("INTERVAL_MIN", "60"))  # used if adaptive==False
    days = int(os.getenv("DAYS", "45"))

    api_key = os.getenv("API_KEY", "").strip()
    api_secret = os.getenv("API_SECRET", "").strip()
    base_url = os.getenv("BASE_URL", "https://api-testnet.bybit.com").strip()
    recv_window = int(os.getenv("RECV_WINDOW", "5000"))

    start_balance_fallback = float(os.getenv("START_BALANCE", "1000"))
    risk_pct = float(os.getenv("RISK_PCT", "0.05"))
    commission_pct = float(os.getenv("COMMISSION_PCT", "0.0"))
    slippage_usdt = float(os.getenv("SLIPPAGE_USDT", "0.0"))

    bb_window = int(os.getenv("BB_WINDOW", "20"))
    bb_std = float(os.getenv("BB_STD", "2.0"))
    atr_window = int(os.getenv("ATR_WINDOW", "14"))
    ema_fast_n = int(os.getenv("EMA_FAST", "20"))
    ema_slow_n = int(os.getenv("EMA_SLOW", "50"))
    stop_mult = float(os.getenv("STOP_MULT", "2.0"))
    trail_mult = float(os.getenv("TRAIL_MULT", "1.0"))

    slope_thr = float(os.getenv("SLOPE_THR", "0.0"))
    atr_pct_thr = float(os.getenv("ATR_PCT_THR", "0.006"))
    bb_bw_thr = float(os.getenv("BB_BW_THR", "0.02"))

    poll_seconds = int(os.getenv("POLL_SECONDS", "5"))
    db_path = os.getenv("DB_PATH", "candles.db")

    # Risk circuit breakers / caps
    equity_cap = float(os.getenv("EQUITY_CAP", "0"))
    daily_max_dd_pct = float(os.getenv("DAILY_MAX_DRAWDOWN_PCT", "0"))
    max_consec_stops = int(os.getenv("MAX_CONSECUTIVE_STOPS", "0"))
    cooldown_bars_after_stop = int(os.getenv("COOLDOWN_BARS_AFTER_STOP", "0"))
    max_trade_notional_usdt = float(os.getenv("MAX_TRADE_NOTIONAL_USDT", "0"))
    atr_pos_cap_pct = float(os.getenv("ATR_POS_CAP_PCT", "0"))

    if not api_key or not api_secret:
        log.error("[SETUP] Missing API_KEY or API_SECRET in environment.")
        raise SystemExit(1)

    log.info(f"[SETUP] symbol={symbol} adaptive={adaptive} tfs={','.join(map(str,multi_intervals))} days={days} risk={risk_pct} "
             f"stopATRx={stop_mult} trailATRx={trail_mult}")
    print(f"[MAIN] SETUP OK | symbol={symbol} adaptive={adaptive} tfs={multi_intervals} days={days} risk={risk_pct}", flush=True)

    store = CandleStore(db_path)
    client = BybitClient(api_key, api_secret, base_url, recv_window)

    # One-time auth check
    try:
        test_eq = client.wallet_balance("USDT")
        log.info(f"[CHECK] API auth OK. Wallet equity: {test_eq:.2f} USDT")
        print(f"[MAIN] API auth OK | equity={test_eq:.2f} USDT", flush=True)
    except Exception as e:
        diag = client.diagnose_auth(symbol, category)
        log.error("[CHECK] API auth FAILED. Details:\n" + diag)
        print("[MAIN] API auth FAILED (see logs)", flush=True)

    # CLI thread
    ENABLE_CLI = True
    if ENABLE_CLI:
        try:
            threading.Thread(target=input_loop, args=(client,), daemon=True).start()
        except Exception:
            pass

    # Instrument info
    ins = client.get_instrument(symbol, category)
    lot = ins.get("lotSizeFilter", {})
    qty_step = float(lot.get("qtyStep", 0.001))
    min_qty = float(lot.get("minOrderQty", 0.001))
    log.info(f"[INFO] qty_step={qty_step} min_qty={min_qty}")
    print(f"[MAIN] Instrument loaded | qty_step={qty_step} min_qty={min_qty}", flush=True)

    # Bootstrap history for all required TFs
    def bootstrap_tf(tf_min: int):
        end = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
        start = end - dt.timedelta(days=days + 1)
        rows = client.get_klines(symbol, tf_min, to_ms(start), to_ms(end), category)
        if rows:
            store.upsert_klines(symbol, tf_min, rows)
            log.info(f"[DATA] History upserted: tf={tf_min} bars={len(rows)}")

    if adaptive:
        for tf in multi_intervals:
            bootstrap_tf(tf)
    else:
        bootstrap_tf(interval_min)

    position: Optional[PositionState] = None
    last_open_ts_by_tf: Dict[int, int] = {}

    # Risk/guard state
    day_anchor_date: Optional[dt.date] = None
    day_start_equity: Optional[float] = None
    consecutive_stops: int = 0
    cooldown_bars_left: int = 0

    # Strategy params bag
    strat_params = {"PULL_K_ATR": 0.5, "BRK_N": 20}

    log.info(f"[ADAPT] Start | tfs={multi_intervals} risk={risk_pct} stopATRx={stop_mult} trailATRx={trail_mult}")

    while True:
        try:
            now = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)

            # Update recent candles for all needed TFs
            tfs_needed = multi_intervals if adaptive else [interval_min]
            for tf in tfs_needed:
                fetch_start = to_ms(now - dt.timedelta(days=3))
                fetch_end = to_ms(now)
                new_rows = client.get_klines(symbol, tf, fetch_start, fetch_end, category)
                if new_rows:
                    store.upsert_klines(symbol, tf, new_rows)

            # Build frames with indicators
            frames: Dict[int, pd.DataFrame] = {}
            for tf in tfs_needed:
                frame_since = to_ms(now - dt.timedelta(days=days))
                df = store.load_frame(symbol, tf, frame_since)
                if len(df) == 0:
                    continue
                df = add_indicators(df, bb_window, bb_std, atr_window, ema_fast_n, ema_slow_n)
                frames[tf] = df

            if not frames:
                time.sleep(poll_seconds); continue

            # Decide regime & execution TF
            if adaptive:
                higher_tf = max(tfs_needed)
                if higher_tf not in frames or len(frames[higher_tf]) < max(atr_window, ema_slow_n) + 3:
                    time.sleep(poll_seconds); continue
                regime = detect_regime(frames[higher_tf], atr_pct_thr, bb_bw_thr, slope_thr)
                exec_tf = choose_exec_tf(regime, tfs_needed)
            else:
                regime = "range"  # legacy default makes BB-driven
                exec_tf = interval_min

            base_df = frames.get(exec_tf)
            if base_df is None or len(base_df) < max(atr_window, ema_slow_n, bb_window) + 3:
                time.sleep(poll_seconds); continue

            # Bar change detection on execution TF
            open_ts = int(base_df.iloc[-1]["ts"])
            if last_open_ts_by_tf.get(exec_tf) is None:
                last_open_ts_by_tf[exec_tf] = open_ts

            if open_ts != last_open_ts_by_tf[exec_tf]:
                # New bar opened on execution TF
                new_open = float(base_df.iloc[-1]["open"])
                print(f"[MAIN] New {exec_tf}m bar: {base_df.iloc[-1]['datetime']} open={new_open:.2f} regime={regime}", flush=True)

                # Daily anchor & reset
                bar_date = base_df.iloc[-1]['datetime'].date()
                try:
                    current_equity = float(client.wallet_balance("USDT"))
                except Exception:
                    current_equity = start_balance_fallback
                if day_anchor_date != bar_date:
                    day_anchor_date = bar_date
                    day_start_equity = current_equity
                    consecutive_stops = 0
                    # cooldown carries over but can be reduced one step on bar change below

                # equity for sizing
                try:
                    equity = float(client.wallet_balance("USDT"))
                except Exception as e:
                    log.warning(f"[API] wallet_balance failed, fallback START_BALANCE. Err={e}")
                    equity = start_balance_fallback

                # Exits first (stop/trail/TP)
                if position is not None:
                    last_closed = base_df.iloc[-2]
                    atr_now = float(last_closed["atr"]) if not math.isnan(last_closed["atr"]) else None
                    if atr_now and atr_now > 0:
                        if position.side == "long":
                            new_trail = new_open - trail_mult * atr_now
                            position.trail = max(position.trail, new_trail)
                            stop_level = max(position.stop, position.trail)
                            if new_open <= stop_level:
                                qty = round_qty(position.size, qty_step, min_qty)
                                if qty > 0:
                                    client.place_order(category=category, symbol=symbol, side="Sell", qty=str(qty), reduceOnly=True)
                                log.info(f"[ADAPT][EXIT] long STOP t={base_df.iloc[-1]['datetime']} regime={position.regime} strat={position.strategy} tf={position.tf} entry={position.entry:.2f} exit={new_open:.2f}")
                                consecutive_stops += 1
                                if cooldown_bars_after_stop > 0:
                                    cooldown_bars_left = max(cooldown_bars_left, cooldown_bars_after_stop)
                                position = None
                        else:
                            new_trail = new_open + trail_mult * atr_now
                            position.trail = min(position.trail, new_trail)
                            stop_level = min(position.stop, position.trail)
                            if new_open >= stop_level:
                                qty = round_qty(position.size, qty_step, min_qty)
                                if qty > 0:
                                    client.place_order(category=category, symbol=symbol, side="Buy", qty=str(qty), reduceOnly=True)
                                log.info(f"[ADAPT][EXIT] short STOP t={base_df.iloc[-1]['datetime']} regime={position.regime} strat={position.strategy} tf={position.tf} entry={position.entry:.2f} exit={new_open:.2f}")
                                consecutive_stops += 1
                                if cooldown_bars_after_stop > 0:
                                    cooldown_bars_left = max(cooldown_bars_left, cooldown_bars_after_stop)
                                position = None

                # TP at target (if defined)
                if position is not None and position.tp is not None:
                    if position.side == "long" and new_open >= position.tp:
                        qty = round_qty(position.size, qty_step, min_qty)
                        if qty > 0:
                            client.place_order(category=category, symbol=symbol, side="Sell", qty=str(qty), reduceOnly=True)
                        log.info(f"[ADAPT][EXIT] long TP t={base_df.iloc[-1]['datetime']} strat={position.strategy} tf={position.tf} tp={position.tp:.2f} open={new_open:.2f}")
                        consecutive_stops = 0
                        position = None
                    elif position.side == "short" and new_open <= position.tp:
                        qty = round_qty(position.size, qty_step, min_qty)
                        if qty > 0:
                            client.place_order(category=category, symbol=symbol, side="Buy", qty=str(qty), reduceOnly=True)
                        log.info(f"[ADAPT][EXIT] short TP t={base_df.iloc[-1]['datetime']} strat={position.strategy} tf={position.tf} tp={position.tp:.2f} open={new_open:.2f}")
                        consecutive_stops = 0
                        position = None

                # Cooldown management (decrement per new bar on exec TF)
                if cooldown_bars_left > 0:
                    cooldown_bars_left -= 1

                # Daily max drawdown guard
                if day_start_equity is not None and daily_max_dd_pct > 0:
                    loss_pct = 0.0
                    if current_equity > 0:
                        loss_pct = (day_start_equity - current_equity) / day_start_equity * 100.0
                    if loss_pct >= daily_max_dd_pct:
                        log.warning(f"[GUARD] Daily DD reached: {loss_pct:.2f}% >= {daily_max_dd_pct}%. Skipping entries for today.")
                        # Mark bar observed and continue
                        last_open_ts_by_tf[exec_tf] = open_ts
                        continue

                # Stop streak guard
                if max_consec_stops > 0 and consecutive_stops >= max_consec_stops:
                    log.warning(f"[GUARD] Consecutive stops = {consecutive_stops} >= {max_consec_stops}. Skipping entry.")
                    last_open_ts_by_tf[exec_tf] = open_ts
                    continue

                # Cooldown guard
                if cooldown_bars_left > 0:
                    log.info(f"[GUARD] Cooldown active: {cooldown_bars_left} bars left. Skipping entry.")
                    last_open_ts_by_tf[exec_tf] = open_ts
                    continue

                # Entries
                if position is None:
                    side, strategy_name, tp_level = generate_signal(base_df, regime, strat_params)
                    last_closed = base_df.iloc[-2]
                    atr_now = float(last_closed["atr"]) if not math.isnan(last_closed["atr"]) else 0.0
                    if side and atr_now > 0:
                        stop_dist = stop_mult * atr_now
                        # Base equity for sizing with optional cap
                        equity_for_risk = equity
                        if equity_cap and equity_cap > 0:
                            equity_for_risk = min(equity_for_risk, equity_cap)
                        size_usdt = equity_for_risk * risk_pct

                        # Volatility cap: if ATR% exceeds threshold, halve position
                        atr_pct_now = (atr_now / float(last_closed["close"])) if float(last_closed["close"]) > 0 else 0.0
                        if atr_pos_cap_pct and atr_pos_cap_pct > 0 and atr_pct_now * 100 >= atr_pos_cap_pct:
                            size_usdt *= 0.5
                            log.info(f"[GUARD] ATR% {atr_pct_now*100:.2f}% >= {atr_pos_cap_pct}%. Halving size.")

                        # Convert to qty via stop distance (risk per trade)
                        size = size_usdt / stop_dist if stop_dist > 0 else 0.0
                        entry_price = new_open + (slippage_usdt if side == "long" else -slippage_usdt)

                        # Notional cap
                        if max_trade_notional_usdt and max_trade_notional_usdt > 0:
                            max_qty_by_notional = max_trade_notional_usdt / entry_price
                        else:
                            max_qty_by_notional = float("inf")

                        # Available balance cap (conservative: assume 1x if leverage unknown)
                        try:
                            avail = float(client.available_balance("USDT"))
                        except Exception:
                            avail = equity  # fallback
                        max_qty_by_avail = (avail * 0.9) / entry_price  # keep some buffer

                        # Apply all caps and round
                        raw_qty = min(size, max_qty_by_notional, max_qty_by_avail)
                        qty = round_qty(raw_qty, qty_step, min_qty)

                        if qty <= 0:
                            log.warning(f"[GUARD] Computed qty <= 0 after caps (size={size:.6f}, notional_cap={max_trade_notional_usdt}, avail={avail:.2f}). Skipping entry.")
                            last_open_ts_by_tf[exec_tf] = open_ts
                            continue

                        if qty > 0:
                            if side == "long":
                                client.place_order(category=category, symbol=symbol, side="Buy", qty=str(qty), reduceOnly=False)
                                entry = entry_price * (1 + commission_pct)
                                position = PositionState(
                                    side="long",
                                    entry=entry,
                                    size=qty,
                                    stop=entry - stop_dist,
                                    trail=entry - stop_dist,
                                    tp=tp_level,
                                    strategy=strategy_name,
                                    regime=regime,
                                    tf=str(exec_tf),
                                    entry_bar_ts=open_ts
                                )
                            else:
                                client.place_order(category=category, symbol=symbol, side="Sell", qty=str(qty), reduceOnly=False)
                                entry = entry_price * (1 - commission_pct)
                                position = PositionState(
                                    side="short",
                                    entry=entry,
                                    size=qty,
                                    stop=entry + stop_dist,
                                    trail=entry + stop_dist,
                                    tp=tp_level,
                                    strategy=strategy_name,
                                    regime=regime,
                                    tf=str(exec_tf),
                                    entry_bar_ts=open_ts
                                )
                            log.info(f"[ADAPT][ENTRY] {position.side} t={base_df.iloc[-1]['datetime']} regime={regime} strat={strategy_name} tf={exec_tf} entry={entry_price:.2f} size={qty:.6f} stop_dist={stop_dist:.2f} eq={equality_fmt(equity)}")
                            print(f"[MAIN] ENTRY {position.side.upper()} | tf={exec_tf} regime={regime} strat={strategy_name} entry={entry_price:.2f} qty={qty:.6f}", flush=True)

                # Lightweight balance debug
                try:
                    avail_dbg = client.available_balance("USDT")
                    log.debug(f"[BAL] equity={equality_fmt(equity)} avail={avail_dbg:.2f} USDT")
                except Exception:
                    pass

                # Mark bar observed
                last_open_ts_by_tf[exec_tf] = open_ts

            # Optional per-bar logs
            if LOG_BAR_DETAILS and not LOG_TRADES_ONLY:
                for tf, df in frames.items():
                    lc = df.iloc[-2]
                    log.debug(f"[BAR] tf={tf} t={lc['datetime']} close={lc['close']:.2f} ema_fast={lc['ema_fast']:.2f} ema_slow={lc['ema_slow']:.2f} atr={lc['atr']:.2f}")

        except Exception as e:
            log.error(f"[LOOP] Error: {e}")
        time.sleep(poll_seconds)

if __name__ == "__main__":
    run_bot()
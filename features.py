import logging
from typing import Dict, Any
import pandas as pd
import numpy as np
import aiosqlite

from config import DB_PATH, SYMBOL

logger = logging.getLogger("FEATURES")

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    # df must have columns: high, low, close
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(),
                    (h - prev_c).abs(),
                    (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()

async def load_recent_1m(limit: int = 500, symbol: str = SYMBOL) -> pd.DataFrame:
    async with aiosqlite.connect(DB_PATH) as db:
        q = """
        SELECT ts_ms, open, high, low, close, volume, turnover
        FROM bars_1m
        WHERE symbol=?
        ORDER BY ts_ms DESC
        LIMIT ?;
        """
        rows = []
        async with db.execute(q, (symbol, limit)) as cur:
            rows = await cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["ts_ms","open","high","low","close","volume","turnover"])
    df = pd.DataFrame(rows, columns=["ts_ms","open","high","low","close","volume","turnover"])
    df = df.sort_values("ts_ms")
    return df

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    adds: ema_fast, ema_slow, atr14, roc1, spread_proxy, vol_roll
    """
    if df.empty:
        return df
    d = df.copy()
    d["ema_fast"] = ema(d["close"], 12)
    d["ema_slow"] = ema(d["close"], 48)
    d["atr14"] = atr(d[["high","low","close"]], 14)
    d["roc1"] = d["close"].pct_change(1).fillna(0.0)
    # прокси спрэда: |close-open| / close
    d["spread_proxy"] = (d["close"].sub(d["open"]).abs() / d["close"]).fillna(0.0)
    d["vol_roll"] = d["volume"].rolling(20, min_periods=1).mean()
    return d

def last_feature_row(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}
    row = df.iloc[-1]
    return {
        "ts_ms": int(row["ts_ms"]),
        "close": float(row["close"]),
        "ema_fast": float(row["ema_fast"]),
        "ema_slow": float(row["ema_slow"]),
        "atr14": float(row["atr14"]),
        "roc1": float(row["roc1"]),
        "spread_proxy": float(row["spread_proxy"]),
        "vol_roll": float(row["vol_roll"]),
    }
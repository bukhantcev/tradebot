import logging
import pandas as pd
import numpy as np
from typing import Dict, Any
import aiosqlite
from config import DB_PATH, SYMBOL
import asyncio
logger = logging.getLogger("FEAT")

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    # df: columns = open, high, low, close
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def build_features(df_1m: pd.DataFrame, df_5m: pd.DataFrame) -> Dict[str, Any]:
    """
    На вход: последние ~300 баров 1m и ~200 баров 5m (минимум по 100).
    Возвращает словарь признаков для текущего t (последняя закрытая 1m свеча).
    """
    if len(df_1m) < 100 or len(df_5m) < 50:
        return {"ready": False}

    d1 = df_1m.copy()
    d1["ema12"] = ema(d1["close"], 12)
    d1["ema48"] = ema(d1["close"], 48)
    d1["atr14"] = atr(d1.rename(columns={"ts_ms":"ts"}), 14)
    d1["roc3"]  = d1["close"].pct_change(3)
    d1["roc10"] = d1["close"].pct_change(10)
    d1["spread"] = (d1["high"] - d1["low"]) / d1["close"]
    d1 = d1.dropna()

    d5 = df_5m.copy()
    d5["ema12"] = ema(d5["close"], 12)
    d5["ema48"] = ema(d5["close"], 48)
    d5["atr14"] = atr(d5.rename(columns={"ts_ms":"ts"}), 14)
    d5 = d5.dropna()

    x = d1.iloc[-1]
    y5 = d5.iloc[-1]

    feats = {
        "ready": True,
        "c": float(x["close"]),
        "emaF": float(x["ema12"]),
        "emaS": float(x["ema48"]),
        "atr":  float(x["atr14"]),
        "roc3": float(x["roc3"]),
        "roc10": float(x["roc10"]),
        "spread": float(x["spread"]),
        "trend5": float(y5["ema12"] - y5["ema48"]),
        "vola5": float(y5["atr14"]),
    }
    for k in ("roc3","roc10","spread"):
        feats[k] = float(np.clip(feats[k], -0.05, 0.05))
    return feats

# ----------------------------
# API для стратегии
# ----------------------------
async def load_recent_1m(limit: int = 500, symbol: str = SYMBOL) -> pd.DataFrame:
    """
    Грузим ТОЛЬКО закрытые 1m свечи (confirm=1), отдаём по возрастанию ts_ms.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        q = """
        SELECT ts_ms, open, high, low, close, volume, turnover
        FROM bars_1m
        WHERE symbol=? AND confirm=1
        ORDER BY ts_ms DESC
        LIMIT ?;
        """
        rows = []
        async with db.execute(q, (symbol, limit)) as cur:
            rows = await cur.fetchall()
    cols = ["ts_ms","open","high","low","close","volume","turnover"]
    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return pd.DataFrame(columns=cols)
    return df.sort_values("ts_ms").reset_index(drop=True)

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Признаки для StrategyEngine / OnlineModel.
    """
    if df.empty:
        return df
    d = df.copy()
    d["ema_fast"] = ema(d["close"], 12)
    d["ema_slow"] = ema(d["close"], 48)
    d["atr14"] = atr(d[["high","low","close"]], 14)
    d["roc1"] = d["close"].pct_change(1).fillna(0.0)
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
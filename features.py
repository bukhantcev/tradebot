import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

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
    # защитные клипы
    for k in ("roc3","roc10","spread"):
        feats[k] = float(np.clip(feats[k], -0.05, 0.05))
    return feats
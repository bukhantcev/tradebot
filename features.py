from typing import List, Dict, Any

def ema(series: List[float], length: int) -> List[float]:
    if length <= 1 or len(series) == 0: return series[:]
    k = 2/(length+1)
    out = []
    prev = series[0]
    out.append(prev)
    for x in series[1:]:
        prev = x*k + prev*(1-k)
        out.append(prev)
    return out

def rsi(prices: List[float], length: int) -> List[float]:
    if len(prices) < length+1: return [50.0]*len(prices)
    gains, losses = [0.0], [0.0]
    for i in range(1, len(prices)):
        diff = prices[i]-prices[i-1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    avg_gain = sum(gains[1:length+1])/length
    avg_loss = sum(losses[1:length+1])/length
    rsis = [50.0]*(length)
    for i in range(length+1, len(prices)+1):
        g = gains[i-1]; l = losses[i-1]
        avg_gain = (avg_gain*(length-1)+g)/length
        avg_loss = (avg_loss*(length-1)+l)/length
        rs = avg_gain / (avg_loss if avg_loss>0 else 1e-9)
        r = 100 - (100/(1+rs))
        rsis.append(r)
    while len(rsis) < len(prices): rsis.append(rsis[-1] if rsis else 50.0)
    return rsis

def atr(high: List[float], low: List[float], close: List[float], length: int) -> List[float]:
    trs = []
    for i in range(len(close)):
        if i == 0:
            trs.append(high[i]-low[i])
        else:
            tr = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
            trs.append(tr)
    if length <= 1: return trs
    # EMA для простоты
    return ema(trs, length)

def compute_indicators(candles: List[Dict[str, Any]], p: Dict[str, Any]) -> Dict[str, Any]:
    closes = [c["close"] for c in candles]
    highs  = [c["high"] for c in candles]
    lows   = [c["low"] for c in candles]

    ef = p["indicators"]["ema_fast"]; es = p["indicators"]["ema_slow"]
    rlen = p["indicators"]["rsi_len"]; alen = p["indicators"]["atr_len"]

    ema_f = ema(closes, ef)
    ema_s = ema(closes, es)
    r = rsi(closes, rlen)
    a = atr(highs, lows, closes, alen)

    return {"ema_fast": ema_f, "ema_slow": ema_s, "rsi": r, "atr": a}
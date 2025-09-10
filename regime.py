from typing import Tuple, Dict
from indicators import OnlineIndicators
from config import Config

class MarketRegime:
    TRND_UP = "TRND_UP"
    TRND_DN = "TRND_DN"
    RNG = "RNG"
    IMP_UP = "IMP_UP"
    IMP_DN = "IMP_DN"

def classify_regime(ind: OnlineIndicators, cfg: Config) -> Tuple[str, Dict[str, float]]:
    if not ind.closes or ind.ema_fast is None or ind.ema_slow is None:
        return MarketRegime.RNG, {"score_rng": 0, "score_up": 0, "score_dn": 0, "score_imp_up": 0, "score_imp_dn": 0}

    close = ind.closes[-1]
    vwap = ind.vwap
    atrz = ind.atr_z() or 1.0

    score_up = 0.0
    score_dn = 0.0
    score_rng = 0.0
    score_imp_up = 0.0
    score_imp_dn = 0.0

    if ind.ema_fast > ind.ema_slow:
        score_up += 1.0
    elif ind.ema_fast < ind.ema_slow:
        score_dn += 1.0

    if vwap is not None:
        if close > vwap:
            score_up += 0.5
        if close < vwap:
            score_dn += 0.5

    if atrz > cfg.atr_mult:
        if ind.ema_fast > ind.ema_slow:
            score_up += 0.5
        elif ind.ema_fast < ind.ema_slow:
            score_dn += 0.5
    else:
        score_rng += 0.5

    ema_spread = abs(ind.ema_fast - ind.ema_slow) / close
    if ema_spread < 0.0007:
        score_rng += 1.0
    low, high = ind.channel()
    if low is not None and high is not None and low < close < high:
        score_rng += 0.5

    if len(ind.highs) >= 50:
        now_range = ind.highs[-1] - ind.lows[-1]
        base_ranges = [h - l for h, l in zip(list(ind.highs)[-50:], list(ind.lows)[-50:])]
        base_ranges_sorted = sorted(base_ranges)
        p90 = base_ranges_sorted[int(0.9 * (len(base_ranges_sorted) - 1))]
        if vwap is not None and now_range > p90:
            if close > max(vwap, high or close):
                score_imp_up += 1.0
            if close < min(vwap, low or close):
                score_imp_dn += 1.0

    choices = [
        (MarketRegime.TRND_UP, score_up),
        (MarketRegime.TRND_DN, score_dn),
        (MarketRegime.RNG, score_rng),
        (MarketRegime.IMP_UP, score_imp_up),
        (MarketRegime.IMP_DN, score_imp_dn),
    ]
    regime = max(choices, key=lambda x: x[1])[0]
    metrics = {
        "score_up": score_up, "score_dn": score_dn, "score_rng": score_rng,
        "score_imp_up": score_imp_up, "score_imp_dn": score_imp_dn,
        "ema_spread": ema_spread, "atr_z": atrz, "vwap": vwap or 0.0
    }
    return regime, metrics
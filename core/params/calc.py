import os

def build_params() -> dict:
    """
    Агрессивные дефолты («на грани»), но все можно переопределить через ENV.
    Разбивка по стратегиям.
    """
    def f(name, default):
        return type(default)(os.getenv(name, default))

    return {
        "global": {
            "ema_fast": 9,
            "ema_slow": 21,
            "rsi_period": 7,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "atr_mult": 1.0,
            "tp_mult": 2.0,
        },
        "Momentum": {
            "min_body_frac": 0.5,        # тело >= 50% диапазона бара
            "min_rng_frac": 0.00015,     # 0.015% цены
            "min_rng_abs": 3.0,          # $3 минимум
            "debug_extreme": True,       # можно выключить позже
        },
        "Reversal": {
            "rsi_overbought": 72,
            "rsi_oversold": 28,
            "confirm_pattern": "pin|engulf",
        },
        "Breakout": {
            "range_bars": 30,
            "min_range": 0.0015,         # 0.15%
            "volume_filter": 0.8,        # 80% медианы — допускаем узкий диапазон
        },
        "Orderbook Density": {
            "orderbook_depth": 50,
            "top_n_levels": 5,
            "imbalance_threshold": 1.5,
        },
        "Knife": {
            "impulse_atr_mult": 1.4,   # было 2.0 — ниже порог импульса по ATR
            "volume_spike_mult": 1.5,  # было 2.0 — чаще допускаем вход по объёму
            "cooldown_bars": 0,        # было 1 — можно входить подряд без паузы
        }
    }
import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    def __init__(self):
        self.bybit_api_key_test: str = os.getenv("BYBIT_API_KEY_TEST", "")
        self.bybit_api_secret_test: str = os.getenv("BYBIT_SECRET_KEY_TEST", "")
        self.bybit_api_key_main: str = os.getenv("BYBIT_API_KEY_MAIN", "")
        self.bybit_api_secret_main: str = os.getenv("BYBIT_SECRET_KEY_MAIN", "")
        self.category: str = os.getenv("CATEGORY", "linear")
        self.symbol: str = os.getenv("SYMBOL", "BTCUSDT")
        self.qty: str = os.getenv("QTY", "0.01")
        self.testnet = os.getenv("BYBIT_TESTNET", "true")
        self.mult_qty: float = float(os.getenv("MULT_QTY", "1"))
        self.sl: float = float(os.getenv("SL", "0.03"))
        self.tp: float = float(os.getenv("TP", "0.1"))
        self.interval: str = os.getenv("INTERVAL", "1")
        self.atr_period: int = int(os.getenv("ATR_PERIOD", "14"))
        self.trend_limit: int = int(os.getenv("TREND_LIMIT", "100"))

        # ---- Strategy tuning (moved into class, env-backed) ----
        # Обнаружение импульса
        self.impulse_vol_surge: float = float(os.getenv("IMPULSE_VOL_SURGE", "0.3"))           # +30% объёма для импульса
        self.impulse_price_change: float = float(os.getenv("IMPULSE_PRICE_CHANGE", "0.15"))    # 0.15% движения цены
        self.impulse_body_strength: float = float(os.getenv("IMPULSE_BODY_STRENGTH", "0.6"))   # 60% тела свечи
        self.impulse_force_threshold: float = float(os.getenv("IMPULSE_FORCE_THRESHOLD", "3.0")) # Порог для force TP

        # Trailing stop
        self.trail_profit_atr: float = float(os.getenv("TRAIL_PROFIT_ATR", "1.0"))             # Начало трейла при 0.8 ATR прибыли
        self.trail_distance_atr: float = float(os.getenv("TRAIL_DISTANCE_ATR", "0.8"))         # 0.5 ATR от цены

        # RSI фильтры (ужесточены)
        self.rsi_extreme_high: float = float(os.getenv("RSI_EXTREME_HIGH", "80.0"))            # Блок BUY
        self.rsi_extreme_low: float = float(os.getenv("RSI_EXTREME_LOW", "20.0"))              # Блок SELL
        self.rsi_emergency_high: float = float(os.getenv("RSI_EMERGENCY_HIGH", "96.0"))        # Экстренное закрытие
        self.rsi_emergency_low: float = float(os.getenv("RSI_EMERGENCY_LOW", "4.0"))          # Экстренное закрытие

        # Динамический SL
        self.atr_sl_mult: float = float(os.getenv("ATR_SL_MULT", "1.5"))                        # Множитель ATR для стопа
        self.max_risk_pct: float = float(os.getenv("MAX_RISK_PCT", "2.0"))                      # Макс риск 2%

        # Минимальное подтверждение
        self.trend_conf_min: float = float(os.getenv("TREND_CONF_MIN", "0.4"))                  # Низкий порог (т.к. импульс уже найден)

        # ---- ДЕТЕКЦИЯ ВЗРЫВНОЙ СВЕЧИ ----
        self.candle_min_vol_ratio: float = float(os.getenv("CANDLE_MIN_VOL_RATIO", "1.3"))         # +30% объёма
        self.candle_min_price_move: float = float(os.getenv("CANDLE_MIN_PRICE_MOVE", "0.08"))      # 0.08% движения (очень мягко)
        self.candle_min_body_ratio: float = float(os.getenv("CANDLE_MIN_BODY_RATIO", "0.5"))       # 50% тела
        self.candle_max_wick_ratio: float = float(os.getenv("CANDLE_MAX_WICK_RATIO", "0.4"))       # макс 40% фитиля
        self.candle_min_range_pct: float = float(os.getenv("CANDLE_MIN_RANGE_PCT", "0.1"))         # 0.1% диапазона

        # ---- МИНИМАЛЬНАЯ СИЛА ДЛЯ ВХОДА ----
        self.min_signal_strength: float = float(os.getenv("MIN_SIGNAL_STRENGTH", "3.0"))           # Сила ≥3 для входа

        # ---- РАЗВОРОТ ПОЗИЦИИ ----
        self.reverse_strength_threshold: float = float(os.getenv("REVERSE_STRENGTH_THRESHOLD", "5.0"))  # Разворот при силе ≥6

        # ---- RSI (ТОЛЬКО КРИТИЧЕСКИЕ УРОВНИ) ----
        self.rsi_critical_high: float = float(os.getenv("RSI_CRITICAL_HIGH", "90.0"))          # Блок BUY
        self.rsi_critical_low: float = float(os.getenv("RSI_CRITICAL_LOW", "10.0"))            # Блок SELL
        self.rsi_emergency_close_high: float = float(os.getenv("RSI_EMERGENCY_CLOSE_HIGH", "95.0"))   # Экстренное закрытие BUY
        self.rsi_emergency_close_low: float = float(os.getenv("RSI_EMERGENCY_CLOSE_LOW", "5.0"))     # Экстренное закрытие SELL

        # ---- FORCE TP ----
        self.force_tp_strength: float = float(os.getenv("FORCE_TP_STRENGTH", "6.0"))            # Force TP при силе ≥8

        # === ДЕТЕКТОРЫ ===
        self.vol_spike_threshold: float = float(os.getenv("VOL_SPIKE_THRESHOLD", "1.0"))              # +40% объёма для сигнала
        self.momentum_threshold: float = float(os.getenv("MOMENTUM_THRESHOLD", "0.06"))               # 0.12% за 3 свечи
        self.volatility_expansion_threshold: float = float(os.getenv("VOLATILITY_EXPANSION_THRESHOLD", "1.25"))   # 1.5x рост диапазона

        # === ФИЛЬТРЫ ===
        self.min_confirming_signals: int = int(os.getenv("MIN_CONFIRMING_SIGNALS", "1"))              # Минимум 2 совпадающих сигнала
        self.min_total_strength: float = float(os.getenv("MIN_TOTAL_STRENGTH", "3.0"))                # Минимальная суммарная сила

        # ---- КАНАЛ / RANGE TRADING ----
        # Параметры канала и фильтров для торговли от границ
        self.channel_period: int = int(os.getenv("CHANNEL_PERIOD", "20"))                 # Период для расчёта границ
        self.bb_multiplier: float = float(os.getenv("BB_MULTIPLIER", "2.0"))             # Множитель для Bollinger

        # ФИЛЬТР НА ФЛЭТ (критично!)
        self.adx_period: int = int(os.getenv("ADX_PERIOD", "14"))
        self.adx_max_trend: float = float(os.getenv("ADX_MAX_TREND", "25.0"))            # Выше = тренд (не торгуем)
        self.min_channel_width_pct: float = float(os.getenv("MIN_CHANNEL_WIDTH_PCT", "0.3"))  # Минимальная ширина канала, %
        self.max_channel_width_pct: float = float(os.getenv("MAX_CHANNEL_WIDTH_PCT", "3.0"))  # Максимальная ширина канала, %

        # ОТСТУПЫ ОТ ГРАНИЦ
        self.entry_offset_pct: float = float(os.getenv("ENTRY_OFFSET_PCT", "5.0"))       # 5% внутрь от границы
        self.tp_offset_pct: float = float(os.getenv("TP_OFFSET_PCT", "10.0"))            # 10% не доходя до противоположной

        # РЕЖИМ
        self.dual_limit_orders: bool = str(os.getenv("DUAL_LIMIT_ORDERS", "false")).lower() in ("1", "true", "yes", "y")
        self.tick_size: float = float(os.getenv("TICK_SIZE", "0.1"))                      # Размер тика для округления

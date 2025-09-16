
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


import os

# ==== БАЗОВЫЕ НАСТРОЙКИ ====
HOST_ROLE = os.getenv("HOST_ROLE", "local")  # local | server
BYBIT_TESTNET = os.getenv("BYBIT_TESTNET", "true").lower() == "true"
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "").strip()
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "").strip()
BYBIT_RECV_WINDOW = int(os.getenv("BYBIT_RECV_WINDOW", "5000"))
BYBIT_SYMBOL = os.getenv("BYBIT_SYMBOL", "BTCUSDT")
BYBIT_LEVERAGE = int(os.getenv("BYBIT_LEVERAGE", "10"))
BYBIT_VERIFY_SSL = os.getenv("BYBIT_VERIFY_SSL", "true").lower() == "true"

# ==== OPENAI ====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ==== TELEGRAM ====
TG_TOKEN_LOCAL = os.getenv("TG_TOKEN_LOCAL", "").strip()
TG_TOKEN_SERVER = os.getenv("TG_TOKEN_SERVER", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()  # один и тот же для обоих окружений

def get_tg_token() -> str:
    return TG_TOKEN_SERVER if HOST_ROLE == "server" else TG_TOKEN_LOCAL

# ==== ПУТИ ====
DATA_DIR = os.getenv("DATA_DIR", "./data")
PARAMS_PATH = os.path.join(DATA_DIR, "params.json")
DUMP_DIR = os.path.join(DATA_DIR, "hourly_dumps")

# ==== СИМВОЛ И КАТЕГОРИЯ ====
CATEGORY = "linear"  # USDT Perp

# ==== ДЕФОЛТ ПАРАМЕТРЫ (если нет params.json) ====
DEFAULT_PARAMS = {
    "risk": 0.02,
    "size_usdt": 50,
    "filters": {"min_spread_ticks": 2, "max_slippage_ticks": 10},
    "indicators": {
        "ema_fast": 9, "ema_slow": 50,
        "rsi_len": 7, "rsi_overbought": 68, "rsi_oversold": 32,
        "atr_len": 14,
        "virtual_tp_atr": 0.35, "virtual_sl_atr": 0.60,
        "tp_widen_mult": 1.2, "sl_widen_mult": 1.2
    },
    "execution": {"order_type": "Market", "tif": "IOC", "tpsl_mode": "Full", "trigger_by": "LastPrice"}
}
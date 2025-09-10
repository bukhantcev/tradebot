import os
from dataclasses import dataclass
from dotenv import load_dotenv

def _get_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

load_dotenv()

@dataclass
class Config:
    # Telegram
    tg_token: str = os.getenv("TG_BOT_TOKEN", "")
    tg_chat_id: int = int(os.getenv("TG_CHAT_ID", "0"))

    # Bybit
    bybit_key: str = os.getenv("BYBIT_API_KEY", "")
    bybit_secret: str = os.getenv("BYBIT_API_SECRET", "")
    bybit_env: str = os.getenv("BYBIT_ENV", "testnet")  # testnet | live
    symbol: str = os.getenv("SYMBOL", "BTCUSDT")
    category: str = os.getenv("CATEGORY", "linear")     # linear | inverse
    leverage: int = int(os.getenv("LEVERAGE", "10"))

    # Risk/Strategy
    risk_pct: float = float(os.getenv("RISK_PCT_DEPOT", "0.005"))
    atr_len: int = int(os.getenv("ATR_LEN", "14"))
    atr_mult: float = float(os.getenv("ATR_MULT", "1.5"))
    tp_r_mult: float = float(os.getenv("TP_R_MULT", "1.0"))
    # Bracket (TP/SL) settings
    tp_pct: float = float(os.getenv("TP_PCT", "0.003"))     # 0.3% = 0.003
    sl_pct: float = float(os.getenv("SL_PCT", "0.002"))     # 0.2% = 0.002

    # Safety timeout for open position (seconds)
    max_hold_sec: int = int(os.getenv("MAX_HOLD_SEC", "900"))
    use_trailing: bool = _get_bool("USE_TRAILING", True)
    trailing_activation: float = float(os.getenv("TRAILING_ACTIVATION", "0.003"))
    trailing_distance: float = float(os.getenv("TRAILING_DISTANCE", "0.002"))

    ema_fast: int = int(os.getenv("EMA_FAST", "9"))
    ema_slow: int = int(os.getenv("EMA_SLOW", "21"))
    channel_lookback: int = int(os.getenv("CHANNEL_LOOKBACK", "30"))
    cooldown_after_2_losses_min: int = int(os.getenv("COOLDOWN_AFTER_2_LOSSES", "30"))

    # Engine
    ws_timeout: int = int(os.getenv("WS_TIMEOUT_SEC", "15"))
    http_timeout: int = int(os.getenv("HTTP_TIMEOUT_SEC", "15"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "5"))
    ws_verify_ssl: bool = _get_bool("WS_VERIFY_SSL", True)
    ws_use_certifi: bool = _get_bool("WS_USE_CERTIFI", True)

    # Logging flags
    log_http: bool = _get_bool("LOG_HTTP", True)
    log_http_bodies: bool = _get_bool("LOG_HTTP_BODIES", False)
    log_ws_raw: bool = _get_bool("LOG_WS_RAW", False)
    log_signals: bool = _get_bool("LOG_SIGNALS", True)

    @property
    def ws_public_url(self) -> str:
        base = "wss://stream-testnet.bybit.com" if self.bybit_env == "testnet" else "wss://stream.bybit.com"
        part = "linear" if self.category == "linear" else "inverse"
        return f"{base}/v5/public/{part}"

    @property
    def rest_base(self) -> str:
        return "https://api-testnet.bybit.com" if self.bybit_env == "testnet" else "https://api.bybit.com"

CFG = Config()
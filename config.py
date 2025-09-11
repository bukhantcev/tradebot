import os
from dotenv import load_dotenv
from log import log

load_dotenv()

class Config:
    # Telegram
    host_role: str = os.getenv("HOST_ROLE", "local").lower()
    telegram_token_local: str = os.getenv("TELEGRAM_TOKEN_LOCAL", "")
    telegram_token_server: str = os.getenv("TELEGRAM_TOKEN_SERVER", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # Bybit
    bybit_api_key: str = os.getenv("BYBIT_API_KEY", "")
    bybit_api_secret: str = os.getenv("BYBIT_API_SECRET", "")
    bybit_base_url: str = os.getenv("BYBIT_BASE_URL", "https://api-testnet.bybit.com")
    bybit_category: str = os.getenv("BYBIT_CATEGORY", "linear").lower()
    symbol: str = os.getenv("SYMBOL", "BTCUSDT")
    leverage: float = float(os.getenv("LEVERAGE", 5))
    min_qty: float = float(os.getenv("MIN_QTY", 0.001))

    # S/R и SL
    sr_5m_limit: int = int(os.getenv("SR_5M_LIMIT", 200))
    sr_pivot_left: int = int(os.getenv("SR_PIVOT_LEFT", 3))
    sr_pivot_right: int = int(os.getenv("SR_PIVOT_RIGHT", 3))
    sl_buffer_ticks: int = int(os.getenv("SL_BUFFER_TICKS", 2))

    # Тайминги
    loop_delay_sec: int = int(os.getenv("LOOP_DELAY_SEC", 5))
    sr_refresh_grace_sec: int = int(os.getenv("SR_REFRESH_GRACE_SEC", 2))

    # SSL
    bybit_verify_ssl: bool = os.getenv("BYBIT_VERIFY_SSL", "true").lower() == "true"
    bybit_ca_bundle: str = os.getenv("BYBIT_CA_BUNDLE", "")

    # Прочее
    use_trailing: bool = os.getenv("USE_TRAILING", "false").lower() == "true"
    trailing_activation: float = float(os.getenv("TRAILING_ACTIVATION", 0.0))
    trailing_distance: float = float(os.getenv("TRAILING_DISTANCE", 0.0))

    @property
    def telegram_token(self) -> str:
        return (
            self.telegram_token_local
            if self.host_role == "local"
            else self.telegram_token_server
        )

CFG = Config()

# Startup config logging
try:
    log.info("=== Scalper bot starting ===")
    log.info(
        f"[META] start -> url={CFG.bybit_base_url} "
        f"category={CFG.bybit_category} symbol={CFG.symbol} "
        f"verify_ssl={CFG.bybit_verify_ssl} "
        f"ca_bundle={'set' if CFG.bybit_ca_bundle else 'none'} "
        f"host_role={CFG.host_role}"
    )
    if not CFG.bybit_verify_ssl:
        log.warning("[HTTP] SSL verification is DISABLED")
    elif CFG.bybit_ca_bundle:
        log.info(f"[HTTP] Using custom CA bundle: {CFG.bybit_ca_bundle}")
except Exception:
    # Logging should never break config import
    pass
# config.py
import os
from dataclasses import dataclass

# .env не обязателен; если есть — подхватим тихо
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


def _as_bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass(frozen=True)
class Config:
    # ===== Runtime / Telegram =====
    host_role: str = os.getenv("HOST_ROLE", "local")  # local | server
    tg_token_local: str = os.getenv("TG_TOKEN_LOCAL", "")
    tg_token_server: str = os.getenv("TG_TOKEN_SERVER", "")
    tg_admin_chat_id: int = int(os.getenv("TG_ADMIN_CHAT_ID", "0"))

    # ===== Bybit creds / endpoints =====
    bybit_api_key: str = os.getenv("BYBIT_API_KEY", "")
    bybit_api_secret: str = os.getenv("BYBIT_API_SECRET", "")
    # если явно задан BYBIT_BASE_URL — используем его; иначе переключаемся по BYBIT_TESTNET
    bybit_base_url_env: str = os.getenv("BYBIT_BASE_URL", "")
    bybit_use_testnet: bool = _as_bool(os.getenv("BYBIT_TESTNET"), True)
    bybit_verify_ssl: bool = _as_bool(os.getenv("BYBIT_VERIFY_SSL"), False)  # на Mac часто нужны кастомные сертификаты
    category: str = os.getenv("BYBIT_CATEGORY", "linear")
    symbol: str = os.getenv("SYMBOL", "BTCUSDT")
    leverage: float = float(os.getenv("LEVERAGE", "5"))
    order_qty: float = float(os.getenv("ORDER_QTY", "0.001"))

    # ===== DB =====
    db_path: str = os.getenv("DB_PATH", "./market.db")

    # ===== Параметры стратегии (S/R раздельно) =====
    # --- 5м уровни: только для биржевого SL (страховка)
    sr5_left: int = int(os.getenv("SR5_LEFT", "3"))
    sr5_right: int = int(os.getenv("SR5_RIGHT", "3"))

    # --- 1м уровни: для генерации сигналов/входов-выходов
    sr1_left: int = int(os.getenv("SR1_LEFT", "1"))
    sr1_right: int = int(os.getenv("SR1_RIGHT", "1"))

    # Допуски для 1м (в процентах цены — например 0.05 = 0.05%)
    sr1_touch_pct: float = float(os.getenv("SR1_TOUCH_PCT", "0.02"))   # «касание» уровня
    sr1_break_pct: float = float(os.getenv("SR1_BREAK_PCT", "0.05"))   # минимальный пробой

    # Буфер для выставления биржевого SL от S/R (в тиках инструмента)
    sl_buffer_ticks: int = int(os.getenv("SL_BUFFER_TICKS", "2"))

    # ===== Тайминги/цикл =====
    loop_sleep_sec: float = float(os.getenv("LOOP_SLEEP_SEC", "1.0"))   # основной цикл стратегии
    kline_refresh_1m_sec: float = float(os.getenv("KLINE_REFRESH_1M_SEC", "10"))
    kline_refresh_5m_sec: float = float(os.getenv("KLINE_REFRESH_5M_SEC", "60"))

    # ===== Производные (заполняем ниже) =====
    tg_token: str = ""           # выбирается по host_role
    bybit_base_url: str = ""     # выбирается по тестнету/переменной
    base_url: str = ""           # алиас для обратной совместимости


CFG = Config()

# --- resolve Telegram token по роли ---
tg = CFG.tg_token_server if CFG.host_role.lower() == "server" else CFG.tg_token_local
object.__setattr__(CFG, "tg_token", tg)

# --- resolve Bybit base url ---
if CFG.bybit_base_url_env:
    base = CFG.bybit_base_url_env
else:
    base = "https://api-testnet.bybit.com" if CFG.bybit_use_testnet else "https://api.bybit.com"
object.__setattr__(CFG, "bybit_base_url", base)
object.__setattr__(CFG, "base_url", base)

# src/config.py
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# Три “топовых” тикера для быстрого запуска
BEST_TICKERS = ["VTBR", "GAZP", "SBER"]  # можно поменять под себя

# Фолбэк: тикер (SECID) → FIGI (для стабильной работы даже при багающем справочнике)
FIGI_FALLBACK = {
    "SBER": "BBG004730N88",
    "GAZP": "BBG004730RP0",
    "VTBR": "BBG004731032",
    "GMKN": "BBG0047315D0",
}

# Пресеты настроек через кнопки
RISK_PRESETS = [0.005, 0.01, 0.02]           # 0.5%, 1%, 2%
INTERVAL_PRESETS = ["1m", "5m"]              # кнопки для интервала
PAYIN_PRESETS = [10_000, 50_000, 100_000]    # RUB

@dataclass
class Settings:
    tg_token: str
    sandbox_token: str
    sandbox_init_rub: float
    figi_default: str

def get_settings() -> Settings:
    tg = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    sbx = os.getenv("SANDBOX_TOKEN", "").strip()
    if not tg:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан в .env")
    if not sbx:
        raise RuntimeError("SANDBOX_TOKEN не задан в .env")

    return Settings(
        tg_token=tg,
        sandbox_token=sbx,
        sandbox_init_rub=float(os.getenv("SANDBOX_INIT_RUB", "100000")),
        figi_default=os.getenv("FIGI_DEFAULT", "BBG004730N88"),
    )

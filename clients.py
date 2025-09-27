import os

# ---- Optional SSL relax (env BYBIT_VERIFY_SSL=false) ----
if os.getenv("BYBIT_VERIFY_SSL", "true").lower() == "false":
    os.environ["PYTHONHTTPSVERIFY"] = "0"

from dotenv import load_dotenv
from pybit.unified_trading import HTTP
from aiogram import Bot, Dispatcher, F


import openai


# ------------------ .env & logging ------------------
load_dotenv()




def make_bybit() -> HTTP:
    testnet = os.getenv("BYBIT_TESTNET", "true").lower() == "true"
    api_key = os.getenv("BYBIT_API_KEY_TEST" if testnet else "BYBIT_API_KEY_MAIN")
    api_secret = os.getenv("BYBIT_SECRET_KEY_TEST" if testnet else "BYBIT_SECRET_KEY_MAIN")
    # pybit HTTP takes testnet=bool and uses requests under the hood.
    return HTTP(testnet=testnet, api_key=api_key, api_secret=api_secret)

BYBIT = make_bybit()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

BOT = Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None
DP = Dispatcher()

# OpenAI
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ------------------ Config from .env ------------------
LOT_SIZE_USDT = float(os.getenv("LOT_SIZE_USDT", "50"))
LEVERAGE = int(os.getenv("LEVERAGE", "5"))

# Flat mode config
FLAT_ENTRY_TICKS = int(os.getenv("FLAT_ENTRY_TICKS", "6"))  # от экстремума внутрь
# SL in ticks for both modes (if AI doesn’t override)
SL_TICKS = int(os.getenv("SL_TICKS", "6000"))                 # настраивается в .env

# Take-profit rule for flat:
# "на 2 тика ниже/выше противоположного от уровня открытия позиции края тела прошлой минуты"
TP_BODY_OFFSET_TICKS = int(os.getenv("TP_BODY_OFFSET_TICKS", "2"))  # TP offset from body edge (ticks)
MARKET_BAND_EXTRA_TICKS = int(os.getenv("MARKET_BAND_EXTRA_TICKS", "4"))  # how many ticks deeper to push IOC Limit fallback

# Interval and bootstrap hours
POLL_TICK_MS = int(os.getenv("POLL_TICK_MS", "1000"))  # per-tick loop delay (ms)
BOOTSTRAP_HOURS = int(os.getenv("BOOTSTRAP_HOURS", "5"))
AI_POLL_SEC = int(os.getenv("AI_POLL_SEC", "60"))      # запасной лимит (основной триггер — закрытие свечи)

# Retry settings
RETRY_ATTEMPTS = 10
RETRY_DELAY_SEC = 1

TREND_SL_MULT = float(os.getenv("TREND_SL_MULT", "5.0"))

MIN_TP_TICKS = int(os.getenv("MIN_TP_TICKS", "2500"))  # минимум для TP в тиках

TREND_CONFIRM_BARS = int(os.getenv("TREND_CONFIRM_BARS", "3"))   # сколько подряд минутных сигналов TREND одной стороны нужно
REVERSE_HYSTERESIS_SEC = int(os.getenv("REVERSE_HYSTERESIS_SEC", "10"))  # пауза после flip (чтоб не дёргаться)

REENTER_AFTER_SL_SEC = int(os.getenv("REENTER_AFTER_SL_SEC", "10"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
USE_LOCAL_DECIDER = os.getenv("USE_LOCAL_DECIDER", "true").lower() == "true"  # default: local

FLAT_CHANNEL_BARS = int(os.getenv("FLAT_CHANNEL_BARS", "20"))
SL_FLAT_CHANNEL_PCT = float(os.getenv("SL_FLAT_CHANNEL_PCT", "2.5"))
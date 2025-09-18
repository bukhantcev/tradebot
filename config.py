import os
from dotenv import load_dotenv

load_dotenv()

# --- General ---
HOST_ROLE = os.getenv("HOST_ROLE", "local")  # local | server
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
FILE_LOG_LEVEL = os.getenv("FILE_LOG_LEVEL", "DEBUG")

# --- Bybit ---
BYBIT_ENV = os.getenv("BYBIT_ENV", "testnet")  # real | testnet
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
LEVERAGE = int(os.getenv("LEVERAGE", "1"))
RISK_PCT = float(os.getenv("RISK_PCT", "1.0"))  # % от equity
LLM_BUDGET_DAILY_USD = float(os.getenv("LLM_BUDGET_DAILY_USD", "3"))

BYBIT_KEY_TESTNET = os.getenv("BYBIT_KEY_TESTNET")
BYBIT_SECRET_TESTNET = os.getenv("BYBIT_SECRET_TESTNET")
BYBIT_KEY_REAL = os.getenv("BYBIT_KEY_REAL")
BYBIT_SECRET_REAL = os.getenv("BYBIT_SECRET_REAL")

# --- OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Telegram ---
TELEGRAM_TOKEN_LOCAL = os.getenv("TELEGRAM_TOKEN_LOCAL")
TELEGRAM_TOKEN_SERVER = os.getenv("TELEGRAM_TOKEN_SERVER")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# --- Data / Storage ---
DB_PATH = os.getenv("DB_PATH", "./data/trades.sqlite")
PARQUET_DIR = os.getenv("PARQUET_DIR", "./data/parquet")

# --- Helpers ---
def get_bybit_keys():
    if BYBIT_ENV == "real":
        return BYBIT_KEY_REAL, BYBIT_SECRET_REAL
    return BYBIT_KEY_TESTNET, BYBIT_SECRET_TESTNET
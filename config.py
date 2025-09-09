import os
import logging.config
from dotenv import load_dotenv

def load_config():
    load_dotenv()
    return {
        "telegram_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
        "openai_key": os.getenv("OPENAI_API_KEY", ""),
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "bybit_key": os.getenv("BYBIT_API_KEY", ""),
        "bybit_secret": os.getenv("BYBIT_API_SECRET", ""),
        "testnet": os.getenv("BYBIT_TESTNET", "true").lower() == "true",
        "symbol": os.getenv("SYMBOL", "BTCUSDT"),
        "account_type": os.getenv("ACCOUNT_TYPE", "UNIFIED"),
        "category": os.getenv("CATEGORY", "linear"),
        "ws_kline_interval": os.getenv("WS_KLINE_INTERVAL", "1"),
        "base_order_usdt": float(os.getenv("BASE_ORDER_USDT", "10")),
        "max_loss_usdt": float(os.getenv("MAX_LOSS_USDT", "5")),
        "no_trade_timeout_sec": int(os.getenv("NO_TRADE_TIMEOUT_SEC", "300")),
        "loss_streak_requery": int(os.getenv("LOSS_STREAK_REQUERY", "2")),
        "snap_1m": int(os.getenv("SNAP_CANDLES_1M", "200")),
        "snap_5m": int(os.getenv("SNAP_CANDLES_5M", "200")),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "log_file": os.getenv("LOG_FILE", "bot.log"),
    }

def configure_logging():
    import logging
    import logging.config
    from pathlib import Path
    if Path("logging.ini").exists():
        logging.config.fileConfig("logging.ini")
    else:
        logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                            format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s")
# logger.py
import os
import logging

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("tradebot.log", encoding="utf-8"),
    ],
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
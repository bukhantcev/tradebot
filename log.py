import logging
import sys
from config import LOG_LEVEL, FILE_LOG_LEVEL

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # корневой уровень = DEBUG

    # Console handler (INFO+)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"
    ))
    logger.addHandler(ch)

    # File handler (DEBUG)
    fh = logging.FileHandler("bot_debug.log", encoding="utf-8")
    fh.setLevel(getattr(logging, FILE_LOG_LEVEL.upper(), logging.DEBUG))
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"
    ))
    logger.addHandler(fh)

    # Silence overly noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)

    return logger

logger = setup_logging()

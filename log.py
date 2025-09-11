import logging
import sys

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("scalper")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger

log = setup_logger()
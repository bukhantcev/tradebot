import logging
from logging import Logger
from logging.handlers import RotatingFileHandler
from typing import Optional
from config import load_config

_cfg = load_config()
_LOG_FILE = _cfg.get("log_file", "bot.log")

# Create a single, shared root logger configuration exactly once.
def _ensure_logging() -> Logger:
    root = logging.getLogger()  # root logger
    if getattr(root, "_tradebot_handlers_installed", False):
        return root

    root.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"
    )

    # Console
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # File (rotating to avoid huge files)
    fh = RotatingFileHandler(_LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Mark as installed to avoid duplicate handlers on re-import
    setattr(root, "_tradebot_handlers_installed", True)
    return root

# Ensure configuration on import
_ensure_logging()

# Expose a module-level logger for convenience
logger: Logger = logging.getLogger("tradebot")

import logging
import sys
from config import CFG

# Ensure 'channel' is always present to avoid KeyError in format strings
class _ChannelDefault(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "channel"):
            record.channel = ""
        return True

# ------- Colorized formatter (TTY only) -------
class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[37m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[41m",
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        msg = super().format(record)
        return f"{color}{msg}{self.RESET}"


def _setup_logger():
    logger = logging.getLogger("bot")

    # Fallback if CFG.log_level is missing
    level_name = getattr(CFG, "log_level", "INFO")
    if isinstance(level_name, str):
        level = getattr(logging, level_name.upper(), logging.INFO)
    else:
        level = int(level_name) if isinstance(level_name, int) else logging.INFO
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)

    # base format includes optional channel
    base_fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    channel_fmt = "[%(asctime)s] %(levelname)s: [%(channel)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    if sys.stdout.isatty():
        fmt = ColorFormatter(channel_fmt, datefmt)
        fallback_fmt = ColorFormatter(base_fmt, datefmt)
    else:
        fmt = logging.Formatter(channel_fmt, datefmt)
        fallback_fmt = logging.Formatter(base_fmt, datefmt)

    handler.setFormatter(fmt)
    handler.addFilter(_ChannelDefault())
    logger.addHandler(handler)
    logger.propagate = False

    # One-time SSL warning
    if not getattr(CFG, "bybit_verify_ssl", True):
        # Log with base logger; _ChannelDefault ensures 'channel' exists
        logger.warning("[HTTP] SSL verification is DISABLED (CFG.bybit_verify_ssl=false)")

    return logger


# Base logger
_base_log = _setup_logger()


class _ChannelAdapter(logging.LoggerAdapter):
    """LoggerAdapter that injects a static [channel] tag into records."""

    def process(self, msg, kwargs):
        extra = kwargs.setdefault("extra", {})
        # keep existing channel if provided explicitly
        extra.setdefault("channel", self.extra.get("channel", ""))
        return msg, kwargs


def chan(name: str) -> logging.LoggerAdapter:
    """Return a channelized logger that prefixes messages with [name]."""
    return _ChannelAdapter(_base_log, {"channel": name})


# ---- Predefined channels for convenience ----
log = _base_log                 # backwards-compatible plain logger
log_sr5 = chan("SR-5m")         # support/resistance on 5m
log_sr1 = chan("SR-1m")         # support/resistance on 1m (derived each 1m close)
log_regime1 = chan("REGIME-1m") # regime computed on 1m
log_sig1 = chan("SIG-1m")       # trading signal on 1m close
log_sl5 = chan("SL-5m")         # exchange stop set using 5m levels
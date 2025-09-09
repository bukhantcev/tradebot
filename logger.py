import logging
from config import load_config

cfg = load_config()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
    force=True
)

logger = logging.getLogger(__name__)

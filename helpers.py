import os

from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN


from clients import TELEGRAM_CHAT_ID, BOT
from logger import log


# ---- Optional SSL relax (env BYBIT_VERIFY_SSL=false) ----
if os.getenv("BYBIT_VERIFY_SSL", "true").lower() == "false":
    os.environ["PYTHONHTTPSVERIFY"] = "0"


from aiogram.enums import ParseMode
from aiogram.utils.chat_action import ChatActionSender
from aiogram.exceptions import TelegramBadRequest





def q(price: float, step: float) -> float:
    return float(Decimal(str(price)).quantize(Decimal(str(step)), rounding=ROUND_DOWN))

def normalize_price(price: float, tick: float) -> float:
    return q(price, tick)

def normalize_qty(qty: float, step: float, min_qty: float) -> float:
    n = q(qty, step)
    return max(n, min_qty)

def ts_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

async def tg_send(text: str):
    if not BOT or not TELEGRAM_CHAT_ID:
        log.debug("[TG] skip (no token/chat id): %s", text)
        return
    try:
        async with ChatActionSender.typing(bot=BOT, chat_id=TELEGRAM_CHAT_ID):
            await BOT.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    except Exception as e:
        log.error("[TG] send error: %s", e)

# --- Safe message edit helper ---
async def safe_edit(message, text: str, markup=None, parse_mode=None):
    """Edit message text safely: ignore 'message is not modified' errors."""
    try:
        await message.edit_text(text, reply_markup=markup, parse_mode=parse_mode)
    except TelegramBadRequest as e:
        if "message is not modified" in str(e).lower():
            return
        raise

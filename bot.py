# bot.py — весь Telegram на aiogram v3
import os
import logging
import asyncio
from typing import Optional, Callable, Awaitable

from dotenv import load_dotenv
load_dotenv()

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.types import (
    Message,
    ReplyKeyboardMarkup,
    KeyboardButton,
)
from aiogram.filters import CommandStart, Command

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
)
log = logging.getLogger("tg")

TG_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# единый синглтон
_bot_singleton = None

def _menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Старт"), KeyboardButton(text="Стоп")],
            [KeyboardButton(text="Баланс"), KeyboardButton(text="Открытые позиции")],
            [KeyboardButton(text="Статистика")],
        ],
        resize_keyboard=True
    )

class AiogramBot:
    def __init__(self):
        if not TG_TOKEN:
            log.warning("[TG] token missing")
        self.bot = Bot(
            TG_TOKEN,
            default=DefaultBotProperties(parse_mode="HTML")
        )
        self.dp = Dispatcher()

        # callbacks (установит main.py)
        self.on_start_cb: Optional[Callable[[], Awaitable[None]]] = None
        self.on_stop_cb: Optional[Callable[[], Awaitable[None]]] = None
        self.on_balance_cb: Optional[Callable[[], Awaitable[None]]] = None
        self.on_positions_cb: Optional[Callable[[], Awaitable[None]]] = None
        self.on_stats_cb: Optional[Callable[[], Awaitable[None]]] = None

        # регистрация хендлеров
        self._register_handlers()

        log.debug("[TG] init token=%s chat_id=%s",
                  (TG_TOKEN[:6]+"…") if TG_TOKEN else "—",
                  TG_CHAT_ID or "—")

    def _register_handlers(self):
        dp = self.dp

        @dp.message(CommandStart())
        async def _cmd_start(msg: Message):
            await self.send_menu()
            if self.on_start_cb:
                await self.on_start_cb()

        @dp.message(Command("start"))
        async def _slash_start(msg: Message):
            await self.send_menu()
            if self.on_start_cb:
                await self.on_start_cb()

        @dp.message(F.text.casefold() == "старт")
        async def _ru_start(msg: Message):
            if self.on_start_cb:
                await self.on_start_cb()

        @dp.message(F.text.casefold() == "стоп")
        async def _ru_stop(msg: Message):
            if self.on_stop_cb:
                await self.on_stop_cb()

        @dp.message(F.text.casefold() == "баланс")
        async def _ru_balance(msg: Message):
            if self.on_balance_cb:
                await self.on_balance_cb()

        @dp.message(F.text.casefold() == "открытые позиции")
        async def _ru_positions(msg: Message):
            if self.on_positions_cb:
                await self.on_positions_cb()

        @dp.message(F.text.casefold() == "статистика")
        async def _ru_stats(msg: Message):
            if self.on_stats_cb:
                await self.on_stats_cb()

    def set_handlers(
        self,
        on_start: Optional[Callable[[], Awaitable[None]]] = None,
        on_stop: Optional[Callable[[], Awaitable[None]]] = None,
        on_balance: Optional[Callable[[], Awaitable[None]]] = None,
        on_positions: Optional[Callable[[], Awaitable[None]]] = None,
        on_stats: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        self.on_start_cb = on_start
        self.on_stop_cb = on_stop
        self.on_balance_cb = on_balance
        self.on_positions_cb = on_positions
        self.on_stats_cb = on_stats

    async def send_message(self, text: str, chat_id: Optional[str] = None):
        cid = chat_id or TG_CHAT_ID
        if not cid:
            log.warning("[TG] no chat_id; cannot send message")
            return
        await self.bot.send_message(cid, text)

    async def send_menu(self):
        if not TG_CHAT_ID:
            return
        await self.bot.send_message(TG_CHAT_ID, "Меню", reply_markup=_menu())

    async def run(self):
        await self.dp.start_polling(self.bot)

def get_bot() -> AiogramBot:
    global _bot_singleton
    if _bot_singleton is None:
        _bot_singleton = AiogramBot()
    return _bot_singleton
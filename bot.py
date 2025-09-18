# bot.py
import asyncio
import logging
from typing import Optional

from aiogram import Bot, Dispatcher, Router, F
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup

log = logging.getLogger("BOT")


def kb_main() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="💰 Баланс", callback_data="balance")],
        [
            InlineKeyboardButton(text="🟢 Старт", callback_data="start"),
            InlineKeyboardButton(text="🔴 Стоп", callback_data="stop"),
        ],
    ])


class TgBot:
    def __init__(self, token: str, chat_id: Optional[int] = None, trader=None):
        self.bot = Bot(token=token)
        self.dp = Dispatcher()
        self.router = Router()
        self.dp.include_router(self.router)

        self.trader = trader
        self.chat_id = chat_id
        self._running_flag = False

        self._register_handlers()

    def _register_handlers(self):
        @self.router.message(Command("start"))
        async def on_start(message: Message):
            if not self.chat_id:
                self.chat_id = message.chat.id
                log.info(f"[BOT] bind chat_id={self.chat_id}")
            await message.answer("Бот готов. Выбирай действие ниже:", reply_markup=kb_main())

        @self.router.message(Command("help"))
        async def on_help(message: Message):
            await message.answer("Команды:\n/start\n/help\n/ping\n/id")

        @self.router.message(Command("ping"))
        async def on_ping(message: Message):
            await message.answer("pong")

        @self.router.message(Command("id"))
        async def on_id(message: Message):
            await message.answer(f"chat_id: <code>{message.chat.id}</code>")

        @self.router.callback_query(F.data == "balance")
        async def on_balance(cb: CallbackQuery):
            if not self.trader:
                await cb.message.answer("Трейдер недоступен")
                await cb.answer()
                return
            eq = self.trader.refresh_equity()
            await cb.message.answer(f"💰 Баланс: <b>{eq:.2f} USDT</b>")
            await cb.answer()

        @self.router.callback_query(F.data == "start")
        async def on_start_trade(cb: CallbackQuery):
            self._running_flag = True
            await cb.message.answer("🟢 Торговля: <b>ON</b>")
            await cb.answer()

        @self.router.callback_query(F.data == "stop")
        async def on_stop_trade(cb: CallbackQuery):
            self._running_flag = False
            await cb.message.answer("🔴 Торговля: <b>OFF</b>")
            await cb.answer()

    @property
    def is_running(self) -> bool:
        return self._running_flag

    async def start_background(self):
        async def _poll():
            try:
                log.info("[BOT] polling start")
                try:
                    await self.bot.delete_webhook(drop_pending_updates=False)
                except Exception:
                    pass
                await self.dp.start_polling(self.bot)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                log.error(f"[BOT] polling error: {e}")

        asyncio.create_task(_poll(), name="tg-polling")

    async def announce_online(self):
        if not self.chat_id:
            return
        try:
            await self.bot.send_message(self.chat_id, "🟢 Bot online")
        except Exception as e:
            log.error(f"[BOT][ANNOUNCE][ERR] {e}")

    async def notify(self, text: str):
        if not self.chat_id:
            return
        try:
            await self.bot.send_message(self.chat_id, text)
        except Exception as e:
            log.error(f"[BOT][ERR] {e}")

    async def shutdown(self):
        try:
            await self.bot.session.close()
        except Exception:
            pass
        log.info("[BOT] shutdown")
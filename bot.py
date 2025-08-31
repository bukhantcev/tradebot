# bot.py
import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from src.config import get_settings
from src.handlers import router, sessions, Session
from src.keyboards import main_kb
from src.tinkoff_sandbox import Sandbox

logging.basicConfig(level=logging.INFO)

async def on_startup(bot: Bot):
    pass

async def on_shutdown(bot: Bot):
    for s in list(sessions.values()):
        if s.engine:
            s.engine.stop()

async def main():
    cfg = get_settings()
    bot = Bot(cfg.tg_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()
    dp.include_router(router)

    @dp.message()
    async def start_or_menu(msg: Message):
        if msg.chat.id not in sessions:
            sbx = Sandbox(cfg.sandbox_token, cfg.sandbox_init_rub)
            sbx.__enter__()  # держим открытой до выключения бота
            sessions[msg.chat.id] = Session(sbx, bot, asyncio.get_running_loop(), msg.chat.id)
        await msg.answer("Готово. Выбирай действие:", reply_markup=main_kb())

    try:
        await dp.start_polling(bot, on_startup=on_startup, on_shutdown=on_shutdown)
    finally:
        for s in list(sessions.values()):
            try:
                s.sbx.__exit__(None, None, None)
            except Exception:
                pass

if __name__ == "__main__":
    asyncio.run(main())

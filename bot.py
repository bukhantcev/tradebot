# bot.py
import asyncio
import logging
from typing import Optional

from aiogram import Bot, Dispatcher, Router, F
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)

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
    """
    Лёгкий Telegram-бот:
      - /start показывает кнопки
      - «Баланс» вызывает trader.refresh_equity()
      - «Старт/Стоп» — просто статусы (переключение оставляем за стратегией/лупом)
      - notify(text) — отправка кратких апдейтов (сигнал, LLM-решение, ордера)

    Инициализация:
        tg = TgBot(token=..., chat_id=..., trader=trader)
        await tg.start_background()   # запустить поллинг
        await tg.notify("бот готов")

    Примечание:
      chat_id можно не указывать — бот сам подхватит первый /start и кэшнет chat_id.
    """

    def __init__(self, token: str, chat_id: Optional[int] = None, trader=None):
        self.bot = Bot(token=token, parse_mode=ParseMode.HTML)
        self.dp = Dispatcher()
        self.router = Router()
        self.dp.include_router(self.router)

        self.trader = trader
        self.chat_id = chat_id
        self._running_flag = False  # визуальный статус для кнопок

        self._register_handlers()

    # ---------------- Handlers ----------------

    def _register_handlers(self):
        @self.router.message(Command("start"))
        async def on_start(message: Message):
            # если chat_id не задан — кэшируем
            if not self.chat_id:
                self.chat_id = message.chat.id
                log.info(f"[BOT] bind chat_id={self.chat_id}")

            await message.answer(
                "Бот запущен.\nВыбирай действие ниже:",
                reply_markup=kb_main(),
            )

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

        # /help (по желанию)
        @self.router.message(Command("help"))
        async def on_help(message: Message):
            await message.answer(
                "Доступно:\n"
                "• /start — меню\n"
                "• Кнопки: Баланс / Старт / Стоп\n"
                "• Уведомления: сигналы, ответы ИИ, ордера"
            )

    # ---------------- Public API ----------------

    async def start_background(self):
        """
        Запускает поллинг в фоне (не блокирует).
        В main.py можно просто:  asyncio.create_task(bot.start_background())
        """
        async def _poll():
            try:
                log.info("[BOT] polling start")
                await self.dp.start_polling(self.bot)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                log.error(f"[BOT] polling error: {e}")

        asyncio.create_task(_poll(), name="tg-polling")

    async def notify(self, text: str):
        """
        Короткие апдейты (сигналы, LLM-ответ, вход/выход, TP/SL).
        Тихо игнорим, если chat_id ещё не известен (появится после /start).
        """
        if not self.chat_id:
            log.debug("[BOT] notify skipped (no chat_id yet)")
            return
        try:
            await self.bot.send_message(self.chat_id, text)
        except Exception as e:
            log.error(f"[BOT][ERR] {e}")

    # Опционально — остановка
    async def shutdown(self):
        try:
            await self.bot.session.close()
        except Exception:
            pass
        log.info("[BOT] shutdown")
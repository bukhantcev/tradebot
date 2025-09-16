import asyncio, json
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from config import get_tg_token, TELEGRAM_CHAT_ID
from params_store import load_params, save_params

BUTTONS = ["Старт", "Стоп", "Статус", "Параметры", "Слить лог", "Рестарт"]
kb = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text=b)] for b in BUTTONS],
    resize_keyboard=True
)

class TgBot:
    def __init__(self, controller):
        self.controller = controller
        self.bot = Bot(token=get_tg_token())
        self.dp = Dispatcher()
        self.dp.message.register(self.on_msg, F.chat.id == int(TELEGRAM_CHAT_ID))

    async def start(self):
        await self.bot.delete_webhook(drop_pending_updates=True)
        await self.dp.start_polling(self.bot)

    async def on_msg(self, m: Message):
        text = (m.text or "").strip()
        if text == "Старт":
            await self.controller.start()
            await m.answer("✅ Запущено", reply_markup=kb)
        elif text == "Стоп":
            await self.controller.stop()
            await m.answer("🛑 Остановлено", reply_markup=kb)
        elif text == "Статус":
            s = await self.controller.status()
            await m.answer(s, reply_markup=kb)
        elif text == "Параметры":
            p = load_params()
            await m.answer(f"<pre>{json.dumps(p, ensure_ascii=False, indent=2)}</pre>", parse_mode="HTML", reply_markup=kb)
        elif text == "Слить лог":
            path = await self.controller.dump_now()
            if path:
                await self.bot.send_document(chat_id=m.chat.id, document=open(path, "rb"))
            else:
                await m.answer("Лога пока нет", reply_markup=kb)
        elif text == "Рестарт":
            await self.controller.restart()
            await m.answer("♻️ Рестарт", reply_markup=kb)
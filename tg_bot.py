import asyncio, json, logging, html
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, FSInputFile
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from config import get_tg_token, TELEGRAM_CHAT_ID
from params_store import load_params

BUTTONS = ["Старт", "Стоп", "Статус", "Параметры", "Слить лог", "Баланс", "Рестарт"]
kb = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text=b)] for b in BUTTONS],
    resize_keyboard=True
)

class TgBot:
    def __init__(self, controller):
        self.controller = controller
        # ✅ aiogram ≥ 3.7: parse_mode переносим в default properties
        self.bot = Bot(
            token=get_tg_token(),
            default=DefaultBotProperties(parse_mode=ParseMode.HTML)
        )
        self.dp = Dispatcher()
        self.dp.message.register(self.on_msg, F.chat.id == int(TELEGRAM_CHAT_ID))
        self.log = logging.getLogger("tg")

    async def start(self):
        await self.bot.delete_webhook(drop_pending_updates=True)
        self.log.info("TG polling started")
        await self.dp.start_polling(self.bot)

    async def notify(self, text: str):
        try:
            await self.bot.send_message(chat_id=int(TELEGRAM_CHAT_ID), text=text)
        except Exception:
            self.log.exception("TG notify failed")

    async def send_file(self, path: str, caption: str | None = None):
        try:
            await self.bot.send_document(chat_id=int(TELEGRAM_CHAT_ID), document=FSInputFile(path), caption=caption)
        except Exception:
            self.log.exception("TG send_file failed")

    async def on_msg(self, m: Message):
        text = (m.text or "").strip()
        self.log.info(f"TG cmd: {text}")
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
            safe = html.escape(json.dumps(p, ensure_ascii=False, indent=2))
            await m.answer(f"<pre>{safe}</pre>", reply_markup=kb)
        elif text == "Слить лог":
            path = await self.controller.dump_now()
            if path:
                self.log.info(f"Sending log file {path}")
                await self.send_file(path, caption="Часовой дамп")
            else:
                await m.answer("Лога пока нет", reply_markup=kb)
        elif text == "Баланс":
            s = await self.controller.short_balance()  # кратко, без требухи
            await m.answer(s, reply_markup=kb)
        elif text == "Рестарт":
            await self.controller.restart()
            await m.answer("♻️ Рестарт", reply_markup=kb)
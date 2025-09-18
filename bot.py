import asyncio
import logging
from aiogram import Bot, Dispatcher
from aiogram.types import Message
from aiogram.filters import Command

from config import HOST_ROLE, TELEGRAM_TOKEN_LOCAL, TELEGRAM_TOKEN_SERVER, TELEGRAM_CHAT_ID

log = logging.getLogger("TGBOT")

def get_token() -> str:
    return TELEGRAM_TOKEN_SERVER if HOST_ROLE == "server" else TELEGRAM_TOKEN_LOCAL

class ControlBus:
    def __init__(self):
        self.started = False
        self.status = "idle"
        self.mode = "auto"
        self.risk_pct = 1.0
        self.commands = asyncio.Queue()

    async def put(self, cmd: dict):
        log.debug(f"[BUS→] {cmd}")
        await self.commands.put(cmd)

    async def get(self):
        cmd = await self.commands.get()
        log.debug(f"[BUS←] {cmd}")
        return cmd

def build_app(bus: ControlBus):
    bot = Bot(get_token(), parse_mode=None)
    dp = Dispatcher()

    async def only_owner(msg: Message) -> bool:
        ok = (str(msg.chat.id) == str(TELEGRAM_CHAT_ID))
        if not ok: log.warning(f"[TG][DENY] chat={msg.chat.id}")
        return ok

    @dp.message(Command("start"))
    async def cmd_start(msg: Message):
        if not await only_owner(msg): return
        await bus.put({"cmd":"start"})
        await msg.answer("Запускаю")

    @dp.message(Command("stop"))
    async def cmd_stop(msg: Message):
        if not await only_owner(msg): return
        await bus.put({"cmd":"stop"})
        await msg.answer("Останавливаю")

    @dp.message(Command("status"))
    async def cmd_status(msg: Message):
        if not await only_owner(msg): return
        await msg.answer(f"status={bus.status} mode={bus.mode} risk={bus.risk_pct}%")

    @dp.message(Command("close_all"))
    async def cmd_close(msg: Message):
        if not await only_owner(msg): return
        await bus.put({"cmd":"close_all"})
        await msg.answer("Закрываю все позиции")

    @dp.message(Command("mode"))
    async def cmd_mode(msg: Message):
        if not await only_owner(msg): return
        parts = msg.text.split()
        if len(parts) >= 2 and parts[1] in ("auto","trend","range"):
            await bus.put({"cmd":"mode","value":parts[1]})
            await msg.answer(f"mode -> {parts[1]}")
        else:
            await msg.answer("Используй: /mode auto|trend|range")

    @dp.message(Command("risk"))
    async def cmd_risk(msg: Message):
        if not await only_owner(msg): return
        parts = msg.text.split()
        if len(parts) >= 2:
            try:
                v = float(parts[1]); v = max(0.2, min(1.0, v))
                await bus.put({"cmd":"risk","value":v})
                await msg.answer(f"risk -> {v}%")
            except Exception:
                await msg.answer("Пример: /risk 0.8")
        else:
            await msg.answer("Пример: /risk 0.8")

    return bot, dp
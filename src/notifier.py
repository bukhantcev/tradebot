# src/notifier.py
import asyncio
from aiogram import Bot

class Notifier:
    def __init__(self, bot: Bot, loop: asyncio.AbstractEventLoop, chat_id: int):
        self.bot = bot
        self.loop = loop
        self.chat_id = chat_id

    def send(self, text: str):
        asyncio.run_coroutine_threadsafe(self.bot.send_message(self.chat_id, text), self.loop)

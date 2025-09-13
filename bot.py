from aiogram import Bot, Dispatcher, F, Router
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import CommandStart
from config import CFG
from log import log
from bybit_client import bybit
from db import db
import strategy as strat
from trader import init_trader
from aiogram.client.default import DefaultBotProperties

bot = Bot(CFG.tg_token, default=DefaultBotProperties(parse_mode="HTML"))
dp = Dispatcher()
rt = Router()
dp.include_router(rt)

def menu_kb():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Старт"), KeyboardButton(text="Стоп")],
            [KeyboardButton(text="Баланс"), KeyboardButton(text="Информация")],
        ],
        resize_keyboard=True
    )

@rt.message(CommandStart())
async def cmd_start(msg: Message):
    await msg.answer("Привет! Выбери действие:", reply_markup=menu_kb())

@rt.message(F.text == "Старт")
async def on_start(msg: Message):
    if msg.chat.id != CFG.tg_admin_chat_id:
        return await msg.answer("Недостаточно прав.")
    tr = init_trader(bot)
    await tr.start()

@rt.message(F.text == "Стоп")
async def on_stop(msg: Message):
    if msg.chat.id != CFG.tg_admin_chat_id:
        return await msg.answer("Недостаточно прав.")
    tr = init_trader(bot)
    await tr.stop()

@rt.message(F.text == "Баланс")
async def on_balance(msg: Message):
    if msg.chat.id != CFG.tg_admin_chat_id:
        return await msg.answer("Недостаточно прав.")
    await bybit.open()
    w = await bybit.wallet_unified()
    lst = w.get("result", {}).get("list", [])
    if not lst:
        return await msg.answer("Не удалось получить баланс.")
    total = lst[0].get("totalEquity", "0")
    # Позиция/UPL
    p = await bybit.positions()
    pl = ""
    try:
        pos = p["result"]["list"][0]
        size = float(pos.get("size", "0") or "0")
        if size > 0:
            upl = pos.get("unrealisedPnl", "")
            side = pos.get("side", "")
            avg = pos.get("avgPrice", pos.get("sessionAvgPrice", ""))
            pl = f"\nОткрыто: {side} {size}\nСредняя: {avg}\nUPL: {upl}"
    except Exception:
        pass
    await msg.answer(f"💰 Баланс (UNIFIED): <b>{total}</b>{pl}")

@rt.message(F.text == "Информация")
async def on_info(msg: Message):
    if msg.chat.id != CFG.tg_admin_chat_id:
        return await msg.answer("Недостаточно прав.")
    c5 = await db.fetch_last_n("5", 200)
    c1 = await db.fetch_last_n("1", 3)
    if not c5 or not c1:
        return await msg.answer("Данных пока недостаточно.")
    s, r, tick = strat.calc_sr(c5, CFG.sr_left, CFG.sr_right)
    m = strat.regime(c1, s, r)
    # поза
    p = await bybit.positions()
    pos_text = "нет"
    try:
        pos = p["result"]["list"][0]
        size = float(pos.get("size", "0") or "0")
        if size > 0:
            pos_text = f'{pos.get("side")} {size} @ {pos.get("avgPrice", pos.get("sessionAvgPrice",""))} SL={pos.get("stopLoss","")}'
    except Exception:
        pass
    await msg.answer(
        f"ℹ️ Инфо\nS/R: <b>{s}</b> / <b>{r}</b>\nРежим: <b>{m}</b>\nПозиция: {pos_text}"
    )
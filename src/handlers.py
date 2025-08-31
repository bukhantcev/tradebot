# src/handlers.py
from typing import Dict, Optional

from aiogram import Router, F, Bot
from aiogram.types import Message, CallbackQuery

from tinkoff.invest import CandleInterval

from .config import BEST_TICKERS
from .keyboards import main_kb, choose_kb, risk_kb, interval_kb, payin_kb
from .notifier import Notifier
from .strategy import StrategyEngine
from .tinkoff_sandbox import Sandbox

router = Router()

class Session:
    def __init__(self, sbx: Sandbox, bot: Bot, loop, chat_id: int):
        self.sbx = sbx
        self.bot = bot
        self.loop = loop
        self.chat_id = chat_id
        self.engine: Optional[StrategyEngine] = None

    def notifier(self) -> Notifier:
        return Notifier(self.bot, self.loop, self.chat_id)

    def ensure_stopped(self):
        if self.engine:
            self.engine.stop()
            self.engine = None

sessions: Dict[int, Session] = {}

@router.message(F.text)
async def any_text(msg: Message):
    await msg.answer("Выбирай действие на клавиатуре ниже:", reply_markup=main_kb())

@router.callback_query(F.data == "back")
async def back_cb(cb: CallbackQuery):
    await cb.message.edit_text("Главное меню:", reply_markup=main_kb())
    await cb.answer()

@router.callback_query(F.data == "start_choose")
async def start_choose(cb: CallbackQuery):
    await cb.message.edit_text("Выбери инструмент:", reply_markup=choose_kb())
    await cb.answer()

@router.callback_query(F.data.startswith("start_ticker:"))
async def start_ticker(cb: CallbackQuery):
    ticker = cb.data.split(":", 1)[1]
    chat_id = cb.message.chat.id
    sess = sessions.get(chat_id)
    if not sess:
        return await cb.answer("Сессия не готова", show_alert=True)
    figi = sess.sbx.resolve_figi(ticker)
    if not figi:
        return await cb.answer("Не удалось найти FIGI", show_alert=True)
    if sess.engine:
        sess.engine.stop()
    sess.engine = StrategyEngine(
        sess.sbx,
        figi=figi,
        interval=CandleInterval.CANDLE_INTERVAL_5_MIN,
        notifier=sess.notifier().send,
    )
    sess.engine.start()
    await cb.message.edit_text(f"Старт стратегии на {ticker} ({figi})", reply_markup=main_kb())
    await cb.answer()

@router.callback_query(F.data == "start_best")
async def start_best(cb: CallbackQuery):
    chat_id = cb.message.chat.id
    sess = sessions.get(chat_id)
    if not sess:
        return await cb.answer("Сессия не готова", show_alert=True)
    picked = None
    figi = None
    for t in BEST_TICKERS:
        f = sess.sbx.resolve_figi(t)
        if f:
            picked, figi = t, f
            break
    if not figi:
        return await cb.answer("Не удалось найти FIGI из списка BEST", show_alert=True)
    if sess.engine:
        sess.engine.stop()
    sess.engine = StrategyEngine(
        sess.sbx,
        figi=figi,
        interval=CandleInterval.CANDLE_INTERVAL_5_MIN,
        notifier=sess.notifier().send,
    )
    sess.engine.start()
    await cb.message.edit_text(f"Старт стратегии (BEST): {picked} → {figi}", reply_markup=main_kb())
    await cb.answer()

@router.callback_query(F.data == "stop")
async def stop_cb(cb: CallbackQuery):
    chat_id = cb.message.chat.id
    sess = sessions.get(chat_id)
    if not sess:
        return await cb.answer("Сессия не готова", show_alert=True)
    if sess.engine:
        sess.engine.stop()
        sess.engine = None
        await cb.answer("Остановлено")
    else:
        await cb.answer("Стратегия и так остановлена")
    await cb.message.edit_text("Ок, остановил.", reply_markup=main_kb())

@router.callback_query(F.data == "cancel_orders")
async def cancel_orders_cb(cb: CallbackQuery):
    chat_id = cb.message.chat.id
    sess = sessions.get(chat_id)
    if not sess:
        return await cb.answer("Сессия не готова", show_alert=True)
    try:
        sess.sbx.cancel_all()
        await cb.answer("Отменил все активные заявки")
    except Exception as e:
        await cb.answer(f"Ошибка: {e}", show_alert=True)
    await cb.message.edit_text("Готово. Что дальше?", reply_markup=main_kb())

@router.callback_query(F.data == "balance")
async def balance_cb(cb: CallbackQuery):
    chat_id = cb.message.chat.id
    sess = sessions.get(chat_id)
    if not sess:
        return await cb.answer("Сессия не готова", show_alert=True)
    total = sess.sbx.get_total_rub()
    await cb.answer()
    await cb.message.edit_text(f"💰 Портфель: ~{total:.2f} RUB", reply_markup=main_kb())

@router.callback_query(F.data == "positions")
async def positions_cb(cb: CallbackQuery):
    chat_id = cb.message.chat.id
    sess = sessions.get(chat_id)
    if not sess:
        return await cb.answer("Сессия не готова", show_alert=True)
    parts = ["📊 Позиции:"]
    for p in sess.sbx.get_positions():
        figi = getattr(p, "figi", "?")
        qty = getattr(p, "quantity", None)
        units = getattr(qty, "units", 0) if qty else 0
        parts.append(f"• {figi}: {units} лот(ов)")
    if sess.engine and sess.engine._in_pos_lots > 0:
        parts.append(f"➡️ Открыта позиция по стратегии: {sess.engine.figi}, {sess.engine._in_pos_lots} лот(ов)")
    txt = "\n".join(parts) if len(parts) > 1 else "Пусто"
    await cb.answer()
    await cb.message.edit_text(txt, reply_markup=main_kb())

@router.callback_query(F.data == "status")
async def status_cb(cb: CallbackQuery):
    chat_id = cb.message.chat.id
    sess = sessions.get(chat_id)
    if not sess:
        return await cb.answer("Сессия не готова", show_alert=True)
    st = sess.engine.status() if sess.engine else "Стратегия не запущена"
    await cb.answer()
    await cb.message.edit_text(f"ℹ️ {st}", reply_markup=main_kb())

@router.callback_query(F.data == "logs")
async def logs_cb(cb: CallbackQuery):
    chat_id = cb.message.chat.id
    sess = sessions.get(chat_id)
    if not sess:
        return await cb.answer("Сессия не готова", show_alert=True)
    logs = (sess.engine.logs[-15:] if (sess.engine and sess.engine.logs) else ["Логов пока нет"])
    txt = "🧾 Последние события:\n" + "\n".join(f"• {l}" for l in logs)
    await cb.answer()
    await cb.message.edit_text(txt, reply_markup=main_kb())

@router.callback_query(F.data == "risk_menu")
async def risk_menu_cb(cb: CallbackQuery):
    await cb.message.edit_text("Выбери риск на сделку:", reply_markup=risk_kb())
    await cb.answer()

@router.callback_query(F.data.startswith("risk_set:"))
async def risk_set_cb(cb: CallbackQuery):
    val = float(cb.data.split(":", 1)[1])
    sess = sessions.get(cb.message.chat.id)
    if not sess or not sess.engine:
        await cb.answer("Стратегия не запущена", show_alert=True)
        return
    sess.engine.set_risk(val)
    await cb.answer("Ок")
    await cb.message.edit_text("Риск обновлён.", reply_markup=main_kb())

@router.callback_query(F.data == "interval_menu")
async def interval_menu_cb(cb: CallbackQuery):
    await cb.message.edit_text("Выбери интервал свечей:", reply_markup=interval_kb())
    await cb.answer()

@router.callback_query(F.data.startswith("interval_set:"))
async def interval_set_cb(cb: CallbackQuery):
    label = cb.data.split(":", 1)[1]
    sess = sessions.get(cb.message.chat.id)
    if not sess or not sess.engine:
        await cb.answer("Стратегия не запущена", show_alert=True)
        return
    sess.engine.set_interval_label(label)
    await cb.answer("Ок")
    await cb.message.edit_text("Интервал обновлён.", reply_markup=main_kb())

@router.callback_query(F.data == "payin_menu")
async def payin_menu_cb(cb: CallbackQuery):
    await cb.message.edit_text("Пополнить на:", reply_markup=payin_kb())
    await cb.answer()

@router.callback_query(F.data.startswith("payin:"))
async def payin_cb(cb: CallbackQuery):
    amount = int(cb.data.split(":", 1)[1])
    sess = sessions.get(cb.message.chat.id)
    if not sess:
        return await cb.answer("Сессия не готова", show_alert=True)
    try:
        sess.sbx.pay_in(amount)
        await cb.answer("Готово")
        await cb.message.edit_text(f"Пополнил песочницу на {amount} RUB", reply_markup=main_kb())
    except Exception as e:
        await cb.answer(f"Ошибка: {e}", show_alert=True)

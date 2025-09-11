import asyncio
import inspect
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from config import CFG
from trader import trader
from log import log
from aiogram.client.default import DefaultBotProperties

bot = Bot(CFG.telegram_token, default=DefaultBotProperties(parse_mode="HTML"))
dp = Dispatcher()

# --- Reply keyboard with control buttons ---
kb = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="/start"), KeyboardButton(text="/stop")],
        [KeyboardButton(text="/status"), KeyboardButton(text="/balance")],
        [KeyboardButton(text="/mode"), KeyboardButton(text="/sr")],
        [KeyboardButton(text="/settings"), KeyboardButton(text="/ping")],
    ],
    resize_keyboard=True
)

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    # Полный перезапуск трейдера: если уже запущен — аккуратно останавливаем и запускаем заново
    await message.answer("♻️ Полный перезапуск…", reply_markup=kb)
    try:
        # Снимаем возможную ручную паузу
        try:
            if getattr(trader, "manual_paused", False):
                setattr(trader, "manual_paused", False)
        except Exception:
            pass
        if getattr(trader, "is_running", False):
            try:
                await trader.stop()
            except Exception as e:
                log.error(f"[TG]/start trader.stop() error={e}")
            # короткая пауза, чтобы освободились ресурсы
            await asyncio.sleep(0.2)
        # Стартуем новый цикл трейдера
        asyncio.create_task(trader.start())
        await message.answer("🚀 Трейдер запущен. Доступные кнопки ниже.", reply_markup=kb)
    except Exception as e:
        log.error(f"[TG]/start error={e}")
        await message.answer(f"⚠️ Не смог запустить: {e}")

@dp.message(Command("stop"))
async def cmd_stop(message: types.Message):
    await message.answer("⏹ Останавливаюсь: закрываю позиции и стоплю цикл…")
    # 1) Попытаться закрыть все открытые позиции/ордера (best-effort)
    try:
        if hasattr(trader, "close_all_positions") and callable(getattr(trader, "close_all_positions")):
            await trader.close_all_positions()
        elif hasattr(trader, "close_all") and callable(getattr(trader, "close_all")):
            await trader.close_all()
        elif hasattr(trader, "flatten") and callable(getattr(trader, "flatten")):
            await trader.flatten()
    except Exception as e:
        log.error(f"[TG]/stop close_all error={e}")
    # 2) Остановить торговый цикл
    try:
        if hasattr(trader, "stop") and callable(getattr(trader, "stop")):
            await trader.stop()
        setattr(trader, "is_running", False)
        # Флаг ручной паузы, чтобы по /start снова запуститься явным действием
        setattr(trader, "manual_paused", True)
    except Exception as e:
        log.error(f"[TG]/stop trader.stop() error={e}")
    # 3) НИЧЕГО не перезапускаем и процесс не завершаем — ждём явной /start
    await message.answer("🛑 Бот остановлен. Позиции закрыты (если были). Для запуска нажми /start.")

@dp.message(Command("status"))
async def cmd_status(message: types.Message):
    await message.answer(f"Mode={trader.mode}, S={trader.support}, R={trader.resistance}")


# --- /balance handler ---
@dp.message(Command("balance"))
async def cmd_balance(message: types.Message):
    # Best-effort: берем из трейдера, если есть метод; иначе короткий ответ
    text = None
    try:
        if hasattr(trader, "wallet_text"):
            fn = getattr(trader, "wallet_text")
            if inspect.iscoroutinefunction(fn):
                text = await fn()
            else:
                text = fn()
        elif hasattr(trader, "wallet_snapshot"):
            snap = await trader.wallet_snapshot()  # ожидаем dict/str
            text = f"💰 Баланс: {snap}"
    except Exception as e:
        log.error(f"[TG] /balance error={e}")
    await message.answer(text or "💰 Баланс: недоступен сейчас")


# --- /mode handler ---
@dp.message(Command("mode"))
async def cmd_mode(message: types.Message):
    await message.answer(f"📈 Режим: <b>{getattr(trader, 'mode', 'n/a')}</b>")


# --- /sr handler ---
@dp.message(Command("sr"))
async def cmd_sr(message: types.Message):
    s = getattr(trader, "support", None)
    r = getattr(trader, "resistance", None)
    await message.answer(f"🔧 Уровни:\nS={s}\nR={r}")


# --- /settings handler ---
@dp.message(Command("settings"))
async def cmd_settings(message: types.Message):
    # безопасный дамп важных полей
    fields = {
        "env": CFG.bybit_env,
        "category": CFG.bybit_category,
        "symbol": CFG.symbol,
        "leverage": getattr(CFG, "leverage", None),
        "risk_pct": getattr(CFG, "risk_pct", None),
        "telegram_chat_id": CFG.telegram_chat_id,
        "verify_ssl": getattr(CFG, "verify_ssl", True),
    }
    lines = "\n".join(f"{k} = {v}" for k, v in fields.items())
    await message.answer(f"<b>Настройки</b>\n<code>{lines}</code>")

@dp.message(Command("ping"))
async def cmd_ping(message: types.Message):
    await message.answer("pong")

async def notify(text: str):
    if CFG.telegram_chat_id:
        try:
            await bot.send_message(CFG.telegram_chat_id, text)
        except Exception as e:
            log.error(f"[TG] notify error={e}")


async def notify_trade_open(
    *,
    symbol: str,
    side: str,
    qty: str,
    entry_price: float,
    stop_loss: float | None,
    take_profit: float | None,
    reason: str,
    expected_exit: str | None = None,
    order_id: str | None = None,
    position_idx: int | None = None
):
    """
    Телеграм-уведомление об открытии позиции.
    """
    sl = f"{stop_loss}" if stop_loss is not None else "—"
    tp = f"{take_profit}" if take_profit is not None else "—"
    exp = expected_exit or "по сигналу/TP/SL"
    oid = f"\n🆔 orderId: <code>{order_id}</code>" if order_id else ""
    pidx = f" (idx={position_idx})" if position_idx is not None else ""
    text = (
        f"🟢 <b>Открыта позиция</b>{pidx}\n"
        f"{symbol} {side} qty={qty}\n"
        f"Вход: <b>{entry_price}</b>\n"
        f"SL: <b>{sl}</b> | TP: <b>{tp}</b>\n"
        f"Ожидаемое закрытие: {exp}\n"
        f"Причина: <i>{reason}</i>{oid}"
    )
    await notify(text)


async def notify_close(
    *,
    symbol: str,
    side: str,
    qty: str,
    exit_price: float,
    pnl: float | None = None,
    reason: str | None = None,
    order_id: str | None = None
):
    """
    Телеграм-уведомление о закрытии позиции.
    """
    pnl_text = f"\nP/L: <b>{pnl}</b>" if pnl is not None else ""
    why = f"\nПричина: <i>{reason}</i>" if reason else ""
    oid = f"\n🆔 orderId: <code>{order_id}</code>" if order_id else ""
    text = (
        f"🔴 <b>Закрыта позиция</b>\n"
        f"{symbol} {side} qty={qty}\n"
        f"Выход: <b>{exit_price}</b>{pnl_text}{why}{oid}"
    )
    await notify(text)

# wire notifications into trader (no circular import because trader doesn't import bot anymore)
try:
    if hasattr(trader, "set_notifiers"):
        trader.set_notifiers(notify_trade_open, notify_close)
        log.info("[TG] notifiers wired: trade_open & trade_close callbacks set")
except Exception as e:
    log.error(f"[TG] failed to set notifiers: {e}")

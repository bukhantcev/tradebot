from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton

from config import CFG
from log import log, mask
from bybit_client import BybitClient
from trader import Trader
import time

bot = Bot(CFG.tg_token)
dp = Dispatcher()
bybit = BybitClient(CFG)
trader = Trader(CFG, bot, bybit)

def kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(resize_keyboard=True, keyboard=[
        [KeyboardButton(text="/start_trade"), KeyboardButton(text="/stop_trade")],
        [KeyboardButton(text="/balance"), KeyboardButton(text="/status")],
        [KeyboardButton(text="/diag"), KeyboardButton(text="/ping")],
    ])

@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer("Готов. Кнопки ниже.", reply_markup=kb())

@dp.message(Command("start_trade"))
async def start_trade(message: Message):
    if CFG.tg_chat_id == 0:
        CFG.tg_chat_id = message.chat.id
    await trader.start()
    await message.answer("▶️ Старт торговли.", reply_markup=kb())

@dp.message(Command("stop_trade"))
async def stop_trade(message: Message):
    await trader.stop()
    await message.answer("🛑 Стоп торговли (все позиции закрыты, ордера отменены).")

@dp.message(Command("balance"))
async def balance(message: Message):
    try:
        w = await bybit.get_wallet_balance()

        def fmt2(x) -> str:
            try:
                return f"{float(x):.2f}"
            except Exception:
                return str(x)

        coins = []
        for a in w.get("result", {}).get("list", []):
            for c in a.get("coin", []):
                sym = c.get("coin")
                if sym in ("USDT", "USD", "BTC", "ETH"):
                    coins.append(f"{sym}: equity={fmt2(c.get('equity', 0))}")
        if not coins:
            coins.append("Нет данных по монетам USDT/USD/BTC/ETH")
        await message.answer("💰 Баланс:\n" + "\n".join(coins))
    except Exception as e:
        await message.answer(f"Ошибка баланса: {e}\nПроверь BYBIT_API_KEY/SECRET и BYBIT_ENV в .env")

@dp.message(Command("status"))
async def status(message: Message):
    regime = trader.last_regime or "NA"
    pos = trader.position_side or "flat"
    tp_real = getattr(trader, 'position_tp', None) or '—'
    txt = [
        f"ℹ️ Режим: {regime}",
        f"Позиция: {pos}",
        f"Qty: {trader.position_qty}",
        f"Entry: {trader.position_entry}",
        f"SL: {trader.position_sl}",
        f"TP: {tp_real}",
        f"Cooldown: {'yes' if time.time() < trader.cooldown_until else 'no'}",
    ]
    await message.answer("\n".join(txt))

@dp.message(Command("help"))
async def help_(message: Message):
    await message.answer(
        "/start_trade — запустить торговлю\n"
        "/stop_trade — остановить и закрыть всё\n"
        "/balance — показать баланс\n"
        "/status — режим/позиция/SL/TP\n"
        "Логи: LOG_HTTP, LOG_HTTP_BODIES, LOG_WS_RAW, LOG_SIGNALS в .env\n"
    )

@dp.message(Command("ping"))
async def ping(message: Message):
    await message.answer("pong")

@dp.message(Command("diag"))
async def diag(message: Message):
    cfg = CFG
    try:
        instr = await bybit.get_instruments_info()
        ok_market = instr.get("retCode") == 0
    except Exception:
        ok_market = False
    txt = (
        "⚙️ DIAG: "
        f"TG_CHAT_ID={cfg.tg_chat_id} | WS={cfg.ws_public_url}\n"
        f"SYMBOL={cfg.symbol} {cfg.category} | ENV={cfg.bybit_env}\n"
        f"API_KEY={mask(cfg.bybit_key)} SECRET={mask(cfg.bybit_secret)}\n"
        f"LEVERAGE={cfg.leverage} | RISK%={cfg.risk_pct}\n"
        f"EMA={cfg.ema_fast}/{cfg.ema_slow} ATR_LEN={cfg.atr_len} CH_LOOKBACK={cfg.channel_lookback}\n"
        f"TRAILING={cfg.use_trailing} act={cfg.trailing_activation}% dist={cfg.trailing_distance}%\n"
        f"REST_OK={ok_market}"
    )
    await message.answer(txt)
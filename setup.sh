#!/usr/bin/env bash
set -e

# 1) папки
mkdir -p src

# 2) файлы
cat > requirements.txt <<'REQ'
aiogram>=3.4,<4
python-dotenv>=1.0,<2
requests>=2.31,<3

# Tinkoff Invest SDK (beta с песочницей)
tinkoff-investments==0.2.0b116

# gRPC стек (совместим с Py3.12 и этим SDK)
grpcio>=1.56,<2
protobuf>=4.25.3,<5
REQ

cat > .env.example <<'ENV'
TELEGRAM_BOT_TOKEN=1234567890:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
SANDBOX_TOKEN=твой_песочный_токен_от_Tinkoff
SANDBOX_INIT_RUB=100000
FIGI_DEFAULT=BBG004730N88
ENV

cat > .gitignore <<'GI'
.venv/
__pycache__/
*.pyc
.env
bot.log
GI

cat > bot.py <<'PY'
# bot.py
import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.types import Message

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
    bot = Bot(cfg.tg_token, parse_mode="HTML")
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
PY

cat > src/config.py <<'PY'
# src/config.py
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# Три “топовых” тикера для быстрого запуска
BEST_TICKERS = ["VTBR", "GAZP", "SBER"]  # можно поменять под себя

# Фолбэк: тикер (SECID) → FIGI (для стабильной работы даже при багающем справочнике)
FIGI_FALLBACK = {
    "SBER": "BBG004730N88",
    "GAZP": "BBG004730RP0",
    "VTBR": "BBG004731032",
    "GMKN": "BBG0047315D0",
}

# Пресеты настроек через кнопки
RISK_PRESETS = [0.005, 0.01, 0.02]           # 0.5%, 1%, 2%
INTERVAL_PRESETS = ["1m", "5m"]              # кнопки для интервала
PAYIN_PRESETS = [10_000, 50_000, 100_000]    # RUB

@dataclass
class Settings:
    tg_token: str
    sandbox_token: str
    sandbox_init_rub: float
    figi_default: str

def get_settings() -> Settings:
    tg = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    sbx = os.getenv("SANDBOX_TOKEN", "").strip()
    if not tg:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан в .env")
    if not sbx:
        raise RuntimeError("SANDBOX_TOKEN не задан в .env")

    return Settings(
        tg_token=tg,
        sandbox_token=sbx,
        sandbox_init_rub=float(os.getenv("SANDBOX_INIT_RUB", "100000")),
        figi_default=os.getenv("FIGI_DEFAULT", "BBG004730N88"),
    )
PY

cat > src/tinkoff_sandbox.py <<'PY'
# src/tinkoff_sandbox.py
import inspect
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional, Iterable

import requests
from tinkoff.invest import (
    Client,
    OrderDirection,
    OrderType,
    CandleInterval,
    InstrumentIdType,
    MoneyValue,
)
from tinkoff.invest.services import Services

from .config import FIGI_FALLBACK

def quotation_to_float(q) -> float:
    try:
        return (q.units or 0) + (q.nano or 0) / 1e9
    except Exception:
        return float(q)

def money_to_float(m) -> float:
    return (m.units or 0) + (m.nano or 0) / 1e9

FIGI_TO_TICKER = {
    "BBG004730N88": "SBER",
    "BBG004730RP0": "GAZP",
    "BBG004731032": "VTBR",
    "BBG0047315D0": "GMKN",
}
ISS_INTERVAL_MAP = {
    CandleInterval.CANDLE_INTERVAL_1_MIN: 1,
    CandleInterval.CANDLE_INTERVAL_5_MIN: 5,
}

def load_candles(services: Services, figi: str, hours: int, interval: CandleInterval):
    to = datetime.now(timezone.utc)
    frm = to - timedelta(hours=hours)
    try:
        return services.market_data.get_candles(figi=figi, from_=frm, to=to, interval=interval).candles
    except Exception:
        pass
    secid = FIGI_TO_TICKER.get(figi)
    if not secid:
        return []
    iss_interval = ISS_INTERVAL_MAP.get(interval, 5)
    url = (
        "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/"
        f"securities/{secid}/candles.json?interval={iss_interval}"
        f"&from={frm.strftime('%Y-%m-%dT%H:%M:%S')}"
    )
    r = requests.get(url, timeout=10)
    j = r.json()
    cols = j["candles"]["columns"]
    idx = {name: i for i, name in enumerate(cols)}
    out = []
    for row in j["candles"]["data"]:
        out.append(type("C", (), {
            "open": float(row[idx.get("open", 0)]),
            "high": float(row[idx.get("high", 1)]),
            "low":  float(row[idx.get("low", 2)]),
            "close":float(row[idx.get("close", 3)]),
            "volume": int(row[idx.get("volume", 6)]) if idx.get("volume") is not None else 0,
        }))
    keep = int(hours * 60 / iss_interval) + 3
    return out[-keep:]

class Sandbox:
    def __init__(self, token: str, init_rub: float):
        self._client = Client(token)
        self.client: Optional[Services] = None
        self.account_id: Optional[str] = None
        self.init_rub = init_rub

    def __enter__(self):
        self.client = self._client.__enter__()
        self._ensure_account()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._client.__exit__(exc_type, exc, tb)

    def _ensure_account(self):
        if not hasattr(self.client, "sandbox"):
            import tinkoff.invest as ti
            raise RuntimeError(
                f"[SDK] sandbox отсутствует. Python={sys.executable} "
                f"Pkg={inspect.getfile(ti)} Ver={getattr(ti,'__version__','unknown')}"
            )
        try:
            try:
                accs = self.client.sandbox.users.get_accounts().accounts
            except AttributeError:
                accs = self.client.sandbox.get_sandbox_accounts().accounts
            if accs:
                self.account_id = accs[0].id
            else:
                opened = self.client.sandbox.open_sandbox_account()
                self.account_id = opened.account_id
            try:
                p = self.client.sandbox.operations.get_portfolio(account_id=self.account_id)
            except AttributeError:
                p = self.client.sandbox.get_sandbox_portfolio(account_id=self.account_id)
            total = money_to_float(p.total_amount_portfolio)
            if total < 1:
                self.pay_in(self.init_rub)
        except Exception as e:
            raise RuntimeError(f"[SBX] init failed: {e}") from e

    def post_market_order(self, figi: str, lots: int, side: OrderDirection):
        return self.client.sandbox.post_sandbox_order(
            figi=figi,
            quantity=lots,
            price=None,
            direction=side,
            account_id=self.account_id,
            order_type=OrderType.ORDER_TYPE_MARKET,
        )

    def cancel_all(self):
        try:
            orders = self.client.sandbox.orders.get_orders(account_id=self.account_id).orders
            for o in orders:
                self.client.sandbox.orders.cancel_order(account_id=self.account_id, order_id=o.order_id)
        except AttributeError:
            for o in self.client.sandbox.get_sandbox_orders(account_id=self.account_id).orders:
                self.client.sandbox.cancel_sandbox_order(account_id=self.account_id, order_id=o.order_id)

    def pay_in(self, amount_rub: int | float):
        self.client.sandbox.sandbox_pay_in(
            account_id=self.account_id,
            amount=MoneyValue(currency="rub", units=int(amount_rub), nano=0),
        )

    def get_portfolio(self):
        try:
            return self.client.sandbox.operations.get_portfolio(account_id=self.account_id)
        except AttributeError:
            return self.client.sandbox.get_sandbox_portfolio(account_id=self.account_id)

    def get_positions(self) -> Iterable:
        pf = self.get_portfolio()
        return getattr(pf, "positions", [])

    def get_total_rub(self) -> float:
        pf = self.get_portfolio()
        return money_to_float(pf.total_amount_portfolio)

    def get_lot_size(self, figi: str) -> int:
        try:
            ins = self.client.instruments.get_instrument_by(
                id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, id=figi
            )
            return ins.instrument.lot or 1
        except Exception:
            return 1

    def resolve_figi(self, ticker: str) -> Optional[str]:
        t = ticker.upper()
        try:
            for inst in self.client.instruments.shares().instruments:
                if inst.ticker.upper() == t:
                    return inst.figi
        except Exception:
            pass
        return FIGI_FALLBACK.get(t)
PY

cat > src/strategy.py <<'PY'
# src/strategy.py
import threading
import time
from typing import Callable, List
from tinkoff.invest import CandleInterval, OrderDirection

from .tinkoff_sandbox import load_candles, quotation_to_float

INTERVAL_LABEL_TO_ENUM = {
    "1m": CandleInterval.CANDLE_INTERVAL_1_MIN,
    "5m": CandleInterval.CANDLE_INTERVAL_5_MIN,
}

class StrategyEngine:
    def __init__(
        self,
        sbx,
        figi: str,
        interval: CandleInterval = CandleInterval.CANDLE_INTERVAL_5_MIN,
        n_enter: int = 55,
        n_exit: int = 20,
        atr_n: int = 14,
        risk_pct: float = 0.01,
        atr_k: float = 2.0,
        notifier: Callable[[str], None] | None = None,
    ):
        self.sbx = sbx
        self.services = sbx.client
        self.figi = figi
        self.interval = interval
        self.n_enter = n_enter
        self.n_exit = n_exit
        self.atr_n = atr_n
        self.risk_pct = risk_pct
        self.atr_k = atr_k
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._in_pos_lots = 0
        self.logs: List[str] = []
        self._notify = notifier or (lambda _: None)

    def log(self, msg: str):
        self.logs.append(msg)
        self.logs = self.logs[-200:]
        self._notify(msg)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="strategy", daemon=True)
        self._thread.start()
        self.log(f"▶️ Стратегия запущена для {self.figi} {self.interval.name} | риск={self.risk_pct*100:.1f}%")

    def stop(self):
        self._stop.set()
        self.log("⏹ Стратегия остановлена")

    def status(self) -> str:
        running = self._thread.is_alive() if self._thread else False
        return f"figi={self.figi} running={running} lots_in_pos={self._in_pos_lots} risk={self.risk_pct*100:.1f}% interval={self._interval_label()}"

    def set_risk(self, pct: float):
        self.risk_pct = pct
        self.log(f"⚙️ Риск обновлён: {pct*100:.1f}%")

    def set_interval_label(self, label: str):
        if label not in INTERVAL_LABEL_TO_ENUM:
            self.log(f"Некорректный интервал: {label}")
            return
        self.interval = INTERVAL_LABEL_TO_ENUM[label]
        self.log(f"⏱ Интервал обновлён: {label}")

    def _interval_label(self) -> str:
        for k, v in INTERVAL_LABEL_TO_ENUM.items():
            if v == self.interval:
                return k
        return "5m"

    def _run(self):
        lot = self.sbx.get_lot_size(self.figi)
        self.log(f"Лот = {lot}")
        last_signal = None
        while not self._stop.is_set():
            try:
                candles = load_candles(self.services, self.figi, hours=96, interval=self.interval)
                if len(candles) < max(self.n_enter, self.n_exit, self.atr_n) + 2:
                    time.sleep(10)
                    continue
                highs = [quotation_to_float(c.high) for c in candles]
                lows = [quotation_to_float(c.low) for c in candles]
                closes = [quotation_to_float(c.close) for c in candles]
                upper = max(highs[-self.n_enter:])
                lower = min(lows[-self.n_exit:])
                last_close = closes[-1]
                atr = self._calc_atr(highs, lows, closes, self.atr_n)
                stop_price = last_close - self.atr_k * atr
                signal = "long" if last_close > upper else ("flat" if last_close < lower else last_signal)
                if (last_signal != "long") and (signal == "long"):
                    qty = self._calc_lots(last_close, stop_price, lot)
                    if qty > 0:
                        self.log(f"🟢 ENTRY long {qty} лот(ов) @~{last_close:.2f}, stop ~{stop_price:.2f}, ATR {atr:.2f}")
                        self.sbx.post_market_order(self.figi, qty, OrderDirection.ORDER_DIRECTION_BUY)
                        self._in_pos_lots = qty
                        last_signal = "long"
                    else:
                        self.log("ℹ️ Пропуск входа: размер позиции по риску = 0")
                        last_signal = "flat"
                elif (last_signal == "long") and (signal == "flat" or last_close < stop_price):
                    if self._in_pos_lots > 0:
                        self.log(f"🔴 EXIT {self._in_pos_lots} лот(ов) @~{last_close:.2f}")
                        self.sbx.post_market_order(self.figi, self._in_pos_lots, OrderDirection.ORDER_DIRECTION_SELL)
                        self._in_pos_lots = 0
                    last_signal = "flat"
                time.sleep(60)
            except Exception as e:
                self.log(f"❗️Ошибка стратегии: {e}")
                time.sleep(5)

    @staticmethod
    def _calc_atr(highs, lows, closes, n):
        trs = []
        for i in range(1, len(closes)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            trs.append(max(hl, hc, lc))
        return sum(trs[-n:]) / n

    def _calc_lots(self, price: float, stop_price: float, lot_size: int) -> int:
        port = self.sbx.get_total_rub()
        risk_rub = port * self.risk_pct
        per_lot_risk = max(price - stop_price, 0.01) * lot_size
        lots = int(risk_rub // per_lot_risk)
        return max(lots, 0)
PY

cat > src/notifier.py <<'PY'
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
PY

cat > src/keyboards.py <<'PY'
# src/keyboards.py
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from .config import BEST_TICKERS, RISK_PRESETS, INTERVAL_PRESETS, PAYIN_PRESETS

def main_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🚀 Старт BEST", callback_data="start_best")],
        [InlineKeyboardButton(text="🎯 Старт (выбор из 3)", callback_data="start_choose")],
        [InlineKeyboardButton(text="⏹ Стоп", callback_data="stop"), InlineKeyboardButton(text="❌ Отменить заявки", callback_data="cancel_orders")],
        [InlineKeyboardButton(text="💰 Баланс", callback_data="balance"),
         InlineKeyboardButton(text="📊 Позиции", callback_data="positions")],
        [InlineKeyboardButton(text="⚙️ Риск", callback_data="risk_menu"),
         InlineKeyboardButton(text="⏱ Интервал", callback_data="interval_menu")],
        [InlineKeyboardButton(text="➕ Пополнить", callback_data="payin_menu"),
         InlineKeyboardButton(text="ℹ️ Статус", callback_data="status")],
        [InlineKeyboardButton(text="🧾 Логи", callback_data="logs")],
    ])

def choose_kb() -> InlineKeyboardMarkup:
    row = [InlineKeyboardButton(text=t, callback_data=f"start_ticker:{t}") for t in BEST_TICKERS]
    return InlineKeyboardMarkup(inline_keyboard=[row, [InlineKeyboardButton(text="⬅️ Назад", callback_data="back")]])

def risk_kb() -> InlineKeyboardMarkup:
    row = [InlineKeyboardButton(text=f"{int(p*1000)/10:.1f}%", callback_data=f"risk_set:{p}") for p in RISK_PRESETS]
    return InlineKeyboardMarkup(inline_keyboard=[row, [InlineKeyboardButton(text="⬅️ Назад", callback_data="back")]])

def interval_kb() -> InlineKeyboardMarkup:
    row = [InlineKeyboardButton(text=lab, callback_data=f"interval_set:{lab}") for lab in INTERVAL_PRESETS]
    return InlineKeyboardMarkup(inline_keyboard=[row, [InlineKeyboardButton(text="⬅️ Назад", callback_data="back")]])

def payin_kb() -> InlineKeyboardMarkup:
    row = [InlineKeyboardButton(text=f"{amt//1000}k", callback_data=f"payin:{amt}") for amt in PAYIN_PRESETS]
    return InlineKeyboardMarkup(inline_keyboard=[row, [InlineKeyboardButton(text="⬅️ Назад", callback_data="back")]])
PY

cat > src/handlers.py <<'PY'
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
PY

# 3) подсказка
echo
echo "✅ Структура создана. Далее:"
echo "python -m venv .venv && source .venv/bin/activate"
echo "pip install -r requirements.txt"
echo "cp .env.example .env  # и впиши TELEGRAM_BOT_TOKEN, SANDBOX_TOKEN"
echo "python bot.py"

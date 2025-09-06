from __future__ import annotations
import os
import sys
import asyncio
import time
import hmac
from hashlib import sha256
from urllib.parse import urlencode
import signal
import subprocess
import logging
import ssl
import certifi
import contextlib
from logging.handlers import RotatingFileHandler
from typing import Optional

import aiohttp
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import CommandStart
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

# ----------------------------
# ENV & logging
# ----------------------------
load_dotenv()
BASE_URL = os.getenv("BASE_URL", "https://api-testnet.bybit.com").rstrip("/")
API_KEY = os.getenv("API_KEY", "").strip()
API_SECRET = os.getenv("API_SECRET", "").strip()
SYMBOL = os.getenv("SYMBOL", "BTCUSDT").strip()
CATEGORY = os.getenv("CATEGORY", "linear").strip()
RECV_WINDOW = int(os.getenv("RECV_WINDOW", "5000"))
BYBIT_VERIFY_SSL = os.getenv("BYBIT_VERIFY_SSL", "1").strip() not in ("0", "false", "no")
LOG_FILE = os.getenv("LOG_FILE", "tradebot.log")
WORK_DIR = os.path.abspath(os.getenv("WORK_DIR", "."))
PYTHON_BIN = os.getenv("PYTHON_BIN", "python3")
STRATEGY_ENTRY = os.getenv("STRATEGY_ENTRY", "main.py")
PID_FILE = os.path.join(WORK_DIR, "strategy.pid")

OUT_LOG_FILE = os.path.join(WORK_DIR, os.getenv("OUT_LOG_FILE", "strategy.out.log"))

TG_TOKEN = (os.getenv("TG_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "").strip()  # optional allowlist single chat

logger = logging.getLogger("tg-bot")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler("tg_bot.log", maxBytes=3_000_000, backupCount=2, encoding="utf-8")
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(handler)

# Runtime subscribers & forwarders
SUB_CHATS: set[int] = set()
FORWARDER_TASKS: dict[int, list[asyncio.Task]] = {}

def _paths_to_tail() -> list[str]:
    paths = [
        os.path.join(WORK_DIR, LOG_FILE) if not os.path.isabs(LOG_FILE) else LOG_FILE,
        OUT_LOG_FILE,
    ]
    # Keep only existing/unique paths; missing ones will be retried by task itself
    seen, uniq = set(), []
    for p in paths:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq

async def start_forwarders_for_chat(bot: Bot, chat_id: int):
    # If already running for this chat — skip
    if chat_id in FORWARDER_TASKS:
        return
    tasks: list[asyncio.Task] = []
    for pth in _paths_to_tail():
        tasks.append(asyncio.create_task(log_forwarder_task(bot, chat_id, pth)))
    FORWARDER_TASKS[chat_id] = tasks
    logger.info(f"Log forwarders started for chat {chat_id}")

async def stop_forwarders_for_chat(chat_id: int):
    tasks = FORWARDER_TASKS.pop(chat_id, [])
    for t in tasks:
        t.cancel()
    for t in tasks:
        with contextlib.suppress(Exception):
            await t
    logger.info(f"Log forwarders stopped for chat {chat_id}")

# SSL context for aiohttp (fixes CERTIFICATE_VERIFY_FAILED on some macOS/Python installs)
if BYBIT_VERIFY_SSL:
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
else:
    # WARNING: disabling verification is insecure; use only for quick diagnostics
    SSL_CONTEXT = ssl._create_unverified_context()

# ----------------------------
# Bybit v5 REST minimal client (HMAC SHA256)
# ----------------------------

def _ts_ms() -> str:
    return str(int(time.time() * 1000))


def _sign(ts: str, recv_window: int, query_str: str) -> str:
    # v5 signature payload = timestamp + api_key + recv_window + query/body
    payload = f"{ts}{API_KEY}{recv_window}{query_str}"
    return hmac.new(API_SECRET.encode(), payload.encode(), sha256).hexdigest()


def _headers(ts: str, sign: str) -> dict:
    return {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-SIGN": sign,
        "X-BAPI-TIMESTAMP": ts,
        "X-BAPI-RECV-WINDOW": str(RECV_WINDOW),
        "Content-Type": "application/json",
    }


async def bybit_get(session: aiohttp.ClientSession, path: str, params: dict) -> dict:
    ts = _ts_ms()
    qs = urlencode(params)
    sign = _sign(ts, RECV_WINDOW, qs)
    url = f"{BASE_URL}{path}?{qs}"
    async with session.get(url, headers=_headers(ts, sign), timeout=aiohttp.ClientTimeout(total=20)) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {text}")
        data = await resp.json()
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit error: {data}")
    return data.get("result", {})

# ----------------------------
# Strategy process control
# ----------------------------

def is_running() -> bool:
    if not os.path.exists(PID_FILE):
        return False
    try:
        with open(PID_FILE, "r", encoding="utf-8") as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def start_strategy() -> str:
    if is_running():
        return "Стратегия уже запущена ✅"
    try:
        strategy_path = os.path.join(WORK_DIR, STRATEGY_ENTRY)
        if not os.path.isfile(strategy_path):
            return f"Не найден файл стратегии: {strategy_path}"

        # Append process output to a dedicated file that the bot can tail
        out_path = OUT_LOG_FILE
        try:
            out = open(out_path, "ab", buffering=0)
        except Exception as e:
            return f"Не удалось открыть лог-файл для стратегии ({out_path}): {e}"

        try:
            p = subprocess.Popen(
                [PYTHON_BIN, "-u", strategy_path],
                cwd=WORK_DIR,
                stdout=out,
                stderr=out,
                start_new_session=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        finally:
            try:
                out.flush()
                out.close()
            except Exception:
                pass

        with open(PID_FILE, "w", encoding="utf-8") as f:
            f.write(str(p.pid))
        return f"Стратегия запущена ▶️ (PID {p.pid}). Вывод пишется в {out_path}"
    except Exception as e:
        return f"Не удалось запустить: {e}"


def stop_strategy() -> str:
    if not os.path.exists(PID_FILE):
        return "PID не найден — возможно стратегия уже остановлена."
    try:
        with open(PID_FILE, "r", encoding="utf-8") as f:
            pid = int(f.read().strip())
        os.kill(pid, signal.SIGTERM)
        # Give it a moment to exit gracefully
        for _ in range(10):
            try:
                os.kill(pid, 0)
                time.sleep(0.2)
            except ProcessLookupError:
                break
        # Force kill if still alive
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        os.remove(PID_FILE)
        return "Стратегия остановлена ⏹"
    except Exception as e:
        return f"Ошибка остановки: {e}"

# ----------------------------
# Business actions
# ----------------------------

BALANCE_PATH = "/v5/account/wallet-balance"
ORDERS_PATH = "/v5/order/realtime"

async def get_balance_text(session: aiohttp.ClientSession) -> str:
    """Return pretty balance text for UNIFIED (UTA) only.

    Notes:
    - Bybit v5 testnet often returns retCode=10001 (accountType only support UNIFIED)
      if you pass CONTRACT. We therefore query UNIFIED exclusively.
    - We try to surface clear, actionable diagnostics to the user.
    """
    try:
        res = await bybit_get(session, BALANCE_PATH, {"accountType": "UNIFIED", "coin": "USDT"})
        lst = res.get("list", []) or []
        if not lst:
            return "Баланс: пустой ответ от API"

        acc = lst[0]
        coins = acc.get("coin", []) or []
        # Prefer coin-level USDT, but also show totals if available
        equity_usdt = None
        avail_usdt = None
        for c in coins:
            if (c.get("coin") or "").upper() == "USDT":
                # Fields vary across builds; try several fallbacks
                equity_usdt = float(c.get("equity") or c.get("walletBalance") or 0)
                avail_usdt = float(c.get("availableToWithdraw") or c.get("availableToWithdrawAmount") or c.get("free") or 0)
                break

        total_equity = acc.get("totalEquity") or acc.get("totalWalletBalance")

        if equity_usdt is not None:
            return (
                f"Баланс (UNIFIED):\n"
                f"• USDT equity: <b>{equity_usdt:.2f}</b>\n"
                f"• Доступно: { (avail_usdt if avail_usdt is not None else 0):.2f} USDT"
                + (f"\n• Всего (по всем монетам): {float(total_equity):.2f}" if total_equity is not None else "")
            )
        # Fallback if there is no explicit USDT coin object
        if total_equity is not None:
            return f"Баланс (UNIFIED): totalEquity={float(total_equity):.2f} (USDT не найден в разрезе монет)"
        return "Баланс: не удалось распарсить ответ"

    except RuntimeError as e:
        # Unwrap Bybit structured error and provide hints
        msg = str(e)
        if "retCode" in msg:
            try:
                # best-effort parse
                import json, re
                m = re.search(r"\{.*\}", msg)
                data = json.loads(m.group(0)) if m else {}
                code = data.get("retCode")
                rmsg = data.get("retMsg")
                if code == 10001:
                    return (
                        "Баланс: аккаунт поддерживает только UNIFIED. "
                        "Убедитесь, что в .env используется UTA на тестнете и не используйте CONTRACT."
                    )
                if code == 10003:
                    return (
                        "Баланс: API key invalid (10003). Проверьте, что ключ создан на TESTNET, "
                        "имеет права Read/Trade и совпадает с BASE_URL=https://api-testnet.bybit.com."
                    )
                if code in (10004, 10005, 10006):
                    return f"Баланс: авторизация отклонена ({code} {rmsg}). Проверьте API Secret/права."
            except Exception:
                pass
        return f"Не удалось получить баланс: {e}"
    except Exception as e:
        return f"Не удалось получить баланс: {e}"


async def get_orders_text(session: aiohttp.ClientSession) -> str:
    try:
        res = await bybit_get(session, ORDERS_PATH, {"category": CATEGORY, "symbol": SYMBOL})
        list_ = res.get("list", [])
        if not list_:
            return "Ордера: нет открытых ✅"
        lines = ["Ордера:"]
        for o in list_:
            side = o.get("side")
            qty = o.get("qty")
            price = o.get("price")
            status = o.get("orderStatus")
            t = o.get("timeInForce")
            lines.append(f"• {side} {qty} @ {price} [{status}/{t}]")
        return "\n".join(lines)
    except Exception as e:
        return f"Ошибка получения ордеров: {e}"

# ----------------------------
# Log forwarder (tail LOG_FILE)
# ----------------------------

EVENT_TAGS = (
    "[SCALP]",
    "[SCALP][ENTRY]",
    "[SCALP][EXIT]",
    "[SCALP][STOP]",
    "[SCALP][TP]",
    "[LEVELS]",
    "[TREND]",
    "[RISK]",
    "[STAT]",
)

async def log_forwarder_task(bot: Bot, chat_id: int, log_path: str):
    pos = 0
    primed = False
    while True:
        try:
            if not os.path.exists(log_path):
                await asyncio.sleep(1.0)
                continue
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                if pos > 0:
                    try:
                        f.seek(pos)
                    except Exception:
                        pos = 0
                if pos == 0 and not primed:
                    try:
                        f.seek(0, os.SEEK_END)
                        pos = f.tell()
                        primed = True
                    except Exception:
                        pass
                for line in f:
                    pos = f.tell()
                    if any(tag in line for tag in EVENT_TAGS):
                        try:
                            await bot.send_message(chat_id, line.strip())
                        except Exception:
                            pass
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(1.0)

# ----------------------------
# Telegram bot wiring (aiogram v3)
# ----------------------------

KB = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="▶️ Запуск"), KeyboardButton(text="⏹ Стоп")],
        [KeyboardButton(text="💰 Баланс"), KeyboardButton(text="📜 Ордера")],
        [KeyboardButton(text="📊 Статус")],
    ],
    resize_keyboard=True
)

def allowed_chat(message: types.Message) -> bool:
    global TG_CHAT_ID
    if TG_CHAT_ID:
        return str(message.chat.id) == TG_CHAT_ID
    # First chat becomes the default; remember only in-memory
    TG_CHAT_ID = str(message.chat.id)
    return True

async def on_start(message: types.Message, bot: Bot):
    if not allowed_chat(message):
        return
    await message.answer(
        "Готов к работе. Кнопки ниже:\n"
        "• ▶️ Запуск — запускает СКАЛЬПИНГ-стратегию\n"
        "• ⏹ Стоп — останавливает стратегию\n"
        "• 💰 Баланс — читает UTA баланс\n"
        "• 📜 Ордера — открытые ордера\n"
        "• 📊 Статус — последние события скальпинга (входы/выходы, тренд, уровни)\n\n"
        "Логи стратегии (теги [SCALP], [ENTRY]/[EXIT]/[STOP]/[TP], [LEVELS], [TREND]) будут автоматически пересылаться сюда.",
        reply_markup=KB,
    )
    # subscribe this chat for log streaming
    await start_forwarders_for_chat(bot, message.chat.id)

# ----------------------------
# Status snippet helper
# ----------------------------

async def get_status_snippet(max_lines: int = 25) -> str:
    path = OUT_LOG_FILE
    try:
        if not os.path.exists(path):
            return "Статус: лог-файл ещё не создан. Нажмите ▶️ Запуск."
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            try:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 100_000))  # последние ~100КБ
            except Exception:
                pass
            lines = f.read().splitlines()
        tags = set(EVENT_TAGS)
        filtered = [ln for ln in lines if any(tag in ln for tag in tags)]
        if not filtered:
            return "Статус: пока нет событий скальпинга."
        return "\n".join(filtered[-max_lines:])
    except Exception as e:
        return f"Статус: ошибка чтения лога — {e}"

async def on_text(message: types.Message, session: aiohttp.ClientSession, bot: Bot):
    if not allowed_chat(message):
        return
    # Ensure logs are forwarded to this chat
    await start_forwarders_for_chat(bot, message.chat.id)
    text = (message.text or "").strip().lower()

    if text.startswith("▶️".lower()) or text.startswith("запуск"):
        msg = start_strategy()
        await message.answer(msg)
        return

    if text.startswith("⏹".lower()) or text.startswith("стоп"):
        msg = stop_strategy()
        await message.answer(msg)
        return

    if text.startswith("💰".lower()) or "баланс" in text:
        await message.answer("Проверяю баланс…")
        await message.answer(await get_balance_text(session))
        return

    if text.startswith("📜".lower()) or "ордер" in text:
        await message.answer(await get_orders_text(session))
        return

    if text.startswith("📊".lower()) or "статус" in text:
        await message.answer(await get_status_snippet())
        return

    await message.answer("Не понял команду. Используйте кнопки ниже (добавлен 📊 Статус).", reply_markup=KB)

async def main():
    if not TG_TOKEN:
        raise SystemExit("TG_TOKEN is empty. Set in .env")

    bot = Bot(token=TG_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()

    # Shared aiohttp session
    session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=SSL_CONTEXT))

    # Handlers
    dp.message.register(on_start, CommandStart())

    async def _on_text_wrapper(message: types.Message):
        await on_text(message, session, bot)

    dp.message.register(_on_text_wrapper, F.text & ~F.via_bot)

    logger.info("Telegram bot polling (aiogram) …")
    try:
        await dp.start_polling(bot)
    finally:
        # stop all forwarders
        try:
            chats = list(FORWARDER_TASKS.keys())
            for cid in chats:
                await stop_forwarders_for_chat(cid)
        except Exception:
            pass
        await session.close()

if __name__ == "__main__":
    try:
        import contextlib  # local import for suppress
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("Bot stopped")

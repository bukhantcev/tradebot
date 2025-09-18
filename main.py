# main.py
import os
import asyncio
import signal
import logging
from typing import Optional, Dict, Any

from bybit_client import BybitClient
from data import DataManager
from trader import Trader
from strategy import StrategyEngine
from bot import TgBot
from features import load_recent_1m

# ---------------- logging ----------------

def setup_logging():
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
    )
    # приглушим шумные библиотеки
    for noisy in ("aiosqlite", "httpcore", "httpx", "websockets", "asyncio", "aiogram"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

setup_logging()
log = logging.getLogger("MAIN")

# ---------------- strategy loop ----------------

async def strategy_loop(strat: StrategyEngine, trader: Trader, poll_sec: float = 1.0):
    """
    Пуллим БД раз в poll_sec и вызываем стратегию,
    когда появляется НОВАЯ закрытая 1m свеча.
    """
    last_ts: Optional[int] = None
    while True:
        try:
            df = await load_recent_1m(limit=1)  # только последняя закрытая минута
            if not df.empty:
                ts = int(df.iloc[-1]["ts_ms"])
                if last_ts is None:
                    last_ts = ts  # инициализируем, не триггерим
                elif ts != last_ts:
                    # появилась новая закрытая минута → спросим стратегию
                    sig = await strat.on_kline_closed()
                    last_ts = ts

                    # если есть вход — отдаём трейдеру
                    if sig.side in ("Buy", "Sell"):
                        payload: Dict[str, Any] = {
                            "close": float(df.iloc[-1]["close"]),
                            "price": float(df.iloc[-1]["close"]),
                            "sl": sig.sl,
                            "tp": sig.tp,
                            "atr": sig.atr,
                            "ts_ms": sig.ts_ms or ts,
                        }
                        await trader.open_market(sig.side, payload)
            await asyncio.sleep(poll_sec)
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"[STRAT_LOOP] {e}")
            await asyncio.sleep(1.0)

# ---------------- app wiring ----------------

async def main():
    # env
    TG_TOKEN = os.getenv("TG_BOT_TOKEN", "").strip()
    TG_CHAT = os.getenv("TG_CHAT_ID", "").strip()  # можно не задавать — подхватится из /start
    POLL_SEC = float(os.getenv("STRAT_POLL_SEC", "1.0"))

    # клиенты/сервисы
    client = BybitClient()
    bot = TgBot(TG_TOKEN, int(TG_CHAT) if TG_CHAT else None)
    trader = Trader(client=client, notifier=bot)
    strat = StrategyEngine(trader=trader, notifier=bot)
    data = DataManager()  # WS + агрегатор + backfill

    # старт бота в фоне
    await bot.start_background()

    # старт датапайплайна
    data_task = asyncio.create_task(data.start(), name="data")

    # стратегический цикл
    strat_task = asyncio.create_task(strategy_loop(strat, trader, poll_sec=POLL_SEC), name="strategy")

    log.info("🚀 Bot online. Отправь /start в Telegram для кнопок.")

    # graceful shutdown по сигналам
    stop_event = asyncio.Event()

    def _on_signal(sig_name):
        log.info(f"[SHUTDOWN] {sig_name}")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, _on_signal, s.name)
        except NotImplementedError:
            # Windows
            pass

    await stop_event.wait()

    # останавливаем таски
    for t in (strat_task, data_task):
        if not t.done():
            t.cancel()
    try:
        await asyncio.gather(strat_task, data_task, return_exceptions=True)
    finally:
        try:
            await data.stop()
        except Exception:
            pass
        try:
            await bot.shutdown()
        except Exception:
            pass
        log.info("👋 bye")

if __name__ == "__main__":
    asyncio.run(main())
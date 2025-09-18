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
from config import TELEGRAM_TOKEN_LOCAL, TELEGRAM_TOKEN_SERVER, TELEGRAM_CHAT_ID

def setup_logging():
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
    )
    for noisy in ("aiosqlite", "httpcore", "httpx", "websockets", "asyncio", "aiogram"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


setup_logging()
log = logging.getLogger("MAIN")


async def strategy_loop(strat: StrategyEngine, trader: Trader, poll_sec: float = 1.0):
    last_ts: Optional[int] = None
    while True:
        try:
            df = await load_recent_1m(limit=1)
            if not df.empty:
                ts = int(df.iloc[-1]["ts_ms"])
                if last_ts is None:
                    last_ts = ts
                elif ts != last_ts:
                    sig = await strat.on_kline_closed()
                    last_ts = ts
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


async def main():
    host_role = os.getenv("HOST_ROLE", "local").strip().lower()
    log.info(f"[MAIN] host_role={host_role} token={'SERVER' if host_role=='server' else 'LOCAL'} chat_id={'set' if TELEGRAM_CHAT_ID else 'not set'}")
    if host_role == "server":
        TG_TOKEN = TELEGRAM_TOKEN_SERVER.strip()
    else:
        TG_TOKEN = TELEGRAM_TOKEN_LOCAL.strip()

    if not TG_TOKEN or ":" not in TG_TOKEN:
        raise RuntimeError("[MAIN][ERR] Telegram token not set or invalid")

    TG_CHAT = TELEGRAM_CHAT_ID.strip()
    POLL_SEC = float(os.getenv("STRAT_POLL_SEC", "1.0"))

    client = BybitClient()
    bot = TgBot(TG_TOKEN, int(TG_CHAT) if TG_CHAT else None)
    trader = Trader(client=client, notifier=bot)
    strat = StrategyEngine(notifier=bot)
    data = DataManager()

    await bot.start_background()
    if TG_CHAT:
        await bot.announce_online()

    data_task = asyncio.create_task(data.start(), name="data")
    strat_task = asyncio.create_task(strategy_loop(strat, trader, poll_sec=POLL_SEC), name="strategy")

    log.info("🚀 Bot online. В Telegram отправь /start, чтобы увидеть кнопки.")

    stop_event = asyncio.Event()

    def _on_signal(sig_name):
        log.info(f"[SHUTDOWN] {sig_name}")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, _on_signal, s.name)
        except NotImplementedError:
            pass

    await stop_event.wait()

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
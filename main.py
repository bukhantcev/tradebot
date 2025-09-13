import asyncio
import signal
import contextlib
from log import log
from bot import dp, bot
from trader import init_trader
from config import CFG

async def main():
    log.info("=== Scalper bot starting ===")
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    tr = init_trader(bot)

    def _stop(*_):
        log.info("[LIFECYCLE] SIGINT/SIGTERM received -> stopping...")
        stop_event.set()
        with contextlib.suppress(RuntimeError):
            asyncio.create_task(tr.stop())

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, _stop)
        except NotImplementedError:
            signal.signal(s, lambda *_: _stop())

    try:
        await dp.start_polling(bot, handle_signals=False, stop_event=stop_event)
    finally:
        await tr.stop()
        await bot.session.close()
        log.info("[LIFECYCLE] shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
import asyncio
import signal
from log import log
from bot import dp, bot
from trader import trader

async def main():
    log.info("[LIFECYCLE] trader loop starting")
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _stop():
        log.info("[LIFECYCLE] stopping")
        stop_event.set()

    loop.add_signal_handler(signal.SIGINT, _stop)
    loop.add_signal_handler(signal.SIGTERM, _stop)

    asyncio.create_task(trader.start())
    await dp.start_polling(bot, handle_signals=False, stop_event=stop_event)
    await trader.stop()
    await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())
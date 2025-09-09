from bot.handlers.handlers import setup_routes, shutdown_cleanup
import asyncio
import logging
from aiogram import Bot, Dispatcher, __version__ as aiogram_version
from config import load_config
from logger import logger

async def main():

    cfg = load_config()
    logger.info("[MAIN] Bot starting…")
    logger.info("[MAIN] aiogram=%s", aiogram_version)

    # Basic config sanity
    if not cfg.get("telegram_token"):
        logger.error("[MAIN] TELEGRAM_BOT_TOKEN is empty. Check your .env")
        return

    bot = Bot(token=cfg["telegram_token"])
    dp = Dispatcher()

    # Ensure polling gets updates if a webhook was set earlier
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        logger.info("[MAIN] Webhook deleted (drop_pending_updates=True)")
    except Exception:
        logger.exception("[MAIN] Failed to delete webhook")

    # Try to validate token early
    try:
        me = await bot.get_me()
        logger.info("[MAIN] Authorized as @%s id=%s", me.username, me.id)
    except Exception:
        logger.exception("[MAIN] Telegram token validation failed (get_me)")
        return

    setup_routes(dp, cfg)

    try:
        await dp.start_polling(bot)
    except Exception:
        logger.exception("[MAIN] start_polling failed")
    finally:
        try:
            await shutdown_cleanup(bot)
        except Exception:
            logger.exception("[MAIN] shutdown_cleanup error")
        await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())
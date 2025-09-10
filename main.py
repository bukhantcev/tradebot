import asyncio
from config import CFG
from log import log
from bot import dp, bot, trader

async def main():
    log("=== Scalper bot starting ===")

    if not CFG.tg_token:
        log("[FATAL] TG_BOT_TOKEN не задан — бот не сможет запуститься.")
        return
    if CFG.tg_chat_id == 0:
        log("[WARN] TG_CHAT_ID=0 — привяжу автоматически при /start_trade.")
    if not CFG.bybit_key or not CFG.bybit_secret:
        log("[WARN] BYBIT ключи не заданы — /balance и торговля не будут работать до их установки.")

    # прединициализация фильтров/плеча
    try:
        await trader.load_filters_and_set_leverage()
    except Exception as e:
        await trader.notify(f"⚠️ Ошибка инициализации Bybit: {e}")

    await dp.start_polling(bot, allowed_updates=["message"])

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
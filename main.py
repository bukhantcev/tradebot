import asyncio
import logging
import time

from log import logger  # инициализация логов
from config import SYMBOL, RISK_PCT
from data import DataManager
from strategy import StrategyEngine
from trader import Trader
from llm import LLMAdvisor
from bot import ControlBus, build_app

async def llm_loop(bus: ControlBus, strat: StrategyEngine, advisor: LLMAdvisor):
    while True:
        try:
            kpis = {"cooldown": 180, "risk": strat.risk_pct}
            market = {"symbol": SYMBOL}
            rec = advisor.advise(kpis, market)
            if "mode" in rec:
                strat.set_mode(rec["mode"])
                bus.mode = rec["mode"]
            if "risk_pct" in rec:
                strat.risk_pct = rec["risk_pct"]
                bus.risk_pct = rec["risk_pct"]
            # SL/TP множители можно пробросить в стратегию в следующих ревизиях
        except Exception as e:
            logger.warning(f"[LLM_LOOP] {e}")
        await asyncio.sleep(600)  # 10 минут

async def tg_commands_loop(bus: ControlBus, strat: StrategyEngine, trader: Trader, data: DataManager):
    while True:
        cmd = await bus.get()
        if cmd["cmd"] == "start":
            bus.started = True
            bus.status = "running"
        elif cmd["cmd"] == "stop":
            bus.status = "stopping"
            trader.close_all()
            bus.started = False
            bus.status = "stopped"
        elif cmd["cmd"] == "close_all":
            trader.close_all()
        elif cmd["cmd"] == "mode":
            strat.set_mode(cmd["value"])
            bus.mode = cmd["value"]
        elif cmd["cmd"] == "risk":
            strat.risk_pct = cmd["value"]
            bus.risk_pct = cmd["value"]

async def trading_loop(bus: ControlBus, strat: StrategyEngine, trader: Trader):
    while True:
        try:
            if not bus.started:
                await asyncio.sleep(1.0)
                continue
            sig = await strat.on_kline_closed()
            if sig.side:
                logger.info(f"[SIGNAL] {sig.side} {sig.reason} SL={sig.sl:.2f} TP={sig.tp:.2f} ATR={sig.atr:.2f}")
                trader.place_trade(sig.side, sig.sl, sig.tp)
        except Exception as e:
            logger.error(f"[TRADING_LOOP] {e}")
        await asyncio.sleep(2.0)

async def main():
    bus = ControlBus()
    bot, dp = build_app(bus)

    data = DataManager(symbol=SYMBOL)
    strat = StrategyEngine(risk_pct=RISK_PCT, symbol=SYMBOL)
    trader = Trader(symbol=SYMBOL, risk_pct=RISK_PCT)
    advisor = LLMAdvisor()

    # Запускаем параллельно:
    tasks = [
        asyncio.create_task(data.start(), name="data"),
        asyncio.create_task(trading_loop(bus, strat, trader), name="trading"),
        asyncio.create_task(llm_loop(bus, strat, advisor), name="llm"),
        asyncio.create_task(dp.start_polling(bot), name="tg"),
        asyncio.create_task(tg_commands_loop(bus, strat, trader, data), name="cmds"),
    ]
    logger.info("[MAIN] started")

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        await data.stop()
        logger.info("[MAIN] stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
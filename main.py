import asyncio
import logging

from log import logger  # init logs
from config import SYMBOL, RISK_PCT
from data import DataManager
from strategy import StrategyEngine
from trader import Trader
from llm import LLMAdvisor
from bot import ControlBus, build_app

log = logging.getLogger("MAIN")

async def llm_loop(bus: ControlBus, strat: StrategyEngine, advisor: LLMAdvisor):
    asyncio.current_task().set_name("llm")
    while True:
        try:
            kpis = {"cooldown": 180, "risk": strat.risk_pct}
            market = {"symbol": SYMBOL}
            rec = advisor.advise(kpis, market)
            if "mode" in rec:
                strat.set_mode(rec["mode"]); bus.mode = rec["mode"]
            if "risk_pct" in rec:
                strat.risk_pct = rec["risk_pct"]; bus.risk_pct = rec["risk_pct"]
        except Exception as e:
            log.warning(f"[LLM_LOOP] {e}")
        await asyncio.sleep(600)

async def tg_commands_loop(bus: ControlBus, strat: StrategyEngine, trader: Trader, data: DataManager):
    asyncio.current_task().set_name("tg-cmds")
    while True:
        cmd = await bus.get()
        if cmd["cmd"] == "start":
            bus.started = True; bus.status = "running"; log.info("[CMD] start")
        elif cmd["cmd"] == "stop":
            log.info("[CMD] stop -> close_all + halt")
            trader.close_all()
            bus.started = False; bus.status = "stopped"
        elif cmd["cmd"] == "close_all":
            log.info("[CMD] close_all")
            trader.close_all()
        elif cmd["cmd"] == "mode":
            strat.set_mode(cmd["value"]); bus.mode = cmd["value"]; log.info(f"[CMD] mode={cmd['value']}")
        elif cmd["cmd"] == "risk":
            strat.risk_pct = cmd["value"]; bus.risk_pct = cmd["value"]; log.info(f"[CMD] risk={cmd['value']}%")

async def trading_loop(bus: ControlBus, strat: StrategyEngine, trader: Trader):
    asyncio.current_task().set_name("trading")
    while True:
        try:
            if not bus.started:
                await asyncio.sleep(1.0)
                continue
            sig = await strat.on_kline_closed()
            if sig.side:
                log.info(f"[TRADE] side={sig.side} reason={sig.reason} SL={sig.sl:.2f} TP={sig.tp:.2f} ATR={sig.atr:.2f}")
                trader.place_trade(sig.side, sig.sl, sig.tp)
            else:
                log.debug(f"[SKIP] {sig.reason}")
        except Exception as e:
            log.error(f"[TRADING_LOOP] {e}")
        await asyncio.sleep(2.0)

async def main():
    asyncio.current_task().set_name("main")
    bus = ControlBus()
    bot, dp = build_app(bus)

    data = DataManager(symbol=SYMBOL)
    strat = StrategyEngine(risk_pct=RISK_PCT, symbol=SYMBOL)
    trader = Trader(symbol=SYMBOL, risk_pct=RISK_PCT)
    advisor = LLMAdvisor()

    tasks = [
        asyncio.create_task(data.start(), name="data"),
        asyncio.create_task(trading_loop(bus, strat, trader), name="trading"),
        asyncio.create_task(llm_loop(bus, strat, advisor), name="llm"),
        asyncio.create_task(dp.start_polling(bot), name="tg"),
        asyncio.create_task(tg_commands_loop(bus, strat, trader, data), name="tg-cmds"),
    ]
    log.info("[MAIN] started")
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        await data.stop()
        log.info("[MAIN] stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
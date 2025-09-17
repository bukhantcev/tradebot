import os, time, asyncio, logging, html, httpx
from typing import List, Dict, Any
from config import BYBIT_SYMBOL, BYBIT_LEVERAGE, CATEGORY, DATA_DIR, DUMP_DIR, BYBIT_TESTNET
from bybit_client import BybitClient
from ws import PublicWS
from params_store import load_params
from hourly_dump import HourlyDump, send_to_openai_and_update_params
from trader import Trader
from strategy import decide
from tg_bot import TgBot

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.INFO)
    logging.getLogger("websockets").setLevel(logging.INFO)
    logging.getLogger("aiogram").setLevel(logging.INFO)

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DUMP_DIR, exist_ok=True)

def candle_from_ws(itm: Dict[str, Any]) -> Dict[str, Any]:
    def f(x):
        try: return float(x)
        except: return 0.0
    return {
        "ts": int(itm.get("start") or 0),
        "open": f(itm.get("open")), "high": f(itm.get("high")),
        "low": f(itm.get("low")), "close": f(itm.get("close")),
        "volume": f(itm.get("volume")), "confirm": bool(itm.get("confirm", False))
    }

class Controller:
    def __init__(self, notifier: TgBot | None):
        self.client = BybitClient()
        self.symbol = BYBIT_SYMBOL
        self.params = load_params()
        self.dump = HourlyDump()
        self.candles_1: List[dict] = []
        self.candles_5: List[dict] = []
        self.running = False
        self.ws = PublicWS(self.symbol, self.on_kline, intervals=("1","5"))
        self.trader: Trader | None = None
        self.tick_size = 0.1
        self.qty_step = 0.001
        self._hour_start = int(time.time() // 3600 * 3600) * 1000
        self.notifier = notifier
        self.log = logging.getLogger("controller")

    async def request_ai_params_from_latest_dump(self):
        """On startup/restart: find the latest dump_*.json and request fresh params from OpenAI.
        If nothing found, just log and continue.
        """
        try:
            files = [f for f in os.listdir(DUMP_DIR) if f.endswith('.json')]
            if not files:
                self.log.info("No dump_*.json found for bootstrap; skipping initial OpenAI request")
                return
            paths = [os.path.join(DUMP_DIR, f) for f in files]
            latest = max(paths, key=lambda p: os.path.getmtime(p))
            if self.notifier:
                await self.notifier.notify("🧠 Старт: обновляю параметры из последнего дампа через ИИ…")
            await send_to_openai_and_update_params(latest, notifier=self.notifier)
            # ИИ мог обновить params.json — перечитаем
            self.params = load_params()
            self.log.info("Startup params refreshed from AI via latest dump: %s", os.path.basename(latest))
        except Exception:
            self.log.exception("Failed to bootstrap params from latest dump")

    async def setup_instrument(self):
        info = await self.client.instruments_info(CATEGORY, self.symbol)
        lst = info.get("result", {}).get("list") or []
        if lst:
            priceFilter = lst[0].get("priceFilter", {})
            lot = lst[0].get("lotSizeFilter", {})
            self.tick_size = float(priceFilter.get("tickSize", "0.1"))
            self.qty_step = float(lot.get("qtyStep", "0.001"))
        self.log.debug(f"Instrument tick_size={self.tick_size} qty_step={self.qty_step}")

    def on_kline(self, interval: str, itm: Dict[str, Any]):
        c = candle_from_ws(itm)
        if interval == "1":
            if not self.candles_1 or c["ts"] >= (self.candles_1[-1]["ts"] or 0):
                self.candles_1.append(c)
                if c.get("confirm"):
                    self.dump.add_candle("1", c)
                    self.log.debug(f"[WS] 1m closed ts={c['ts']} close={c['close']}")
        else:
            if not self.candles_5 or c["ts"] >= (self.candles_5[-1]["ts"] or 0):
                self.candles_5.append(c)
                if c.get("confirm"):
                    self.dump.add_candle("5", c)
                    self.log.debug(f"[WS] 5m closed ts={c['ts']} close={c['close']}")

    async def on_kline_tick(self):
        if not self.running or not self.candles_1 or not self.candles_5:
            return
        last = self.candles_1[-1]
        if not last.get("confirm"):
            return
        sig = decide(self.candles_1, self.candles_5, self.params)
        self.dump.add_signal({"t": int(time.time()*1000), **sig})
        if sig["decision"] in ("long","short"):
            price = last["close"]
            if self.notifier:
                await self.notifier.notify(f"📊 Сигнал: {sig['decision']} @ {price}")
            qty = max(self.params["size_usdt"] / max(price, 1e-9), self.qty_step)
            tp = sig["tp"]; sl = sig["sl"]
            side = "Buy" if sig["decision"]=="long" else "Sell"
            await self.trader.open_market(side, qty, tp, sl)
            self.dump.add_trade({"dir": sig["decision"], "opened_at": int(time.time()*1000),
                                 "open_price": price, "tp": tp, "sl": sl, "reason": sig.get("reason","")})

    async def start(self):
        if self.running: return
        await self.setup_instrument()
        self.trader = Trader(self.client, self.symbol, self.tick_size, self.qty_step, notifier=self.notifier)
        await self.trader.ensure_leverage(BYBIT_LEVERAGE)
        self.running = True
        if self.notifier:
            await self.notifier.notify(f"▶️ Старт бота для {self.symbol} ({'testnet' if BYBIT_TESTNET else 'main'})")
        await self.request_ai_params_from_latest_dump()

    async def stop(self):
        self.running = False
        if self.trader:
            await self.trader.close_all()
        if self.notifier:
            await self.notifier.notify("⏹ Бот остановлен")

    async def status(self) -> str:
        pos = await self.client.position_list(CATEGORY, self.symbol)
        return f"Account: {'testnet' if BYBIT_TESTNET else 'main'}\nSymbol: {self.symbol}\nTickSize: {self.tick_size} QtyStep: {self.qty_step}\nPositions: {pos}"

    async def short_balance(self) -> str:
        try:
            bal = await self.client.wallet_balance()
            lst = bal.get("result", {}).get("list", [])
            # Берём USDT (первый кошелёк)
            usdt = 0.0
            if lst:
                for acc in lst:
                    for c in acc.get("coin", []):
                        if c.get("coin") == "USDT":
                            usdt = float(c.get("equity", "0"))
                            break
            return f"Баланс: {usdt:.2f} USDT"
        except Exception:
            return "Баланс: н/д"

    async def dump_now(self):
        self.dump.set_params_used(self.params)
        self.dump.set_meta(self.symbol, "testnet" if BYBIT_TESTNET else "main", self._hour_start, int(time.time()*1000))
        path = self.dump.flush_to_file()
        return path

    async def restart(self):
        await self.stop()
        await asyncio.sleep(0.5)
        await self.start()
        if self.notifier:
            await self.notifier.notify("♻️ Бот перезапущен")

    async def hourly_job(self):
        while True:
            await asyncio.sleep(5)
            now_ms = int(time.time()*1000)
            if now_ms - self._hour_start >= 3600*1000:
                try:
                    if self.notifier:
                        await self.notifier.notify("⏱ Новый час: формируем дамп и шлём в ИИ")
                    path = await self.dump_now()
                    await send_to_openai_and_update_params(path, notifier=self.notifier)
                    # перечитать — ИИ мог обновить params.json
                    self.params = load_params()
                    self.dump = HourlyDump()
                    self._hour_start = int(time.time() // 3600 * 3600) * 1000
                except (httpx.HTTPStatusError, httpx.RequestError) as e:
                    self.log.error("hourly_job OpenAI error: %s", str(e))
                    if self.notifier:
                        await self.notifier.notify(
                            f"⚠️ Hourly: сбой запроса к OpenAI, работаем дальше.\n<pre>{html.escape(str(e))}</pre>"
                        )
                    # не прерываем цикл — просто пропускаем этот час
                except Exception as e:
                    self.log.exception("hourly_job unexpected error")
                    if self.notifier:
                        await self.notifier.notify(
                            f"⚠️ Hourly: непредвиденная ошибка, цикл не остановлен.\n<pre>{html.escape(str(e))}</pre>"
                        )

    async def minute_job(self):
        while True:
            try:
                await self.on_kline_tick()
            except Exception:
                logging.getLogger("controller").exception("minute_job error")
            await asyncio.sleep(2)

async def main():
    setup_logging()
    bot = TgBot(None)
    ctl = Controller(notifier=bot)
    bot.controller = ctl
    await ctl.ws.connect()

    await asyncio.gather(
        bot.start(),
        ctl.minute_job(),
        ctl.hourly_job()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
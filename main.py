import os, time, asyncio
from typing import List, Dict, Any
from config import BYBIT_SYMBOL, BYBIT_LEVERAGE, CATEGORY, DATA_DIR, DUMP_DIR, BYBIT_TESTNET
from bybit_client import BybitClient
from ws import PublicWS
from params_store import load_params
from hourly_dump import HourlyDump, send_to_openai_and_update_params
from trader import Trader
from strategy import decide
from tg_bot import TgBot

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DUMP_DIR, exist_ok=True)

def candle_from_ws(itm: Dict[str, Any]) -> Dict[str, Any]:
    # WS kline item fields (unified): { "start": "...", "end": "...", "open": "...", "high": "...", "low": "...", "close": "...", "volume": "...", "confirm": true/false }
    def f(x): return float(x)
    return {
        "ts": int(itm.get("start") or 0),
        "open": f(itm.get("open", 0)), "high": f(itm.get("high", 0)),
        "low": f(itm.get("low", 0)), "close": f(itm.get("close", 0)),
        "volume": f(itm.get("volume", 0)), "confirm": bool(itm.get("confirm", False))
    }

class Controller:
    def __init__(self, notifier=None):
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

    async def setup_instrument(self):
        info = await self.client.instruments_info(CATEGORY, self.symbol)
        lst = info.get("result", {}).get("list") or []
        if lst:
            priceFilter = lst[0].get("priceFilter", {})
            lot = lst[0].get("lotSizeFilter", {})
            self.tick_size = float(priceFilter.get("tickSize", "0.1"))
            self.qty_step = float(lot.get("qtyStep", "0.001"))

    async def on_kline_tick(self):
        # Каждую новую закрытую минуту — решение
        if len(self.candles_1) < 100 or len(self.candles_5) < 30:
            return
        if not self.running:
            return
        last = self.candles_1[-1]
        if not last.get("confirm"):  # ждём закрытия свечи
            return
        sig = decide(self.candles_1, self.candles_5, self.params)
        self.dump.add_signal({"t": int(time.time()*1000), **sig})
        if sig["decision"] in ("long","short"):
            # размер по USDT -> qty = size_usdt / price
            price = last["close"]
            if self.notifier:
                await self.notifier.notify(f"📊 Сигнал: {sig['decision']} @ {price}")
            qty = max(self.params["size_usdt"] / price, self.qty_step)
            # Биржевые TP/SL по цене из сигнала
            tp = sig["tp"]; sl = sig["sl"]
            side = "Buy" if sig["decision"]=="long" else "Sell"
            r = await self.trader.open_market(side, qty, tp, sl)
            self.dump.add_trade({"dir": sig["decision"], "opened_at": int(time.time()*1000),
                                 "open_price": price, "tp": tp, "sl": sl, "reason": sig.get("reason","")})

    def on_kline(self, interval: str, itm: Dict[str, Any]):
        c = candle_from_ws(itm)
        if interval == "1":
            if not self.candles_1 or c["ts"] >= (self.candles_1[-1]["ts"] or 0):
                self.candles_1.append(c)
        else:
            if not self.candles_5 or c["ts"] >= (self.candles_5[-1]["ts"] or 0):
                self.candles_5.append(c)

    async def start(self):
        if self.running: return
        await self.setup_instrument()
        self.trader = Trader(self.client, self.symbol, self.tick_size, self.qty_step, notifier=self.notifier)
        await self.trader.ensure_leverage(BYBIT_LEVERAGE)
        self.running = True
        if self.notifier:
            await self.notifier.notify(f"▶️ Старт бота для {self.symbol} ({'testnet' if BYBIT_TESTNET else 'main'})")

    async def stop(self):
        self.running = False
        if self.trader:
            await self.trader.close_all()
        if self.notifier:
            await self.notifier.notify("⏹ Бот остановлен")

    async def status(self) -> str:
        pos = await self.client.position_list(CATEGORY, self.symbol)
        bal = await self.client.wallet_balance()
        return f"Account: {'testnet' if BYBIT_TESTNET else 'main'}\nSymbol: {self.symbol}\nTickSize: {self.tick_size} QtyStep: {self.qty_step}\nPositions: {pos}\nBalance: {bal}"

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
                # закрываем час, отправляем в OpenAI, обновляем параметры, удаляем дамп
                if self.notifier:
                    await self.notifier.notify("⏱ Новый час: формируем дамп и шлём в ИИ")
                path = await self.dump_now()
                await send_to_openai_and_update_params(path)
                self.params = load_params()  # подменяем сразу
                self.dump = HourlyDump()
                self._hour_start = int(time.time() // 3600 * 3600) * 1000

    async def minute_job(self):
        # опрашиваем раз в ~2с и пытаемся отреагировать на закрытие свечи
        while True:
            try:
                await self.on_kline_tick()
            except Exception:
                pass
            await asyncio.sleep(2)

async def main():
    bot = TgBot(None)
    ctl = Controller(notifier=bot)
    bot.controller = ctl
    await ctl.ws.connect()
    # параллельно: TG, мин. цикл, часовой цикл
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
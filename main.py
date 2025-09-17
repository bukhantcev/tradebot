import os, time, asyncio, logging, html, httpx
from typing import List, Dict, Any
from config import BYBIT_SYMBOL, BYBIT_LEVERAGE, CATEGORY, DATA_DIR, DUMP_DIR, BYBIT_TESTNET
from bybit_client import BybitClient
from ws import PublicWS, preload_klines
from params_store import load_params, save_params
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
        self.ws = PublicWS(self.symbol, self.on_kline, intervals=("1","5"), deliver_only_confirm=True)
        self.trader: Trader | None = None
        self.tick_size = 0.1
        self.qty_step = 0.001
        self._hour_start = int(time.time() // 3600 * 3600) * 1000
        self.notifier = notifier
        self.log = logging.getLogger("controller")

        # Anti-churn state
        self.last_processed_ts: int = 0  # last closed 1m candle we acted on
        self.prev_decision: str = "hold"  # last non-hold decision
        self.last_entry_bar_ts: int = 0  # ts of the bar when we last opened a position
        self.bar_1m_ms: int = 60_000

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
        if not c.get("confirm"):
            self.log.debug(f"[WS_TICK] {interval}m forming ts={c['ts']} close={c['close']}")
        if interval == "1":
            if not self.candles_1:
                self.candles_1.append(c)
            else:
                last_ts = self.candles_1[-1]["ts"]
                if c["ts"] > last_ts:
                    self.candles_1.append(c)
                elif c["ts"] == last_ts:
                    # update in place for forming candle
                    self.candles_1[-1] = c
            if c.get("confirm"):
                # only once per closed bar
                if not self.dump or (self.candles_1 and c["ts"] == self.candles_1[-1]["ts"]):
                    self.dump.add_candle("1", c)
                    self.log.debug(f"[WS] 1m closed ts={c['ts']} close={c['close']}")
        else:
            if not self.candles_5:
                self.candles_5.append(c)
            else:
                last_ts5 = self.candles_5[-1]["ts"]
                if c["ts"] > last_ts5:
                    self.candles_5.append(c)
                elif c["ts"] == last_ts5:
                    self.candles_5[-1] = c
            if c.get("confirm"):
                if not self.dump or (self.candles_5 and c["ts"] == self.candles_5[-1]["ts"]):
                    self.dump.add_candle("5", c)
                    self.log.debug(f"[WS] 5m closed ts={c['ts']} close={c['close']}")

    async def on_kline_tick(self):
        if not self.running or not self.candles_1 or not self.candles_5:
            return
        last = self.candles_1[-1]
        if not last.get("confirm"):
            return
        # act only once per closed bar
        if last["ts"] == self.last_processed_ts:
            self.log.debug(f"[SKIP] already processed bar ts={last['ts']}")
            return

        # read cooldown from params (defaults to 3 bars)
        min_bars_between = int(self.params.get("min_bars_between_trades", 3))
        if self.last_entry_bar_ts and (last["ts"] - self.last_entry_bar_ts) < min_bars_between * self.bar_1m_ms:
            remain = (min_bars_between * self.bar_1m_ms - (last["ts"] - self.last_entry_bar_ts)) // self.bar_1m_ms
            self.log.info(f"[SKIP] cooldown {remain} bars remaining")
            self.last_processed_ts = last["ts"]
            return

        sig = decide(self.candles_1, self.candles_5, self.params)
        self.dump.add_signal({"t": int(time.time()*1000), **sig})

        # process only on decision change and only for entries
        if sig["decision"] in ("long", "short"):
            if sig["decision"] == self.prev_decision:
                self.log.info(f"[SKIP] same decision as previous: {sig['decision']}")
                self.last_processed_ts = last["ts"]
                return
            price = last["close"]
            if self.notifier:
                await self.notifier.notify(f"📊 Сигнал: {sig['decision']} @ {price}")
            qty = max(self.params["size_usdt"] / max(price, 1e-9), self.qty_step)
            tp = sig["tp"]; sl = sig["sl"]
            side = "Buy" if sig["decision"] == "long" else "Sell"
            await self.trader.open_market(side, qty, tp, sl)
            self.dump.add_trade({"dir": sig["decision"], "opened_at": int(time.time()*1000),
                                 "open_price": price, "tp": tp, "sl": sl, "reason": sig.get("reason", "")})
            # update anti-churn state
            self.prev_decision = sig["decision"]
            self.last_entry_bar_ts = last["ts"]
        else:
            # reset prev decision on hold
            self.prev_decision = "hold"

        self.last_processed_ts = last["ts"]

    async def preload_history(self):
        """Load enough historical candles on startup to satisfy strategy warmup, then continue."""
        try:
            ema_slow = self.params["indicators"]["ema_slow"]
            required_m1 = max(50, ema_slow) + 1
            required_m5 = ema_slow + 1
            before_m1 = len(self.candles_1)
            before_m5 = len(self.candles_5)
            self.log.info(f"[PRELOAD] requesting klines: m1={required_m1} m5={required_m5}")
            self.log.info("[PRELOAD] start REST fetch for historical klines")
            await preload_klines(self.symbol, self.on_kline, {"1": required_m1, "5": required_m5}, testnet=BYBIT_TESTNET, category=CATEGORY)
            after_m1 = len(self.candles_1)
            after_m5 = len(self.candles_5)
            if after_m1 < required_m1 or after_m5 < required_m5:
                self.log.warning(f"[PRELOAD] insufficient candles after REST: m1={after_m1}/{required_m1} m5={after_m5}/{required_m5}")
            self.log.info(f"[PRELOAD] done REST fetch; added m1={after_m1-before_m1} m5={after_m5-before_m5}")
        except Exception:
            self.log.exception("[PRELOAD] failed (will continue with WS only)")

    async def start(self):
        if self.running: return
        await self.setup_instrument()
        self.trader = Trader(self.client, self.symbol, self.tick_size, self.qty_step, notifier=self.notifier)
        await self.trader.ensure_leverage(BYBIT_LEVERAGE)
        self.running = True
        if self.notifier:
            await self.notifier.notify(f"▶️ Старт бота для {self.symbol} ({'testnet' if BYBIT_TESTNET else 'main'})")

        # 1) Сначала подтягиваем историю для прогрева индикаторов
        await self.preload_history()

        # 2) Делаем моментальный дамп из прогретых данных и отправляем в ИИ,
        # чтобы стартовые параметры были актуальными прямо на запуске
        try:
            bootstrap_dump = await self.dump_now()
            await send_to_openai_and_update_params(bootstrap_dump, notifier=self.notifier)
            self.params = load_params()
            self.log.info("[BOOTSTRAP] AI params refreshed from preloaded dump")
        except Exception:
            self.log.exception("[BOOTSTRAP] failed to refresh params from AI; continue with existing params")

        # 3) После этого подключаем вебсокет и идём в торговый цикл
        await self.ws.connect()
        self.log.info("[WS] connected after preload")

    async def stop(self):
        self.running = False
        if self.trader:
            await self.trader.close_all()
        # Persist current (possibly AI-updated) params as defaults for next runs
        try:
            save_params(self.params)
            self.log.info("[PARAMS] persisted on stop (set as new defaults)")
        except Exception:
            self.log.exception("[PARAMS] failed to persist on stop")
        if self.notifier:
            await self.notifier.notify("⏹ Бот остановлен (параметры сохранены как дефолтные)")

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
        """Build a start_dump from current in-memory candles + current params (fresh HourlyDump)."""
        hd = HourlyDump()
        # meta window by 1m candles if available
        if self.candles_1:
            start_ts = int(self.candles_1[0]["ts"]) if "ts" in self.candles_1[0] else int(time.time()*1000)
            end_ts = int(self.candles_1[-1]["ts"]) if "ts" in self.candles_1[-1] else int(time.time()*1000)
        else:
            start_ts = end_ts = int(time.time()*1000)
        hd.set_meta(self.symbol, "testnet" if BYBIT_TESTNET else "main", start_ts, end_ts)
        # inject candles from memory (preload)
        for c in self.candles_1:
            hd.add_candle("1", c)
        for c in self.candles_5:
            hd.add_candle("5", c)
        # params currently loaded (defaults at startup)
        hd.set_params_used(self.params)
        path = hd.flush_to_file()
        self.log.info(f"[BOOTSTRAP] start_dump written: {path}")
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
    await ctl.start()

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
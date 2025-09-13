import asyncio
from typing import Optional
from log import log
from config import CFG
from db import db
from bybit_client import bybit
import strategy as strat

from aiogram import Bot

class Trader:
    def __init__(self, bot: Bot):
        self.bot = bot
        self.running = False
        self.tasks: list[asyncio.Task] = []
        self.last_1m_close_ts: int | None = None
        self.last_5m_close_ts: int | None = None
        self.tick_size: float = 0.1
        self.s5: float | None = None
        self.r5: float | None = None


    async def preload(self):
        # meta
        meta = await bybit.instruments_info(CFG.symbol)
        try:
            price_filter = meta["result"]["list"][0]["priceFilter"]
            self.tick_size = float(price_filter["tickSize"])
        except Exception:
            self.tick_size = 0.1
        log.info("[META] tickSize=%s", self.tick_size)

        # set leverage (ignore "not modified")
        await bybit.set_leverage(CFG.leverage)

        # preload candles
        k5 = await bybit.kline("5", 200)
        k1 = await bybit.kline("1", 300)
        rows5 = []
        for r in k5["result"]["list"]:
            ts, o,h,l,c,v = int(r[0]), float(r[1]),float(r[2]),float(r[3]),float(r[4]),float(r[5])
            rows5.append((ts,o,h,l,c,v))
        rows1 = []
        for r in k1["result"]["list"]:
            ts, o,h,l,c,v = int(r[0]), float(r[1]),float(r[2]),float(r[3]),float(r[4]),float(r[5])
            rows1.append((ts,o,h,l,c,v))
        await db.upsert_candles("5", rows5)
        await db.upsert_candles("1", rows1)
        await db.trim_to("5", 200)
        await db.trim_to("1", 300)
        if rows1:
            self.last_1m_close_ts = rows1[-1][0]
        if rows5:
            self.last_5m_close_ts = rows5[-1][0]
        log.info("[PRELOAD] 5m=%s 1m=%s", len(rows5), len(rows1))

        # initialize 5m SR once after preload
        candles5_for_sr = await db.fetch_last_n("5", 200)
        try:
            s5_init, r5_init, tick_init = strat.calc_sr_5m(candles5_for_sr, CFG.sr5_left, CFG.sr5_right)
            self.s5, self.r5 = s5_init, r5_init
            if tick_init:
                self.tick_size = tick_init
            log.info("[SR-5m][INIT] support=%s resistance=%s tick=%s", self.s5, self.r5, self.tick_size)
        except Exception as e:
            log.warning("[SR-5m][INIT] failed: %s", e)

    async def _candles_updater(self):
        """
        Каждые ~10с обновляем 1m, каждые ~60с обновляем 5m. Поддерживаем объём 200/300.
        """
        try:
            while self.running:
                # 1m
                k1 = await bybit.kline("1", 5)
                rows1 = []
                for r in k1["result"]["list"]:
                    ts, o,h,l,c,v = int(r[0]), float(r[1]),float(r[2]),float(r[3]),float(r[4]),float(r[5])
                    rows1.append((ts,o,h,l,c,v))
                await db.upsert_candles("1", rows1)
                await db.trim_to("1", 300)

                # 5m каждую минуту
                if int(asyncio.get_event_loop().time()) % 60 < 10:
                    k5 = await bybit.kline("5", 5)
                    rows5 = []
                    for r in k5["result"]["list"]:
                        ts, o,h,l,c,v = int(r[0]), float(r[1]),float(r[2]),float(r[3]),float(r[4]),float(r[5])
                        rows5.append((ts,o,h,l,c,v))
                    await db.upsert_candles("5", rows5)
                    await db.trim_to("5", 200)

                await asyncio.sleep(10)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.exception("[CANDLES] updater crashed: %s", e)

    async def _strategy_loop(self):
        """
        Ждём закрытие новой 1m — считаем сигнал.
        """
        try:
            while self.running:
                candles1 = await db.fetch_last_n("1", 3)
                # Fetch latest 5m candles for SR change detection
                candles5_latest = await db.fetch_last_n("5", 3)
                if candles1:
                    log.debug("[STRAT] fetched candles -> 1m=%s 5m_last_ts=%s", len(candles1), (candles5_latest[-1][0] if candles5_latest else None))
                    last_ts = candles1[-1][0]
                    # Если появился новый закрытый бар (в клайне Bybit list всегда закрытые бары)
                    if self.last_1m_close_ts is None:
                        self.last_1m_close_ts = last_ts
                    elif last_ts != self.last_1m_close_ts:
                        self.last_1m_close_ts = last_ts

                        # Detect new 5m close and (re)compute 5m SR ONLY on 5m close
                        if candles5_latest:
                            last_5m_ts_now = candles5_latest[-1][0]
                            if self.last_5m_close_ts is None:
                                self.last_5m_close_ts = last_5m_ts_now
                            elif last_5m_ts_now != self.last_5m_close_ts:
                                self.last_5m_close_ts = last_5m_ts_now
                                candles5_for_sr = await db.fetch_last_n("5", 200)
                                s5, r5, tick = strat.calc_sr_5m(candles5_for_sr, CFG.sr5_left, CFG.sr5_right)
                                self.s5, self.r5 = s5, r5
                                if tick:
                                    self.tick_size = tick
                                log.info("[SR-5m] 5m closed -> support=%s resistance=%s tick=%s", s5, r5, self.tick_size)

                        # Compute 1m SR on EVERY 1m close for signals
                        candles1_for_sr = await db.fetch_last_n("1", 300)
                        s1, r1 = strat.calc_sr_1m(candles1_for_sr, CFG.sr1_left, CFG.sr1_right)

                        # Minute regime
                        mode = strat.regime_1m(candles1_for_sr)
                        log.info("[REGIME-1m] mode=%s price=%s s1=%s r1=%s 1m_close_ts=%s", mode, candles1[-1][4], s1, r1, self.last_1m_close_ts)

                        # Signal on 1m close near 1m levels
                        sig = strat.signal_on_1m_close(
                            candles1_for_sr, mode, s1, r1,
                            touch_pct=CFG.sr1_touch_pct, break_pct=CFG.sr1_break_pct
                        )

                        if sig:
                            # Ensure we have 5m SR for SL; if not yet computed, compute once here
                            if self.s5 is None or self.r5 is None:
                                candles5_for_sr = await db.fetch_last_n("5", 200)
                                s5, r5, tick = strat.calc_sr_5m(candles5_for_sr, CFG.sr5_left, CFG.sr5_right)
                                self.s5, self.r5 = s5, r5
                                if tick:
                                    self.tick_size = tick
                                log.info("[SR-5m][LAZY] support=%s resistance=%s tick=%s", self.s5, self.r5, self.tick_size)

                            sl_preview = strat.compute_sl_exchange("long" if sig == "long" else "short", self.s5, self.r5, self.tick_size)
                            log.info("[SIG-1m] signal=%s -> will open side=%s sl_preview=%s (tick=%s)", sig, ("Buy" if sig == "long" else "Sell"), sl_preview, self.tick_size)
                            await self._maybe_trade(sig, self.s5, self.r5, self.tick_size)
                        else:
                            log.info("[SIG-1m] no-signal (mode=%s, close=%s)", mode, candles1[-1][4])
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.exception("[STRATEGY] crashed: %s", e)

    async def _maybe_trade(self, signal: str, support: float, resistance: float, tick: float):
        # Проверяем позицию
        pos = await bybit.positions()
        lst = pos.get("result", {}).get("list", [])
        side_now = lst and lst[0].get("side", "")
        size_now = float(lst[0].get("size", "0") or "0") if lst else 0.0
        if size_now > 0 and side_now:
            log.info("[TRADE] skip: already have open position on exchange (side=%s)", side_now)
            return

        # Открываем
        side = "long" if signal == "long" else "short"
        resp = await bybit.create_order(side, CFG.order_qty)
        log.debug("[ORDER][raw-response] %r", resp)
        if resp.get("retCode") != 0:
            log.error("[ORDER] create failed: retCode=%s retMsg=%s payload=%s",
                      resp.get("retCode"), resp.get("retMsg"), resp)
            return

        # Ставим биржевой SL сразу по 5м S/R
        # positionIdx: 0 (both), 1 (long), 2 (short). Мы ведём net-mode, используем 0.
        sl_price = strat.compute_sl_exchange(side, support, resistance, tick)
        log.info("[SL] compute -> side=%s support=%s resistance=%s tick=%s sl=%s", side, support, resistance, tick, sl_price)
        await bybit.trading_stop(0, sl_price)
        log.info("[SL] set -> side=%s stopLoss=%s", "Buy" if side=="long" else "Sell", sl_price)

        # Подтверждаем на бирже
        await asyncio.sleep(1)  # дать бирже обновить позицию
        pos2 = await bybit.positions()
        lst2 = pos2.get("result", {}).get("list", [])
        if lst2 and float(lst2[0].get("size", "0") or "0") > 0 and lst2[0].get("side"):
            entry = float(lst2[0].get("avgPrice", "0") or lst2[0].get("sessionAvgPrice", "0") or "0")
            side_ex = lst2[0]["side"]
            await self.bot.send_message(CFG.tg_admin_chat_id,
                f"🟢 Открыта позиция\n{CFG.symbol} {side_ex}\nВход: {entry}\nSL(биржевой): {sl_price}")
            log.info("[CONFIRM-OPEN] confirmed on exchange -> side=%s qty=%s entry=%s", side_ex, lst2[0].get("size"), entry)
        else:
            await self.bot.send_message(CFG.tg_admin_chat_id,
                f"🔴 Позиция НЕ подтверждена на бирже. Проверить вручную.")
            log.warning("[CONFIRM-OPEN] not found on exchange after order")

    async def start(self):
        if self.running:
            return
        self.running = True
        await bybit.open()
        await db.open()
        await self.preload()
        # Guard: ensure initial 5m SR exists before loops
        if self.s5 is None or self.r5 is None:
            candles5_for_sr = await db.fetch_last_n("5", 200)
            try:
                s5_init, r5_init, tick_init = strat.calc_sr_5m(candles5_for_sr, CFG.sr5_left, CFG.sr5_right)
                self.s5, self.r5 = s5_init, r5_init
                if tick_init:
                    self.tick_size = tick_init
                log.info("[SR-5m][BOOT] support=%s resistance=%s tick=%s", self.s5, self.r5, self.tick_size)
            except Exception as e:
                log.warning("[SR-5m][BOOT] failed: %s", e)
        await self.bot.send_message(CFG.tg_admin_chat_id, "✅ Данные подгружены (5m×200, 1m×300). Запускаю демоны/стратегию.")
        self.tasks = [
            asyncio.create_task(self._candles_updater(), name="candles_updater"),
            asyncio.create_task(self._strategy_loop(), name="strategy_loop"),
        ]
        log.info("[LIFECYCLE] trader loop started")

    async def stop(self):
        if not self.running:
            return
        log.info("[LIFECYCLE] trader stopping")
        self.running = False
        # останавливаем задачи
        for t in self.tasks:
            t.cancel()

        # корректно дождаться завершения всех задач
        if self.tasks:
            results = await asyncio.gather(*self.tasks, return_exceptions=True)
            for t, res in zip(self.tasks, results):
                name = t.get_name() if hasattr(t, "get_name") else repr(t)
                if isinstance(res, asyncio.CancelledError):
                    log.info("[TASK] %s cancelled", name)
                elif isinstance(res, Exception):
                    log.warning("[TASK] %s ended with exception: %r", name, res)
                else:
                    log.info("[TASK] %s finished cleanly", name)

        self.tasks.clear()

        # закрыть все позиции и убедиться что нет открытых
        res = await bybit.close_all()
        if res.get("closed"):
            log.info("[CLOSEALL] market-close executed")
            await asyncio.sleep(1.0)
        # verify
        pos = await bybit.positions()
        lst = pos.get("result", {}).get("list", [])
        size_now = float(lst[0].get("size", "0") or "0") if lst else 0.0
        if size_now == 0:
            await self.bot.send_message(CFG.tg_admin_chat_id, "⛔️ Бот остановлен. Открытых позиций нет.")
            log.info("[STATE] reset done")
        else:
            await self.bot.send_message(CFG.tg_admin_chat_id, "⚠️ Бот остановлен, но позиция ещё открыта! Проверь вручную.")
            log.warning("[STATE] stop but position still open")

trader: Trader | None = None
def init_trader(bot: Bot) -> Trader:
    global trader
    if trader is None:
        trader = Trader(bot)
    return trader
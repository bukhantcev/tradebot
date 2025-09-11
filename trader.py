import time
import asyncio
from typing import Optional, Tuple

from config import CFG
from log import log
from bybit_client import bybit
from sr import find_sr
from strategy import detect_mode, trade_signal


def _round_to_tick(value: float, tick: float) -> float:
    if tick <= 0:
        return value
    # округляем вниз для лонга (ниже цены) и вверх для шорта будем делать снаружи
    return (int(value / tick)) * tick


class Trader:
    def __init__(self):
        self.running: bool = False
        self.support: Optional[float] = None
        self.resistance: Optional[float] = None
        self.mode: str = "FLAT"

        # анти-спам по входам: исполняем только на смене сигнала + пауза
        self._last_exec_signal: Optional[str] = None
        self._last_exec_ts: float = 0.0

        # трекинг факта открытой позиции на бирже (для синхронизации и нотификаций о внешнем закрытии)
        self._open_sides = set()  # hedge: {'Buy','Sell'}, one-way: {'ONEWAY'}

        # настройки с дефолтами, если нет в CFG
        self._entry_cooldown_sec: float = float(getattr(CFG, "min_entry_interval_sec", 20))
        self._sl_offset_ticks: int = int(getattr(CFG, "sl_offset_ticks", 5))  # на сколько тиков за SR ставить SL
        self._sl_pct: float = float(getattr(CFG, "sl_pct", 0.0))              # если >0, стоп по % от цены
        self._notify_open = None
        self._notify_close = None

    async def start(self):
        log.info("=== Scalper bot starting ===")
        await bybit.init_symbol_meta()
        await bybit.set_leverage(CFG.leverage)
        await bybit.get_total_equity()
        self.running = True
        while self.running:
            try:
                await self.tick()
            except Exception as e:
                log.error(f"[LOOP] error={e}")
            await asyncio.sleep(CFG.loop_delay_sec)

    async def stop(self):
        self.running = False
        log.info("[LIFECYCLE] trader stopping")

    def _calc_stop_loss(self, side: str, price: float) -> Optional[float]:
        """
        Рассчитывает SL:
        - если задан CFG.sl_pct > 0: от цены входа
        - иначе: от уровней SR с отступом в _sl_offset_ticks
        Округление к шагу тика внутри.
        """
        tick = float(getattr(bybit, "tick_size", 0.1))
        off_ticks = max(1, self._sl_offset_ticks)

        if self._sl_pct and self._sl_pct > 0:
            if side == "Buy":
                raw = price * (1.0 - self._sl_pct)
                sl = _round_to_tick(raw, tick)  # округляем вниз — безопаснее для триггера
            else:
                # для шорта SL выше цены, округляем вверх
                raw = price * (1.0 + self._sl_pct)
                sl = _round_to_tick(raw + (tick - 1e-12), tick)
            return float(sl)

        # по SR
        if side == "Buy" and self.support is not None:
            raw = float(self.support) - off_ticks * tick
            sl = _round_to_tick(raw, tick)
            return float(sl)

        if side == "Sell" and self.resistance is not None:
            raw = float(self.resistance) + off_ticks * tick
            # округлить вверх к тику
            sl = _round_to_tick(raw + (tick - 1e-12), tick)
            return float(sl)

        return None

    def _can_execute(self, sig: Optional[str]) -> bool:
        if not sig or sig == "hold":
            return False
        now = time.time()
        # исполняем только если сигнал изменился или давно не торговали
        changed = sig != self._last_exec_signal
        cooled = (now - self._last_exec_ts) >= self._entry_cooldown_sec
        return changed and cooled

    def set_notifiers(self, open_cb=None, close_cb=None):
        """Inject async callbacks for Telegram notifications to avoid circular imports."""
        self._notify_open = open_cb
        self._notify_close = close_cb

    def _position_idx(self, side: str) -> int:
        """
        Bybit v5 positionIdx mapping:
        - One-way: 0 (side не требуется)
        - Hedge: 1 для лонга (Buy), 2 для шорта (Sell)
        """
        if getattr(CFG, "hedge_mode", False):
            return 1 if side == "Buy" else 2
        return 0

    async def _safe_call(self, cb, **kwargs):
        if cb is None:
            return
        try:
            res = cb(**kwargs)
            if asyncio.iscoroutine(res):
                await res
        except Exception as e:
            log.error(f"[TG] notify callback failed: {e}")

    async def _enter_with_sl(self, side: str, qty: float, price_for_sl: float):
        order_res = await bybit.place_market_order(
            side=side,
            qty=qty,
            position_idx=self._position_idx(side)
        )
        self._last_exec_signal = "long" if side == "Buy" else "short"
        self._last_exec_ts = time.time()
        # отмечаем локально, что позиция ожидаемо открыта
        if getattr(CFG, "hedge_mode", False):
            self._open_sides.add(side)
        else:
            self._open_sides.add("ONEWAY")

        # посчитать и поставить SL
        sl = self._calc_stop_loss(side, price_for_sl)
        if sl is None:
            log.warning(
                f"[SL] skip: cannot compute stopLoss side={side} price={price_for_sl} "
                f"support={self.support} resistance={self.resistance}"
            )
            # Notify about position open without SL
            await self._safe_call(
                self._notify_open,
                symbol=CFG.symbol,
                side=side,
                qty=str(qty),
                entry_price=float(price_for_sl),
                stop_loss=None,
                take_profit=None,
                reason=f"{self.mode} сигнал стратегии",
                expected_exit="по смене режима или ломке уровня",
                order_id=(order_res.get("result", {}) or {}).get("orderId"),
                position_idx=self._position_idx(side),
            )
            return

        try:
            await bybit.set_trading_stop_insurance_sl(
                position_idx=self._position_idx(side),
                stop_loss=sl,
                side=(side if getattr(CFG, "hedge_mode", False) else None)
            )
            log.info(f"[SL] set -> side={side} stopLoss={sl}")
        except Exception as e:
            log.error(f"[SL] failed side={side} stopLoss={sl} err={e}")

        # Telegram notification about the opened position
        await self._safe_call(
            self._notify_open,
            symbol=CFG.symbol,
            side=side,
            qty=str(qty),
            entry_price=float(price_for_sl),
            stop_loss=float(sl) if sl is not None else None,
            take_profit=None,
            reason=f"{self.mode} сигнал стратегии",
            expected_exit="по смене режима или ломке уровня",
            order_id=(order_res.get("result", {}) or {}).get("orderId"),
            position_idx=self._position_idx(side),
        )

    async def wallet_text(self) -> str:
        """Возвращает только число баланса, округлённое до 2 знаков (как строка)."""
        try:
            eq = await bybit.get_total_equity()
            # Чистое число
            if isinstance(eq, (int, float)):
                return f"{eq:.2f}"
            # Ответ вида { result: { list: [ { totalEquity: "..." } ] } }
            if isinstance(eq, dict):
                res = eq.get("result") or {}
                lst = res.get("list") or []
                if lst and isinstance(lst[0], dict) and lst[0].get("totalEquity") is not None:
                    return f"{float(lst[0]['totalEquity']):.2f}"
            return "n/a"
        except Exception as e:
            log.error(f"[BALANCE] failed: {e}")
            return "n/a"


    async def _current_open_sides(self) -> set:
        """
        Возвращает множество открытых сторон по данным биржи:
        - hedge_mode=False: {'ONEWAY'} если size != 0, иначе пустое множество
        - hedge_mode=True: подмножество из {'Buy','Sell'} по positionIdx (1=Buy, 2=Sell)
        """
        sides = set()
        try:
            pos = await bybit.get_positions()
            items = []
            if isinstance(pos, dict):
                res = pos.get("result") or {}
                items = res.get("list") or res.get("positions") or []
            elif isinstance(pos, list):
                items = pos
            for it in items:
                try:
                    size = float(str(it.get("size") or it.get("qty") or 0))
                    if size == 0:
                        continue
                    if not getattr(CFG, "hedge_mode", False):
                        sides.add("ONEWAY")
                        break
                    pidx = int(str(it.get("positionIdx") or it.get("position_index") or 0))
                    if pidx == 1:
                        sides.add("Buy")
                    elif pidx == 2:
                        sides.add("Sell")
                    else:
                        # если Bybit не отдал pidx, пробуем по полю side
                        s = (it.get("side") or "").strip()
                        if s in ("Buy", "Sell"):
                            sides.add(s)
                except Exception:
                    continue
        except Exception as e:
            log.error(f"[POS] fetch/sides failed: {e}")
        return sides

    async def _has_open_position(self, side: Optional[str] = None) -> bool:
        """
        Проверяет на бирже наличие открытой позиции.
        - В one-way режиме: любая позиция с size != 0 считается открытой.
        - В hedge режиме: матчит конкретную сторону через positionIdx (1=Buy, 2=Sell), если side указан.
        """
        try:
            pos = await bybit.get_positions()  # ожидается ответ v5 /positions
        except Exception as e:
            log.error(f"[POS] fetch failed: {e}")
            return False

        try:
            # Универсальный парсинг: Bybit v5 обычно кладёт позиции в result.list
            items = []
            if isinstance(pos, dict):
                res = pos.get("result") or {}
                items = res.get("list") or res.get("positions") or []
            elif isinstance(pos, list):
                items = pos
            else:
                items = []

            if not items:
                return False

            hedge = bool(getattr(CFG, "hedge_mode", False))
            # маппинг позиции по side в hedge
            target_idx = None
            if hedge and side in ("Buy", "Sell"):
                target_idx = 1 if side == "Buy" else 2

            for it in items:
                try:
                    # допускаем строковые числа
                    size = float(str(it.get("size") or it.get("qty") or 0))
                    if size == 0:
                        continue

                    if not hedge:
                        # one-way: любая непустая позиция
                        return True

                    # hedge: сверяем positionIdx
                    pidx = int(str(it.get("positionIdx") or it.get("position_index") or 0))
                    if target_idx is None or pidx == target_idx:
                        return True
                except Exception:
                    continue

            return False
        except Exception as e:
            log.error(f"[POS] parse failed: {e}")
            return False

    async def tick(self):
        kline = await bybit.get_kline_5m(CFG.sr_5m_limit)
        if not kline or "result" not in kline:
            return

        # --- синхронизация состояния позиций и нотификация о внешнем закрытии ---
        try:
            current_sides = await self._current_open_sides()
            # если ранее считали, что позиция была, а теперь её нет — это внешнее закрытие (ручное/по стопу)
            closed_sides = self._open_sides - current_sides
            for s in closed_sides:
                human_side = None if s == "ONEWAY" else s
                log.info(f"[POS] externally closed -> side={human_side or 'one-way'}")
                await self._safe_call(
                    self._notify_close,
                    symbol=CFG.symbol,
                    side=human_side,
                    reason="позиция закрыта на бирже (вручную/по стопу)"
                )
            # обновляем трекинг (для one-way это либо пусто, либо {'ONEWAY'})
            self._open_sides = set(current_sides)
        except Exception as e:
            log.error(f"[POS] sync failed: {e}")

        klines = kline["result"]["list"]
        self.support, self.resistance = find_sr(klines)
        price = float(klines[0][4])

        self.mode = detect_mode(price, self.support, self.resistance)
        log.info(f"[REGIME] mode={self.mode} price={price} support={self.support} resistance={self.resistance}")

        sig = trade_signal(self.mode, price, self.support, self.resistance)

        # дополнительная защита: сверяемся с биржей — если позиция уже есть, не переоткрываем
        if sig in ("long", "short"):
            side = "Buy" if sig == "long" else "Sell"
            if await self._has_open_position(side if getattr(CFG, "hedge_mode", False) else None):
                log.info(f"[TRADE] skip: already have open position on exchange (side={side})")
                # синхронизируем локальный последний сигнал, чтобы не спамить
                self._last_exec_signal = sig
                self._last_exec_ts = time.time()
                # удостоверимся, что локальный трекинг отражает факт открытой позиции
                if getattr(CFG, "hedge_mode", False):
                    self._open_sides.add(side)
                else:
                    self._open_sides.add("ONEWAY")
                return

        # анти-спам: только на смене сигнала и после cooldown
        if not self._can_execute(sig):
            log.debug(f"[TRADE] skip: sig={sig} last={self._last_exec_signal} "
                      f"cooldown={self._entry_cooldown_sec:.0f}s")
            return

        log.info(f"[TRADE] signal -> {sig}")

        if sig == "long":
            await self._enter_with_sl("Buy", CFG.min_qty, price)
        elif sig == "short":
            await self._enter_with_sl("Sell", CFG.min_qty, price)


trader = Trader()
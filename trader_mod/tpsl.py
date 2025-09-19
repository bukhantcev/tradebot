# trader_mod/tpsl.py
import asyncio
import logging
import time

log = logging.getLogger("TRADER")


def _fix_tpsl(self, side: str, price: float, sl: float, tp: float, tick: float) -> tuple[float, float]:
    p = float(price)
    sl_f, tp_f = float(sl), float(tp)
    t = max(float(tick), 0.0) or 0.1

    if side == "Buy":
        if sl_f >= p: sl_f = p - t
        if tp_f <= p: tp_f = p + t
        sl_f = self._round_step(sl_f, t)
        tp_f = self._ceil_step(tp_f, t)
        if sl_f >= p: sl_f = p - 2 * t
        if tp_f <= p: tp_f = p + 2 * t
    else:
        if sl_f <= p: sl_f = p + t
        if tp_f >= p: tp_f = p - t
        sl_f = self._ceil_step(sl_f, t)
        tp_f = self._round_step(tp_f, t)
        if sl_f <= p: sl_f = p + 2 * t
        if tp_f >= p: tp_f = p - 2 * t
    return sl_f, tp_f


def _normalize_tpsl_with_anchor(self, side: str, base_price: float, sl: float, tp: float, tick: float) -> tuple[float, float]:
    try:
        last = float(self._last_price()) or 0.0
    except Exception:
        last = 0.0

    anchor = float(base_price or 0.0)
    if side == "Buy":
        if last > 0: anchor = max(anchor, last)
        if tp <= anchor: tp = self._ceil_step(anchor + tick, tick)
        if sl >= anchor: sl = self._round_step(anchor - tick, tick)
    else:
        if last > 0: anchor = min(anchor, last) if anchor > 0 else last
        if tp >= anchor: tp = self._round_step(anchor - tick, tick)
        if sl <= anchor: sl = self._ceil_step(anchor + tick, tick)

    sl_f, tp_f = _fix_tpsl(self, side, anchor if anchor > 0 else (base_price or last or tp), sl, tp, tick)
    return sl_f, tp_f


def _cancel_realigner(self):
    t = getattr(self, "_realign_task", None)
    if t and not t.done():
        try:
            t.cancel()
        except Exception:
            pass
    self._realign_task = None


async def _realign_tpsl(self, side: str, desired_sl: float, desired_tp: float, tick: float, debounce: float = 0.8, max_tries: int = 30):
    try:
        side = "Buy" if side == "Buy" else "Sell"
        tries = 0
        while tries < max_tries:
            tries += 1

            ps, sz = self._position_side_and_size()
            if not ps or sz <= 0:
                log.debug("[REALIGN] flat — stop")
                break

            sl_norm, tp_norm = _normalize_tpsl_with_anchor(self, side, base_price=0.0, sl=desired_sl, tp=desired_tp, tick=tick)

            try:
                r = self.client.trading_stop(
                    self.symbol,
                    side=side,
                    stop_loss=sl_norm,
                    take_profit=tp_norm,
                    tpslMode="Full",
                    tpTriggerBy="LastPrice",
                    slTriggerBy="MarkPrice",
                    tpOrderType="Market",
                    positionIdx=0,
                )
                rc = r.get("retCode")
                if rc in (0, None, 34040):
                    tag = "OK" if rc in (0, None) else "UNCHANGED"
                    log.info(f"[REALIGN][{tag}] sl={self._fmt(sl_norm)} tp={self._fmt(tp_norm)} (try {tries})")
                    if abs(tp_norm - desired_tp) <= 2 * max(tick, 1e-9) and abs(sl_norm - desired_sl) <= 2 * max(tick, 1e-9):
                        break
                else:
                    log.debug(f"[REALIGN][RC] rc={rc} msg={r.get('retMsg')}")
            except Exception as e:
                log.debug(f"[REALIGN][EXC] {e}")

            await asyncio.sleep(max(0.1, float(debounce)))
    except asyncio.CancelledError:
        pass
    except Exception as e:
        log.debug(f"[REALIGN][STOP] {e}")
    finally:
        self._realign_task = None


async def _wait_position_open(self, timeout: float = 10.0, interval: float = 0.3) -> bool:
    """
    Короткое ожидание появления позиции (size>0), чтобы fallback trading-stop не падал rc=10001.
    Возвращает True, если позиция открылась в течение timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            pl = self.client.position_list(self.symbol)
            lst = pl.get("result", {}).get("list", [])
            for it in lst:
                size_str = it.get("size") or it.get("positionValue") or ""
                try:
                    size = float(size_str) if size_str != "" else 0.0
                except Exception:
                    size = 0.0
                if size > 0:
                    return True
        except Exception:
            pass
        await asyncio.sleep(interval)
    return False


async def _wait_position_flat(self, timeout: float = 3600.0, interval: float = 0.5) -> bool:
    """
    Ждём, пока позиция станет плоской (size == 0).
    Возвращает True, если успели за timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            pl = self.client.position_list(self.symbol)
            lst = pl.get("result", {}).get("list", [])
            any_size = 0.0
            for it in lst:
                sz = float(it.get("size") or 0.0)
                any_size += sz
            if any_size <= 0:
                return True
        except Exception:
            pass
        await asyncio.sleep(interval)
    return False


async def _await_fill_or_retry(self, order_id: str, side: str, qty: float) -> bool:
    """
    Расширенный ретрай исполнения входа.
    Порядок:
      1) Быстрый чек: ждём факт появления позиции (size>0).
      2) Если исходный ордер filled/partiallyFilled — короткое подтверждение.
      3) Если cancelled/rejected/new/пусто — серия повторов:
         - Market без preferSmart.
         - Market-safe с slip_percent=0.20.
         - Market-safe с slip_percent=0.30.
         - Уменьшаем qty до ~75% (не ниже minQty) и ещё раз с 0.30.
      После каждого шага ждём фактического открытия позиции.
    """
    # 0) быстрый чек
    if await self._wait_position_open(timeout=10.0, interval=0.3):
        return True

    # 1) статус исходного ордера
    try:
        st = self.client.get_order_status(self.symbol, order_id)
    except Exception:
        st = {"status": None, "cumExecQty": 0.0, "qty": 0.0}

    status = (st.get("status") or "").lower()
    filled = float(st.get("cumExecQty") or 0.0) > 0.0

    if filled or status in ("filled", "partiallyfilled"):
        # короткое подтверждение факта появления позиции
        return await self._wait_position_open(timeout=5.0, interval=0.3)

    # 2) если отменён/отклонён/непонятен — запускаем расширенные повторы
    if status in ("cancelled", "rejected", "new", "") or not status:
        log.warning(f"[ORDER][RETRY] status={status or 'n/a'} -> start fallback sequence")

        # вспомогательная функция ожидания с логом
        async def _await_open(tag: str, t: float = 10.0) -> bool:
            ok = await self._wait_position_open(timeout=t, interval=0.3)
            if ok:
                log.info(f"[ORDER][RETRY][{tag}] position detected")
            return ok

        # A) Market без preferSmart (иногда Smart даёт авто-cancel)
        try:
            r = self.client.place_order(
                self.symbol,
                side,
                qty,
                order_type="Market",
                position_idx=0,
            )
            if r.get("retCode") == 0 and await _await_open("PLAIN"):
                return True
        except Exception as e:
            log.warning(f"[ORDER][RETRY][PLAIN][EXC] {e}")

        # B) Market-safe со slip_percent=0.20
        try:
            r = self.client.place_market_safe(self.symbol, side, qty, position_idx=0, slip_percent=0.20)
            if r.get("retCode") == 0 and await _await_open("SAFE20"):
                return True
        except Exception as e:
            log.warning(f"[ORDER][RETRY][SAFE20][EXC] {e}")

        # C) Market-safe со slip_percent=0.30
        try:
            r = self.client.place_market_safe(self.symbol, side, qty, position_idx=0, slip_percent=0.30)
            if r.get("retCode") == 0 and await _await_open("SAFE30"):
                return True
        except Exception as e:
            log.warning(f"[ORDER][RETRY][SAFE30][EXC] {e}")

        # D) Понижаем qty до ~75% (не ниже minQty) и пробуем снова с 0.30
        try:
            f = self.ensure_filters()
            min_qty = float(f.get("minQty", 0.001))
            step = float(f.get("qtyStep", 0.001))
        except Exception:
            min_qty, step = 0.001, 0.001

        adj_qty = qty * 0.75
        # округление вниз к шагу
        n = int(max(adj_qty, 0.0) / max(step, 1e-9))
        adj_qty = max(min_qty, n * step)
        if adj_qty >= qty:
            # форсируем уменьшение хотя бы на 1 шаг
            adj_qty = max(min_qty, (int(qty / max(step, 1e-9)) - 1) * step)

        if adj_qty >= min_qty:
            log.info(f"[ORDER][RETRY][QTY↓] {self._fmt(qty)} -> {self._fmt(adj_qty)}")
            try:
                r = self.client.place_market_safe(self.symbol, side, adj_qty, position_idx=0, slip_percent=0.30)
                if r.get("retCode") == 0 and await _await_open("SAFE30_Q↓"):
                    return True
            except Exception as e:
                log.warning(f"[ORDER][RETRY][SAFE30_Q↓][EXC] {e}")
        else:
            log.warning(f"[ORDER][RETRY][QTY↓][SKIP] adj_qty<{min_qty}")

        log.error("[ORDER][FAIL] no fill after extended retries")
        return False

    # 3) иначе — ещё короткие попытки дождаться появления позиции
    for _ in range(20):
        if await self._wait_position_open(timeout=1.0, interval=0.3):
            return True
    return False
async def _apply_sl_failsafe(self, side: str, sl: float) -> bool:
    """
    Быстрая страховочная установка ТОЛЬКО SL сразу после открытия позиции.
    – Нормализует SL к последней цене с учётом тика.
    – Ставит через trading_stop(tpslMode=Full, только stop_loss), slTriggerBy=MarkPrice.
    – Несколько ретраев, подробные логи. Возвращает True при успешной установке или
      если биржа вернула UNCHANGED (34040), что для нас тоже ок.
    """
    try:
        f = self.ensure_filters()
        tick = float(f.get("tickSize") or 0.1)
        side = "Buy" if side == "Buy" else "Sell"

        # Подправим SL относительно текущего якоря (Last/Mark) так, чтобы он был валиден
        last = 0.0
        try:
            last = float(self._last_price()) or 0.0
        except Exception:
            last = 0.0

        sl_f = float(sl)
        if last > 0:
            if side == "Buy":
                # для лонга SL должен быть ниже цены
                if sl_f >= last:
                    sl_f = last - tick
                sl_f = self._round_step(sl_f, tick)
                if sl_f >= last:
                    sl_f = last - 2 * tick
            else:
                # для шорта SL должен быть выше цены
                if sl_f <= last:
                    sl_f = last + tick
                sl_f = self._ceil_step(sl_f, tick)
                if sl_f <= last:
                    sl_f = last + 2 * tick
        else:
            # нет last — просто приведём к шагу без принуждения
            if side == "Buy":
                sl_f = self._round_step(sl_f, tick)
            else:
                sl_f = self._ceil_step(sl_f, tick)

        for attempt in range(1, 6):
            try:
                r = self.client.trading_stop(
                    self.symbol,
                    side=side,
                    stop_loss=sl_f,
                    tpslMode="Full",
                    slTriggerBy="MarkPrice",
                    positionIdx=0,
                )
                rc = r.get("retCode")
                if rc in (0, None, 34040):
                    tag = "OK" if rc in (0, None) else "UNCHANGED"
                    log.info(f"[SL][FAILSAFE][{tag}] SL={self._fmt(sl_f)} (try {attempt})")
                    return True
                else:
                    log.warning(f"[SL][FAILSAFE][RC] rc={rc} msg={r.get('retMsg')} (try {attempt})")
            except Exception as e:
                log.warning(f"[SL][FAILSAFE][EXC] {e} (try {attempt})")
            await asyncio.sleep(0.25 * attempt)
        log.error("[SL][FAILSAFE][FAIL] cannot set stop-loss after retries")
        return False
    except Exception as e:
        log.warning(f"[SL][FAILSAFE][CRASH] {e}")
        return False


async def _apply_tpsl_failsafe(self, side: str, base_price: float, sl: float, tp: float) -> bool:
    """
    Страховочная постановка TP+SL пакетом после появления позиции.
    – Нормализует цели через _normalize_tpsl_with_anchor(...)
    – Ставит через trading_stop(..., tpTriggerBy=LastPrice, slTriggerBy=MarkPrice, tpOrderType=Market)
    – Несколько ретраев. Возвращает True, если удалось поставить или биржа ответила UNCHANGED.
    """
    try:
        f = self.ensure_filters()
        tick = float(f.get("tickSize") or 0.1)
        side = "Buy" if side == "Buy" else "Sell"

        sl_n, tp_n = _normalize_tpsl_with_anchor(self, side, base_price=float(base_price or 0.0), sl=float(sl), tp=float(tp), tick=tick)

        for attempt in range(1, 6):
            try:
                r = self.client.trading_stop(
                    self.symbol,
                    side=side,
                    stop_loss=sl_n,
                    take_profit=tp_n,
                    tpslMode="Full",
                    tpTriggerBy="LastPrice",
                    slTriggerBy="MarkPrice",
                    tpOrderType="Market",
                    positionIdx=0,
                )
                rc = r.get("retCode")
                if rc in (0, None, 34040):
                    tag = "OK" if rc in (0, None) else "UNCHANGED"
                    log.info(f"[TPSL][FAILSAFE][{tag}] SL={self._fmt(sl_n)} TP={self._fmt(tp_n)} (try {attempt})")
                    return True
                else:
                    log.warning(f"[TPSL][FAILSAFE][RC] rc={rc} msg={r.get('retMsg')} (try {attempt})")
            except Exception as e:
                log.warning(f"[TPSL][FAILSAFE][EXC] {e} (try {attempt})")
            await asyncio.sleep(0.25 * attempt)
        log.error("[TPSL][FAILSAFE][FAIL] cannot set TP/SL after retries")
        return False
    except Exception as e:
        log.warning(f"[TPSL][FAILSAFE][CRASH] {e}")
        return False


async def _watchdog_close_on_lastprice(self, side: str, sl_price: float, tp_price: float, check_interval: float = 0.3, max_wait: float = 3600.0) -> bool:
    deadline = __import__("time").monotonic() + max_wait
    side = "Buy" if side == "Buy" else "Sell"
    opp = self._opposite(side)

    while __import__("time").monotonic() < deadline:
        try:
            ps, sz = self._position_side_and_size()
            if not ps or sz <= 0:
                return True

            last = self._last_price()
            if last <= 0:
                await asyncio.sleep(check_interval)
                continue

            trigger = False
            if side == "Buy":
                if (sl_price and last <= sl_price) or (tp_price and last >= tp_price):
                    trigger = True
            else:
                if (sl_price and last >= sl_price) or (tp_price and last <= tp_price):
                    trigger = True

            if trigger:
                log.info(f"[WATCH][CROSS] Last={self._fmt(last)} vs SL={self._fmt(sl_price)} / TP={self._fmt(tp_price)} -> force close")
                self._cancel_realigner()
                try:
                    await self.close_market(opp, sz)
                except Exception as e:
                    log.warning(f"[WATCH][CLOSE][EXC] {e}")
                ok_flat = await self._wait_position_flat(timeout=30.0, interval=0.25)
                return ok_flat
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning(f"[WATCH][EXC] {e}")
        await asyncio.sleep(check_interval)
    return False
# trader_mod/utils.py
import asyncio
import logging
from typing import Any, Dict

log = logging.getLogger("TRADER")


def _start_minute_logger(self):
    if self._minute_task and not self._minute_task.done():
        return

    async def _loop():
        import time
        last_min = int(time.time() // 60)
        while True:
            try:
                await asyncio.sleep(1.0)
                m = int(time.time() // 60)
                if m != last_min:
                    last_min = m
                    try:
                        px = self._last_price()
                    except Exception:
                        px = 0.0
                    try:
                        side, sz = self._position_side_and_size()
                    except Exception:
                        side, sz = (None, 0.0)
                    sl_price = self._minute_sl
                    mode = self._minute_mode
                    log.info(f"[STAT][MIN] mode={mode} px={self._fmt(px)} pos={side or 'Flat'} size={self._fmt(sz)} SL={(self._fmt(sl_price) if sl_price else 'None')}")
            except asyncio.CancelledError:
                log.info("[MIN][LOOP] cancelled")
                break
            except Exception:
                await asyncio.sleep(2.0)

    self._minute_task = asyncio.create_task(_loop(), name="minute_stat")
    log.info("[MIN][START] minute logger started")


def set_minute_status(self, mode: str, sl_price: float | None):
    self._minute_mode = mode
    self._minute_sl = sl_price
    log.info(f"[MIN][MODE] set mode={mode} SL={(self._fmt(sl_price) if sl_price else 'None')}")
    _start_minute_logger(self)


async def stop_minute_logger(self):
    t = self._minute_task
    if t and not t.done():
        try:
            t.cancel()
            try:
                await asyncio.wait_for(t, timeout=1.0)
                log.info("[MIN][STOP] minute logger stopped")
            except asyncio.TimeoutError:
                log.warning("[MIN][STOP][TIMEOUT] cancel wait timed out; detaching task")
            except Exception as e:
                log.warning(f"[MIN][STOP][EXC] {e}")
        finally:
            self._minute_task = None
    else:
        self._minute_task = None


async def open_market(self, side: str, signal: Dict[str, Any]):
    def _sg(key, default=None):
        if isinstance(signal, dict):
            return signal.get(key, default)
        return getattr(signal, key, default)

    def _sg_multi(keys, default=None):
        for k in keys:
            v = _sg(k, None)
            if v is not None:
                return v
        return default

    if self.equity <= 0 or self.available <= 0:
        self.refresh_equity()
    self.ensure_filters()
    self.ensure_leverage()

    price = float(_sg_multi(["price", "close"]) or 0.0)
    if price <= 0:
        price = float(_sg("tp") or 0.0) or 1.0

    sl = float(_sg_multi(["sl", "stop_loss", "stopLoss"]) or 0.0)
    tp = float(_sg_multi(["tp", "take_profit", "takeProfit"]) or 0.0)
    if sl <= 0 or tp <= 0:
        log.info("[ENTER][SKIP] bad SL/TP")
        self._cancel_realigner()
        await stop_minute_logger(self)
        return

    set_minute_status(self, "normal", float(sl) if sl else None)

    qty = self._calc_qty(side, price, sl)
    log.info(f"[QTY] risk%={self.risk_pct*100:.2f} stop={abs(price-sl):.2f} equity={self.equity:.2f} avail={self.available:.2f} -> qty={self._fmt(qty)}")
    if qty <= 0:
        log.info("[ENTER][SKIP] qty=0")
        self._cancel_realigner()
        await stop_minute_logger(self)
        return

    f = self.ensure_filters()
    tick = f["tickSize"]
    sl_r = self._ceil_step(sl, tick) if side == "Buy" else self._round_step(sl, tick)
    tp_r = self._round_step(tp, tick) if side == "Buy" else self._ceil_step(tp, tick)
    sl_r, tp_r = self._fix_tpsl(side, price, sl_r, tp_r, tick)

    prev_high = _sg_multi(["prev_high", "prevHigh", "prevH", "previous_high", "prev_high_price"])
    prev_low = _sg_multi(["prev_low", "prevLow", "prevL", "previous_low", "prev_low_price"])
    use_ext = bool(self.entry_extremes and prev_high and prev_low)
    if use_ext:
        log.info(f"[EXT][MODE] ON  prevH={self._fmt(prev_high)} prevL={self._fmt(prev_low)} qty={self._fmt(qty)}")
    else:
        reason = []
        if not self.entry_extremes:
            reason.append("flag_off")
        if not prev_high or not prev_low:
            reason.append("no_prev_hl")
        try:
            fields = list(signal.keys()) if isinstance(signal, dict) else list(vars(signal).keys())
        except Exception:
            fields = ["?"]
        log.info(f"[EXT][MODE] OFF ({','.join(reason) if reason else 'n/a'}) fields={fields}")

    log.info("[EXT][CHECKPOINT] minute logger set status")
    try:
        set_minute_status(self, "ext" if use_ext else "normal", float(sl) if sl else None)
    except Exception as e:
        log.exception(f"[EXT][LOGGER][EXC] {e}")

    if not use_ext:
        try:
            ps, sz = self._position_side_and_size()
            if ps and sz > 0:
                log.info(f"[ENTER][FLAT] close existing {ps} size={self._fmt(sz)} before new entry")
                await self.close_market(self._opposite(ps), sz)
                for _ in range(20):
                    p2, s2 = self._position_side_and_size()
                    if not p2 or s2 <= 0:
                        break
                    await asyncio.sleep(0.25)
            log.info("[EXT][CHECKPOINT] after flat enforcement")
        except Exception as e:
            log.exception(f"[EXT][FLAT][EXC] {e}")
            log.info("[EXT][CHECKPOINT] after flat enforcement (with exception)")
    else:
        log.info("[ENTER][FLAT] skip in ext mode (only pending limits will be replaced)")

    try:
        log.info(f"[EXT][CHECKPOINT] before branch use_ext={use_ext}")
    except Exception:
        log.exception("[EXT][CHECKPOINT][EXC] before branch")

    if use_ext:
        log.info("[EXT][CHECKPOINT] entering use_ext branch")
        log.info(f"[EXT][LIM][FLOW] side={side} prevH={self._fmt(prev_high)} prevL={self._fmt(prev_low)} qty={self._fmt(qty)}")
        try:
            await self._enter_extremes_with_limits(side, float(prev_high), float(prev_low), qty, sl=float(sl), tp=float(tp))
            log.info("[EXT][RETURN] _enter_extremes_with_limits finished")
        except Exception as e:
            log.exception(f"[EXT][CRASH] _enter_extremes_with_limits exception: {e}")
        finally:
            await stop_minute_logger(self)
        return

    # --- Маркет-вход (обычный) ---
    log.info(f"[ENTER] {side} qty={self._fmt(qty)}")
    attempt_qty = qty
    order_id = None
    for _ in range(6):
        r = self.client.place_order(
            self.symbol,
            side,
            attempt_qty,
            order_type="Market",
            preferSmart=True,
        )
        rc = r.get("retCode")
        if rc == 0:
            order_id = r.get("result", {}).get("orderId", "")
            log.info(f"[ORDER←] OK id={order_id}")
            if self.notifier:
                try:
                    await self.notifier.notify(f"✅ {side} {self.symbol} qty={self._fmt(attempt_qty)} (id {order_id})")
                except Exception:
                    pass
            break
        if rc == 110007:
            f2 = self.ensure_filters()
            step = f2.get("qtyStep", 0.001)
            min_qty = f2.get("minQty", 0.001)
            new_qty = self._round_down_qty(attempt_qty * 0.75)
            if new_qty >= attempt_qty:
                new_qty = self._round_down_qty(attempt_qty - step)
            if new_qty < min_qty:
                log.error(f"[ENTER][FAIL] insufficient balance for minimal qty; last={self._fmt(attempt_qty)}")
                self._cancel_realigner()
                await stop_minute_logger(self)
                return
            log.info(f"[ENTER][RETRY_QTY] 110007 -> {self._fmt(attempt_qty)} -> {self._fmt(new_qty)}")
            attempt_qty = new_qty
            continue
        if rc == 30208:
            log.warning("[ENTER][RETRY] 30208: add slippage Percent=0.05")
            r2 = self.client.place_order(
                self.symbol, side, attempt_qty, order_type="Market",
                slippageToleranceType="Percent", slippageTolerance="0.05"
            )
            if r2.get("retCode") == 0:
                order_id = r2.get("result", {}).get("orderId", "")
                log.info(f"[ORDER←] OK id={order_id} (slip 0.05%)")
                if self.notifier:
                    try:
                        await self.notifier.notify(f"✅ {side} {self.symbol} qty={self._fmt(attempt_qty)} (id {order_id}, slip 0.05%)")
                    except Exception:
                        pass
                break
            log.warning("[ENTER][RETRY] 30208: add slippage TickSize=5")
            r3 = self.client.place_order(
                self.symbol, side, attempt_qty, order_type="Market",
                slippageToleranceType="TickSize", slippageTolerance="5"
            )
            if r3.get("retCode") == 0:
                order_id = r3.get("result", {}).get("orderId", "")
                log.info(f"[ORDER←] OK id={order_id} (slip 5 ticks)")
                if self.notifier:
                    try:
                        await self.notifier.notify(f"✅ {side} {self.symbol} qty={self._fmt(attempt_qty)} (id {order_id}, slip 5 ticks)")
                    except Exception:
                        pass
                break
            log.error(f"[ENTER][FAIL] {r3}")
            self._cancel_realigner()
            await stop_minute_logger(self)
            return
        log.error(f"[ENTER][FAIL] {r}")
        self._cancel_realigner()
        await stop_minute_logger(self)
        return

    if not order_id:
        log.error("[ENTER][FAIL] no orderId")
        self._cancel_realigner()
        await stop_minute_logger(self)
        return

    if not await self._await_fill_or_retry(order_id, side, attempt_qty):
        log.warning("[ENTER][ABORT] no fill")
        self._cancel_realigner()
        await stop_minute_logger(self)
        return

    # --- FAILSAFE SL: отдельная страховка стоп-лосса на бирже ---
    try:
        if sl:
            ok_fs_sl = await self._apply_sl_failsafe(side, float(sl))
            if ok_fs_sl:
                log.info(f"[FAILSAFE][SL] placed native SL for {side} at {self._fmt(float(sl))}")
            else:
                log.warning("[FAILSAFE][SL] native SL not confirmed")
    except Exception as e:
        log.warning(f"[FAILSAFE][SL][EXC] {e}")

    # SL сразу (без TP)
    if sl:
        try:
            ps0, sz0 = self._position_side_and_size()
            if not ps0 or sz0 <= 0:
                log.debug("[SL][NORM][SKIP] flat right after entry")
            else:
                tr = self.client.trading_stop(
                    self.symbol, side=side, stop_loss=float(sl),
                    tpslMode="Full", slTriggerBy="LastPrice", positionIdx=0,
                )
                rc = tr.get("retCode")
                if rc in (0, None, 34040):
                    tag = "OK" if rc in (0, None) else "UNCHANGED"
                    log.info(f"[SL][NORM] set SL={self._fmt(sl)} {tag}")
                else:
                    log.warning(f"[SL][NORM][FAIL] {tr}")
        except Exception as e:
            log.warning(f"[SL][NORM][EXC] {e}")

    qty = attempt_qty

    actual_side = side
    base_price = None
    try:
        pos = self.client.position_list(self.symbol)
        items = pos.get("result", {}).get("list", [])
        if items:
            it = items[0]
            actual_side = it.get("side") or actual_side
            base_price = float(it.get("avgPrice") or it.get("entryPrice") or 0.0)
    except Exception:
        pass

    f3 = self.ensure_filters()
    tick = f3["tickSize"]
    if not base_price or base_price <= 0:
        base_price = float(self._last_price()) or price

    sl_adj, tp_adj = self._normalize_tpsl_with_anchor(actual_side, base_price, sl_r, tp_r, tick)

    log.info(f"[TPSL][NORM] side={actual_side} base={self._fmt(base_price)} sl={self._fmt(sl_adj)} tp={self._fmt(tp_adj)} (anchor=Last/base)")

    ps1, sz1 = self._position_side_and_size()
    if not ps1 or sz1 <= 0:
        log.debug("[TPSL][SKIP] flat before setting final TP/SL")
    else:
        r2 = self.client.trading_stop(
            self.symbol, side=actual_side,
            stop_loss=sl_adj, take_profit=tp_adj,
            tpslMode="Full", tpTriggerBy="LastPrice",
            slTriggerBy="LastPrice", tpOrderType="Market",
            positionIdx=0,
        )
        rc2 = r2.get("retCode")
        if rc2 in (0, None, 34040):
            tag = "OK" if rc2 in (0, None) else "UNCHANGED"
            log.info(f"[TPSL] sl={self._fmt(sl_adj)} tp={self._fmt(tp_adj)} {tag}")
            if self.notifier:
                try:
                    await self.notifier.notify(f"🎯 TP/SL set: SL {self._fmt(sl_adj)} / TP {self._fmt(tp_adj)}")
                except Exception:
                    pass
        else:
            log.warning(f"[TPSL][FAIL] {r2}")

    # --- FAILSAFE TP/SL: дублируем биржевым брекетом как страховку ---
    try:
        ok_fs_bracket = await self._apply_tpsl_failsafe(actual_side, float(base_price), float(sl_adj), float(tp_adj))
        if ok_fs_bracket:
            log.info(f"[FAILSAFE][BRACKET] native TP/SL placed (SL {self._fmt(sl_adj)} / TP {self._fmt(tp_adj)})")
        else:
            log.warning("[FAILSAFE][BRACKET] native TP/SL not confirmed")
    except Exception as e:
        log.warning(f"[FAILSAFE][BRACKET][EXC] {e}")

    self._cancel_realigner()
    try:
        self._realign_task = asyncio.create_task(
            self._realign_tpsl(actual_side, sl_r, tp_r, tick),
            name="tpsl_realign"
        )
    except Exception:
        self._realign_task = None


async def close_market(self, side: str, qty: float):
    if qty <= 0:
        log.info("[EXIT][SKIP] qty=0")
        return
    log.info(f"[EXIT] {side} qty={self._fmt(qty)}")
    r = self.client.place_order(
        self.symbol, side, qty, order_type="Market", preferSmart=True,
    )
    if r.get("retCode") == 0:
        oid = r.get("result", {}).get("orderId", "")
        log.info(f"[ORDER←] CLOSE OK id={oid}")
        if self.notifier:
            try:
                await self.notifier.notify(f"❌ Close {side} {self.symbol} qty={self._fmt(qty)} (id {oid})")
            except Exception:
                pass
    elif r.get("retCode") == 30208:
        log.warning("[ORDER][RETRY] 30208: close with slippage Percent=0.05")
        r = self.client.place_order(
            self.symbol, side, qty, order_type="Market",
            slippageToleranceType="Percent", slippageTolerance="0.05",
        )
        if r.get("retCode") == 0:
            oid = r.get("result", {}).get("orderId", "")
            log.info(f"[ORDER←] CLOSE OK id={oid} (slip 0.05%)")
            if self.notifier:
                try:
                    await self.notifier.notify(f"❌ Close {side} {self.symbol} qty={self._fmt(qty)} (id {oid}, slip 0.05%)")
                except Exception:
                    pass
        else:
            log.error(f"[ORDER][FAIL] {r}")
    else:
        log.error(f"[ORDER][FAIL] {r}")
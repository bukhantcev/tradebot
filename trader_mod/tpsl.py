# trader_mod/tpsl.py
import asyncio
import logging

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


async def _await_fill_or_retry(self, order_id: str, side: str, qty: float) -> bool:
    ok = await self._wait_position_open(timeout=10.0, interval=0.3)
    if ok:
        return True

    try:
        st = self.client.get_order_status(self.symbol, order_id)
    except Exception:
        st = {"status": None, "cumExecQty": 0.0, "qty": 0.0}
    status = (st.get("status") or "").lower()
    filled = float(st.get("cumExecQty") or 0.0) > 0.0

    if filled or status in ("filled", "partiallyfilled"):
        ok2 = await self._wait_position_open(timeout=5.0, interval=0.3)
        return ok2

    if status in ("cancelled", "rejected") or not status:
        log.warning(f"[ORDER][CANCELLED] status={status or 'n/a'} -> retry smart market")

        r = self.client.place_market_safe(self.symbol, side, qty, position_idx=0, slip_percent=0.10)
        if r.get("retCode") == 0:
            if await self._wait_position_open(timeout=10.0, interval=0.3):
                return True

        r = self.client.place_market_safe(self.symbol, side, qty, position_idx=0, slip_percent=0.20)
        if r.get("retCode") == 0:
            if await self._wait_position_open(timeout=10.0, interval=0.3):
                return True

        log.error("[ORDER][FAIL] no fill after retries")
        return False

    for _ in range(20):
        if await self._wait_position_open(timeout=1.0, interval=0.3):
            return True
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
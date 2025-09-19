# trader_mod/extremes.py
import asyncio
import logging
import time

log = logging.getLogger("TRADER")


def _place_conditional(self, side: str, trigger_price: float, qty: float, trigger_direction: int):
    body = {
        "category": "linear",
        "symbol": self.symbol,
        "side": side,
        "orderType": "Market",
        "qty": self._fmt(qty),
        "timeInForce": "IOC",
        "positionIdx": 0,
        "triggerPrice": self._fmt(trigger_price),
        "triggerDirection": trigger_direction,
    }
    body["orderLinkId"] = f"ext-{int(time.time()*1000)}-{side[0].lower()}"
    return self.client._request("POST", "/v5/order/create", body=body)


def _cancel_order(self, order_id=None, order_link_id=None):
    body = {"category": "linear", "symbol": self.symbol}
    if order_id: body["orderId"] = order_id
    if order_link_id: body["orderLinkId"] = order_link_id
    return self.client._request("POST", "/v5/order/cancel", body=body)


def _cancel_all_orders(self):
    body = {"category": "linear", "symbol": self.symbol}
    return self.client._request("POST", "/v5/order/cancel-all", body=body)


def _order_status_brief(self, order_id: str) -> str:
    try:
        st = self.client.get_order_status(self.symbol, order_id)
        s = (st.get("status") or "").lower()
        ce = float(st.get("cumExecQty") or 0.0)
        q = float(st.get("qty") or 0.0)
        return f"{s or 'n/a'} {ce}/{q}"
    except Exception:
        return "n/a"


async def enter_by_extremes(self, side: str, prev_high: float, prev_low: float, qty: float, sl_r: float, tp_r: float):
    if qty <= 0 or prev_high <= 0 or prev_low <= 0:
        log.info("[COND][SKIP] bad params")
        return

    r_sell = _place_conditional(self, "Sell", prev_high, qty, trigger_direction=1)
    r_buy = _place_conditional(self, "Buy", prev_low, qty, trigger_direction=2)

    oid_sell = r_sell.get("result", {}).get("orderId") if r_sell.get("retCode") == 0 else None
    oid_buy = r_buy.get("result", {}).get("orderId") if r_buy.get("retCode") == 0 else None

    if r_sell.get("retCode") != 0: log.warning(f"[COND][ERR] sell {r_sell}")
    if r_buy.get("retCode") != 0: log.warning(f"[COND][ERR] buy {r_buy}")

    if self.notifier:
        try:
            await self.notifier.notify(f"⏳ Cond placed: Sell@{self._fmt(prev_high)} / Buy@{self._fmt(prev_low)} qty={self._fmt(qty)}")
        except Exception:
            pass

    ok = await self._wait_position_open(timeout=300.0, interval=0.5)
    if not ok:
        log.warning("[COND][TIMEOUT] no fill within 5m — cancel both")
        if oid_sell: _cancel_order(self, order_id=oid_sell)
        if oid_buy: _cancel_order(self, order_id=oid_buy)
        return

    if oid_sell: _cancel_order(self, order_id=oid_sell)
    if oid_buy: _cancel_order(self, order_id=oid_buy)

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

    f = self.ensure_filters()
    tick = f["tickSize"]
    if base_price and base_price > 0:
        sl_adj, tp_adj = self._fix_tpsl(actual_side, base_price, sl_r, tp_r, tick)
    else:
        sl_adj, tp_adj = sl_r, tp_r

    log.info(f"[TPSL][NORM] side={actual_side} base={self._fmt(base_price) if base_price else 'n/a'} sl={self._fmt(sl_adj)} tp={self._fmt(tp_adj)}")

    r2 = self.client.trading_stop(
        self.symbol,
        side=actual_side,
        stop_loss=sl_adj,
        take_profit=tp_adj,
        tpslMode="Full",
        positionIdx=0,
    )
    if r2.get("retCode") in (0, None):
        log.info(f"[TPSL] sl={self._fmt(sl_adj)} tp={self._fmt(tp_adj)} OK")
        if self.notifier:
            try:
                await self.notifier.notify(f"🎯 TP/SL set: SL {self._fmt(sl_adj)} / TP {self._fmt(tp_adj)}")
            except Exception:
                pass
    else:
        log.warning(f"[TPSL][FAIL] {r2}")


async def enter_extremes_with_limits(self, side: str, prev_high: float, prev_low: float, qty: float, sl: float, tp: float):
    if qty <= 0:
        log.info("[EXT][LIM][SKIP] qty<=0")
        self._cancel_realigner()
        return

    f = self.ensure_filters()
    tick = float(f.get("tickSize", 0.1) or 0.1)
    eps_ticks = max(1, int(self.ext_eps_ticks))
    eps = max(tick * eps_ticks, tick)

    async def _prev_hl_rest() -> tuple[float | None, float | None]:
        try:
            r = self.client._request(
                "GET",
                "/v5/market/kline",
                params={"category": "linear", "symbol": self.symbol, "interval": "1", "limit": 3},
            )
            kl = (r.get("result", {}) or {}).get("list", [])
            if not kl:
                return (None, None)
            last = kl[-2] if len(kl) >= 2 else kl[-1]

            def _get(v, idx):
                try:
                    if isinstance(v, dict):
                        return float(v.get(idx) or v.get(idx.lower()) or 0.0)
                    return float(v[idx])
                except Exception:
                    return 0.0

            ph = _get(last, 2)
            pl = _get(last, 3)
            return (ph if ph > 0 else None, pl if pl > 0 else None)
        except Exception:
            return (None, None)

    cur_prev_high = float(prev_high or 0.0)
    cur_prev_low = float(prev_low or 0.0)

    active_oid: str | None = None
    max_cycles = 120

    for _ in range(max_cycles):
        if cur_prev_high <= 0 or cur_prev_low <= 0:
            ph, pl = await _prev_hl_rest()
            if ph and pl:
                cur_prev_high, cur_prev_low = ph, pl
            else:
                log.info("[EXT][LIM][SKIP] no prev HL from REST; retry in 5s")
                await asyncio.sleep(5)
                continue

        if side == "Buy":
            entry_price = self._ceil_step(cur_prev_low + eps, tick)
            tp_ref = float(tp) if (tp and tp > 0) else (max(cur_prev_high - eps, entry_price + tick))
            sl_ref = float(sl) if (sl and sl > 0) else (entry_price - 1.5 * tick)
            sl_adj, tp_adj = self._fix_tpsl("Buy", entry_price, sl_ref, tp_ref, tick)
        else:
            entry_price = self._round_step(cur_prev_high - eps, tick)
            tp_ref = float(tp) if (tp and tp > 0) else (min(cur_prev_low + eps, entry_price - tick))
            sl_ref = float(sl) if (sl and sl > 0) else (entry_price + 1.5 * tick)
            sl_adj, tp_adj = self._fix_tpsl("Sell", entry_price, sl_ref, tp_ref, tick)

        log.info(f"[EXT][LIM][PLACE] {side} limit entry={self._fmt(entry_price)} tp={self._fmt(tp_adj)} sl={self._fmt(sl_adj)} qty={self._fmt(qty)}")

        try:
            if self.available <= 0 or self.equity <= 0:
                self.refresh_equity()
        except Exception:
            pass

        attempt_qty = min(qty, self._calc_qty(side, entry_price, sl_adj))
        f_loc = self.ensure_filters()
        min_qty = float(f_loc.get("minQty", 0.001))
        if attempt_qty < min_qty:
            affordable = (self.available / (entry_price / max(self.leverage, 1.0))) >= min_qty
            if affordable:
                attempt_qty = min_qty
                log.info(f"[EXT][LIM][FORCE] adjusted qty up to min {self._fmt(min_qty)}")
            else:
                log.error(f"[EXT][LIM][SKIP] qty not affordable at entry={self._fmt(entry_price)} -> {self._fmt(attempt_qty)} < min {self._fmt(min_qty)}")
                self._cancel_realigner()
                await asyncio.sleep(60)
                cur_prev_high = cur_prev_low = 0.0
                continue

        active_oid = None
        for _try in range(6):
            r = self.client.place_order(
                self.symbol, side, attempt_qty, order_type="Limit",
                price=entry_price, timeInForce="GTC", position_idx=0,
            )
            rc = r.get("retCode")
            if rc == 0:
                active_oid = r.get("result", {}).get("orderId", "")
                log.info(f"[EXT][LIM][ORDER] id={active_oid} price={self._fmt(entry_price)} qty={self._fmt(attempt_qty)}")
                break
            if rc == 110007:
                new_qty = self._round_down_qty(attempt_qty * 0.75)
                if new_qty >= attempt_qty:
                    new_qty = self._round_down_qty(attempt_qty - float(f_loc.get('qtyStep', 0.001)))
                if new_qty < min_qty:
                    log.error(f"[EXT][LIM][FAIL] 110007: insufficient balance, last qty={self._fmt(attempt_qty)} < min {self._fmt(min_qty)}")
                    active_oid = None
                    break
                log.info(f"[EXT][LIM][RETRY_QTY] 110007 -> {self._fmt(attempt_qty)} -> {self._fmt(new_qty)}")
                attempt_qty = new_qty
                continue
            log.error(f"[EXT][LIM][FAIL] place_order {r}")
            active_oid = None
            break

        if not active_oid:
            self._cancel_realigner()
            await asyncio.sleep(60)
            cur_prev_high = cur_prev_low = 0.0
            continue

        end_ts = time.time() + 60
        filled = False
        while time.time() < end_ts:
            if await self._wait_position_open(timeout=1.0, interval=0.25):
                filled = True
                break
            await asyncio.sleep(0.5)

        if not filled:
            try:
                _cancel_order(self, order_id=active_oid)
            except Exception:
                pass
            log.warning("[EXT][LIM][REPLACE] no fill in 60s — re-evaluate last closed HL")
            ph, pl = await _prev_hl_rest()
            cur_prev_high = float(ph or 0.0)
            cur_prev_low = float(pl or 0.0)
            continue

        actual_side = side
        base_price = entry_price
        try:
            pos = self.client.position_list(self.symbol)
            items = pos.get("result", {}).get("list", [])
            if items:
                it = items[0]
                actual_side = it.get("side") or side
                b = float(it.get("avgPrice") or it.get("entryPrice") or 0.0)
                if b > 0:
                    base_price = b
        except Exception:
            pass

        sl_final, tp_final = self._normalize_tpsl_with_anchor(actual_side, base_price, sl_adj, tp_adj, tick)
        log.info(f"[EXT][LIM][TPSL] side={actual_side} base={self._fmt(base_price)} sl={self._fmt(sl_final)} tp={self._fmt(tp_final)} (anchor=Last/base)")
        # --- FAILSAFE: immediate SL as insurance in case trading_stop fails ---
        try:
            await self._apply_sl_failsafe(actual_side, float(sl_final))
        except Exception as e:
            log.warning(f"[FAILSAFE][SL][EXC] {e}")

        ps, sz = self._position_side_and_size()
        if not ps or sz <= 0:
            log.debug("[EXT][LIM][TPSL][SKIP] position flat right after fill")
        else:
            tr = self.client.trading_stop(
                self.symbol, side=actual_side,
                stop_loss=sl_final, take_profit=tp_final,
                tpslMode="Full", tpTriggerBy="LastPrice",
                slTriggerBy="LastPrice", tpOrderType="Market",
                positionIdx=0,
            )
            rc = tr.get("retCode")
            if rc in (0, None, 34040):
                tag = "OK" if rc in (0, None) else "UNCHANGED"
                log.info(f"[EXT][LIM][TPSL][{tag}] sl={self._fmt(sl_final)} tp={self._fmt(tp_final)}")
            else:
                log.warning(f"[EXT][LIM][TPSL][FAIL] {tr}")
            # --- FAILSAFE: bracket order (TP/SL) as exchange-side backup ---
            try:
                await self._apply_tpsl_failsafe(actual_side, float(sl_final), float(tp_final))
            except Exception as e:
                log.warning(f"[FAILSAFE][BRACKET][EXC] {e}")

        self._cancel_realigner()
        try:
            self._realign_task = asyncio.create_task(
                self._realign_tpsl(actual_side, sl_adj, tp_adj, tick),
                name="tpsl_realign"
            )
        except Exception:
            self._realign_task = None

        log.info("[EXT][LIM][WATCH] arming TP/SL watchdog (LastPrice cross)")
        while True:
            ok_flat = await self._watchdog_close_on_lastprice(actual_side, sl_final, tp_final, check_interval=0.25, max_wait=300.0)
            if ok_flat:
                break
            log.warning("[EXT][LIM][WATCH][TIMEOUT] position still open; keep monitoring…")

        self._cancel_realigner()

        try:
            ca = _cancel_all_orders(self)
            rc = ca.get("retCode")
            if rc in (0, None):
                log.info("[EXT][LIM][CANCEL_ALL][OK]")
            else:
                log.warning(f"[EXT][LIM][CANCEL_ALL][WARN] rc={rc} msg={ca.get('retMsg')}")
        except Exception as e:
            log.warning(f"[EXT][LIM][CANCEL_ALL][EXC] {e}")

        cur_prev_high = 0.0
        cur_prev_low = 0.0
        active_oid = None
        continue

    log.warning("[EXT][LIM][ABORT] max cycles reached; stop")
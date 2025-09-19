# trader_mod/extremes.py
import time
import asyncio
import logging
from .utils import round_step, ceil_step, fmt
log = logging.getLogger("TRADER")

class Extremes:
    def __init__(self, account, tpsl, client, symbol, risk, eps_ticks: int, get_last_price, get_pos_side_size, cancel_order, cancel_all, set_realign_task, stop_realign_task, notifier=None):
        self.account = account
        self.tpsl = tpsl
        self.client = client
        self.symbol = symbol
        self.risk = risk
        self.eps_ticks = max(1, int(eps_ticks))
        self.get_last_price = get_last_price
        self.get_pos_side_size = get_pos_side_size
        self.cancel_order = cancel_order
        self.cancel_all = cancel_all
        self.set_realign_task = set_realign_task
        self.stop_realign_task = stop_realign_task
        self.notifier = notifier

    def _coerce_qty(self, q) -> float | None:
        try:
            if q is None:
                return None
            if isinstance(q, (int, float)):
                return float(q)
            # tolerate strings like "0.003"
            if isinstance(q, str):
                q = q.strip()
                if not q:
                    return None
                return float(q)
            # sometimes a dict gets passed accidentally
            if isinstance(q, dict):
                for key in ("qty", "size", "amount", "value"):
                    if key in q:
                        try:
                            return float(q[key])
                        except Exception:
                            continue
                # try to extract from nested structure (common fields we saw)
                for key in ("close", "price"):
                    if key in q and isinstance(q[key], (int, float)):
                        # cannot deduce quantity from price-only dict
                        continue
                return None
        except Exception:
            return None

    async def _prev_hl_rest(self) -> tuple[float | None, float | None]:
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
                    if isinstance(v, dict): return float(v.get(idx) or v.get(idx.lower()) or 0.0)
                    return float(v[idx])
                except Exception:
                    return 0.0
            return (_get(last, 2) or None, _get(last, 3) or None)
        except Exception:
            return (None, None)

    async def run_limits(self, side: str, prev_high: float, prev_low: float, qty: float, sl: float, tp: float, close_market, wait_position_open, wait_position_flat, normalize_with_anchor):
        raw_qty = qty
        qty = self._coerce_qty(qty)
        if qty is None or qty <= 0:
            log.error(f"[EXT][LIM][QTY][ERR] cannot parse qty from {raw_qty!r}")
            self.stop_realign_task()
            return

        if qty <= 0:
            log.info("[EXT][LIM][SKIP] qty<=0")
            self.stop_realign_task()
            return

        f = self.account.ensure_filters()
        tick = float(f.get("tickSize", 0.1) or 0.1)
        eps = max(tick * self.eps_ticks, tick)
        cur_prev_high = float(prev_high or 0.0)
        cur_prev_low = float(prev_low or 0.0)

        active_oid = None
        for _ in range(120):  # до 2 часов переустановок
            if cur_prev_high <= 0 or cur_prev_low <= 0:
                ph, pl = await self._prev_hl_rest()
                if ph and pl:
                    cur_prev_high, cur_prev_low = ph, pl
                else:
                    log.info("[EXT][LIM][SKIP] no prev HL from REST; retry in 5s")
                    await asyncio.sleep(5); continue

            if side == "Buy":
                entry = ceil_step(cur_prev_low + eps, tick)
                tp_ref = float(tp) if (tp and tp > 0) else max(cur_prev_high - eps, entry + tick)
                sl_ref = float(sl) if (sl and sl > 0) else (entry - 1.5*tick)
            else:
                entry = round_step(cur_prev_high - eps, tick)
                tp_ref = float(tp) if (tp and tp > 0) else min(cur_prev_low + eps, entry - tick)
                sl_ref = float(sl) if (sl and sl > 0) else (entry + 1.5*tick)

            sl_adj, tp_adj = self.tpsl.fix_tpsl(side, entry, sl_ref, tp_ref, tick)
            log.info(f"[EXT][LIM][PLACE] {side} limit entry={fmt(entry)} tp={fmt(tp_adj)} sl={fmt(sl_adj)} qty={fmt(qty)}")

            if self.account.available <= 0 or self.account.equity <= 0:
                self.account.refresh_equity()

            attempt_qty = min(float(qty), self.risk.calc_qty(side, entry, sl_adj))
            min_qty = float(f.get("minQty", 0.001))
            if attempt_qty < min_qty:
                affordable = (self.account.available / (entry / max(self.account.leverage, 1.0))) >= min_qty
                if affordable:
                    attempt_qty = min_qty
                    log.info(f"[EXT][LIM][FORCE] adjusted qty up to min {fmt(min_qty)}")
                else:
                    log.error(f"[EXT][LIM][SKIP] qty not affordable at entry={fmt(entry)} -> {fmt(attempt_qty)} < min {fmt(min_qty)}")
                    self.stop_realign_task()
                    await asyncio.sleep(60)
                    cur_prev_high = cur_prev_low = 0.0
                    continue

            active_oid = None
            for _try in range(6):
                r = self.client.place_order(
                    self.symbol, side, attempt_qty, order_type="Limit",
                    price=entry, timeInForce="GTC", position_idx=0
                )
                rc = r.get("retCode")
                if rc == 0:
                    active_oid = r.get("result", {}).get("orderId", "")
                    log.info(f"[EXT][LIM][ORDER] id={active_oid} price={fmt(entry)} qty={fmt(attempt_qty)}")
                    break
                if rc == 110007:
                    try:
                        new_qty = float(self.risk.round_down_qty(float(attempt_qty) * 0.75))
                    except Exception:
                        new_qty = float(attempt_qty) * 0.75
                    if new_qty >= float(attempt_qty):
                        new_qty = self.risk.round_down_qty(float(attempt_qty) - float(f.get('qtyStep', 0.001)))
                        try:
                            new_qty = float(new_qty)
                        except Exception:
                            pass
                    if new_qty < min_qty:
                        log.error(f"[EXT][LIM][FAIL] 110007: insufficient balance, last qty={fmt(attempt_qty)} < min {fmt(min_qty)}")
                        active_oid = None
                        break
                    log.info(f"[EXT][LIM][RETRY_QTY] 110007 -> {fmt(attempt_qty)} -> {fmt(new_qty)}")
                    attempt_qty = new_qty
                    continue
                log.error(f"[EXT][LIM][FAIL] place_order {r}")
                active_oid = None
                break

            if not active_oid:
                self.stop_realign_task()
                await asyncio.sleep(60)
                cur_prev_high = cur_prev_low = 0.0
                continue

            end_ts = time.time() + 60
            filled = False
            while time.time() < end_ts:
                if await wait_position_open(timeout=1.0, interval=0.25):
                    filled = True
                    break
                await asyncio.sleep(0.5)

            if not filled:
                try:
                    self.cancel_order(order_id=active_oid)
                except Exception:
                    pass
                log.warning("[EXT][LIM][REPLACE] no fill in 60s — re-evaluate last closed HL")
                ph, pl = await self._prev_hl_rest()
                cur_prev_high = float(ph or 0.0)
                cur_prev_low = float(pl or 0.0)
                continue

            # позиция открыта — продолжаем
            try:
                pos = self.client.position_list(self.symbol)
                items = pos.get("result", {}).get("list", [])
                actual_side = (items[0].get("side") if items else side) or side
                base_price = float((items[0].get("avgPrice") if items else 0.0) or 0.0) or entry
            except Exception:
                actual_side, base_price = side, entry

            sl_final, tp_final = normalize_with_anchor(actual_side, base_price, sl_adj, tp_adj, tick)
            log.info(f"[EXT][LIM][TPSL] side={actual_side} base={fmt(base_price)} sl={fmt(sl_final)} tp={fmt(tp_final)} (anchor=Last/base)")

            ps, sz = self.get_pos_side_size()
            if ps and sz > 0:
                tr = self.client.trading_stop(
                    self.symbol, side=actual_side, stop_loss=sl_final, take_profit=tp_final,
                    tpslMode="Full", tpTriggerBy="LastPrice", slTriggerBy="MarkPrice",
                    tpOrderType="Market", positionIdx=0,
                )
                rc = tr.get("retCode")
                tag = "OK" if rc in (0, None) else ("UNCHANGED" if rc == 34040 else f"RC{rc}")
                log.info(f"[EXT][LIM][TPSL][{tag}] sl={fmt(sl_final)} tp={fmt(tp_final)}")

            # запустить реалайнер на финальных уровнях (после нормализации якорем)
            self.stop_realign_task()
            self.set_realign_task(actual_side, sl_final, tp_final, tick)

            # сторож до закрытия
            log.info("[EXT][LIM][WATCH] arming TP/SL watchdog (LastPrice cross)")
            while True:
                ok_flat = await self.tpsl.watchdog_close_on_last(
                    actual_side, sl_final, tp_final, close_side=("Sell" if actual_side=="Buy" else "Buy"),
                    check_interval=0.25, max_wait=300.0
                )
                if ok_flat:
                    break
                log.warning("[EXT][LIM][WATCH][TIMEOUT] position still open; keep monitoring…")

            self.stop_realign_task()
            try:
                ca = self.cancel_all()
                if ca.get("retCode") in (0, None):
                    log.info("[EXT][LIM][CANCEL_ALL][OK]")
            except Exception:
                pass

            cur_prev_high = cur_prev_low = 0.0
            active_oid = None
            continue

        log.warning("[EXT][LIM][ABORT] max cycles reached; stop")
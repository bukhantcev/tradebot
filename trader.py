# trader.py
import math
import logging
from typing import Optional, Dict, Any
import asyncio
import os
import time
from datetime import datetime, timezone
from config import SYMBOL, RISK_PCT, LEVERAGE
from bybit_client import BybitClient

log = logging.getLogger("TRADER")


class Trader:
    __lp_fail = 0
    """
    Минималистичный трейдер:
    - refresh_equity(): получить equity (USDT) одной строкой
    - ensure_filters(): разово подтянуть tickSize/qtyStep/min
    - ensure_leverage(): ставит плечо только если отличается
    - open_market(side, signal): маркет-вход + TPSL
    - close_market(side, qty): маркет-выход
    Опциональный notifier: объект с notify(str).
    """

    def __init__(
        self,
        client: BybitClient,
        symbol: str = SYMBOL,
        risk_pct: float = RISK_PCT,
        leverage: float = LEVERAGE,
        notifier: Optional[Any] = None,
    ):
        self.client = client
        self.symbol = symbol
        self.risk_pct = float(risk_pct)
        self.leverage = float(leverage)
        self.notifier = notifier

        # кэш маркет-фильтров
        self._filters: Optional[Dict[str, float]] = None
        # последняя известная equity (USDT)
        self.equity: float = 0.0
        self.available: float = 0.0  # доступный баланс для маржи (USDT)
        # режим входа по экстремам предыдущей закрытой свечи (выкл. по умолчанию)
        self.entry_extremes: bool = os.getenv("ENTRY_EXTREMES", "0") == "1"
        # сузить уровни на N тиков внутрь (экстрем-режим)
        try:
            self.ext_eps_ticks = int(os.getenv("EXT_EPS_TICKS", "2"))
        except Exception:
            self.ext_eps_ticks = 2
        # таск минутного логгера
        self._minute_task = None
        self._minute_mode: str = "normal"
        self._minute_sl: float | None = None
        self._realign_task = None  # background TPSL realigner task

    def _round_down_qty(self, qty: float) -> float:
        """Округляет вниз с учётом шага количества; запасной шаг 0.001."""
        try:
            f = self.ensure_filters()
            step = float(f.get("qtyStep", 0.001))
        except Exception:
            step = 0.001
        if step <= 0:
            step = 0.001
        n = int(qty / step)
        return max(step, n * step)

    def _start_minute_logger(self):
        """Запускает один общий минутный логгер; режим/SL берутся из self._minute_mode/_minute_sl."""
        if self._minute_task and not self._minute_task.done():
            return

        async def _loop():
            last_min = int(time.time() // 60)
            while True:
                try:
                    # sleep first to avoid hot loop on immediate reschedules
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
                        log.info(
                            f"[STAT][MIN] mode={mode} px={self._fmt(px)} pos={side or 'Flat'} size={self._fmt(sz)} SL={(self._fmt(sl_price) if sl_price else 'None')}"
                        )
                except asyncio.CancelledError:
                    log.info("[MIN][LOOP] cancelled")
                    break
                except Exception:
                    # backoff on any unexpected exception to avoid tight spinning
                    await asyncio.sleep(2.0)

        self._minute_task = asyncio.create_task(_loop(), name="minute_stat")
        log.info("[MIN][START] minute logger started")

    def set_minute_status(self, mode: str, sl_price: float | None):
        self._minute_mode = mode
        self._minute_sl = sl_price
        log.info(f"[MIN][MODE] set mode={mode} SL={(self._fmt(sl_price) if sl_price else 'None')}")
        self._start_minute_logger()

    async def _stop_minute_logger(self):
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

    # ---------- УТИЛЫ ----------

    def _round_step(self, value: float, step: float) -> float:
        if step <= 0:
            return value
        return math.floor(value / step + 1e-12) * step

    def _ceil_step(self, value: float, step: float) -> float:
        if step <= 0:
            return value
        return math.ceil(value / step - 1e-12) * step

    def _fmt(self, x: float) -> str:
        return f"{x:.6f}".rstrip("0").rstrip(".")

    def _fix_tpsl(self, side: str, price: float, sl: float, tp: float, tick: float) -> (float, float):
        """
        Нормализует SL/TP относительно рыночной цены и тика, чтобы не ловить 30208 на order.create.
          • Buy:  SL < price, TP > price  (минимум на 1 тик от цены)
          • Sell: SL > price, TP < price
        Возвращает кортеж (sl_fixed, tp_fixed) с корректировкой на 1–2 тика при необходимости.
        """
        p = float(price)
        sl_f, tp_f = float(sl), float(tp)
        t = max(float(tick), 0.0) or 0.1

        if side == "Buy":
            if sl_f >= p:
                sl_f = p - t
            if tp_f <= p:
                tp_f = p + t
            sl_f = self._round_step(sl_f, t)
            tp_f = self._ceil_step(tp_f, t)
            if sl_f >= p:
                sl_f = p - 2 * t
            if tp_f <= p:
                tp_f = p + 2 * t
        else:  # Sell
            if sl_f <= p:
                sl_f = p + t
            if tp_f >= p:
                tp_f = p - t
            sl_f = self._ceil_step(sl_f, t)
            tp_f = self._round_step(tp_f, t)
            if sl_f <= p:
                sl_f = p + 2 * t
            if tp_f >= p:
                tp_f = p - 2 * t

        return sl_f, tp_f

    def _normalize_tpsl_with_anchor(self, side: str, base_price: float, sl: float, tp: float, tick: float) -> tuple[float, float]:
        """
        Корректирует SL/TP с учётом текущей рыночной цены (LastPrice) как «якоря».
        Затем прогоняем через _fix_tpsl(...), чтобы вписаться в тик.
        """
        try:
            last = float(self._last_price()) or 0.0
        except Exception:
            last = 0.0

        anchor = float(base_price or 0.0)
        if side == "Buy":
            if last > 0:
                anchor = max(anchor, last)
            if tp <= anchor:
                tp = self._ceil_step(anchor + tick, tick)
            if sl >= anchor:
                sl = self._round_step(anchor - tick, tick)
        else:  # Sell
            if last > 0:
                anchor = min(anchor, last) if anchor > 0 else last
            if tp >= anchor:
                tp = self._round_step(anchor - tick, tick)
            if sl <= anchor:
                sl = self._ceil_step(anchor + tick, tick)

        sl_f, tp_f = self._fix_tpsl(side, anchor if anchor > 0 else (base_price or last or tp), sl, tp, tick)
        return sl_f, tp_f

    # ---------- ПУБЛИЧНЫЕ МЕТОДЫ ----------

    def refresh_equity(self) -> float:
        """Запрос баланса: totalEquity и totalAvailableBalance"""
        try:
            r = self.client.wallet_balance(account_type="UNIFIED")
            lst = r["result"]["list"][0]
            usdt_total = float(lst["totalEquity"])
            usdt_avail = float(lst.get("totalAvailableBalance") or lst.get("availableBalance") or usdt_total)
            self.equity = usdt_total
            self.available = usdt_avail
            log.info(f"[BALANCE] equity={usdt_total:.2f} avail={usdt_avail:.2f} USDT")
            if self.notifier:
                try:
                    asyncio.create_task(self.notifier.notify(f"💰 Баланс: {usdt_total:.2f} USDT (доступно {usdt_avail:.2f})"))
                except Exception:
                    pass
            return usdt_total
        except Exception as e:
            log.error(f"[BALANCE][ERR] {e}")
            return 0.0

    def ensure_filters(self) -> Dict[str, float]:
        """Получить tickSize / qtyStep / minOrderQty (кэшируется)"""
        if self._filters:
            return self._filters
        r = self.client.instruments_info(category="linear", symbol=self.symbol)
        it = r.get("result", {}).get("list", [{}])[0]
        tick = float(it.get("priceFilter", {}).get("tickSize", "0.1"))
        qty_step = float(it.get("lotSizeFilter", {}).get("qtyStep", "0.001"))
        min_qty = float(it.get("lotSizeFilter", {}).get("minOrderQty", "0.001"))
        mov = float(it.get("lotSizeFilter", {}).get("minOrderValue", "0") or 0.0)
        self._filters = {"tickSize": tick, "qtyStep": qty_step, "minQty": min_qty, "minNotional": mov}
        log.debug(f"[FILTERS] {self._filters}")
        return self._filters

    def ensure_leverage(self):
        """Ставит плечо только если отличается (подавляем 110043 как норму)"""
        try:
            pl = self.client.position_list(self.symbol)
            cur_lev = float(pl.get("result", {}).get("list", [{}])[0].get("leverage") or 0.0)
            if abs(cur_lev - self.leverage) < 1e-9:
                log.debug(f"[LEV] already {cur_lev}x")
                return
        except Exception as e:
            log.debug(f"[LEV] read failed: {e}")
        r = self.client.set_leverage(self.symbol, self.leverage, self.leverage)
        rc = r.get("retCode")
        if rc in (0, 110043):  # 110043 = leverage not modified
            log.info(f"[LEV] {self.leverage}x OK")
        else:
            log.warning(f"[LEV] rc={rc} msg={r.get('retMsg')}")

    # ---------- РАСЧЁТ QTY ----------

    def _calc_qty(self, side: str, price: float, sl: float) -> float:
        """
        Считаем qty как минимум из:
          • риск-бейзд:  qty_risk = (equity * risk_pct) / stop_dist
          • по доступной марже: qty_afford = available / (price/leverage * fee_buf)
        Затем приводим к шагу qtyStep и проверяем минимальные ограничения.
        """
        f = self.ensure_filters()
        stop_dist = abs(price - sl)
        if stop_dist <= 0:
            return 0.0

        # 1) по риску
        risk_amt = max(self.equity * self.risk_pct, 0.0)
        qty_risk = risk_amt / stop_dist

        # 2) по доступной марже (буфер на комиссии/маржу)
        FEE_BUF = 1.003
        margin_per_qty = price / max(self.leverage, 1.0)
        if margin_per_qty <= 0:
            return 0.0
        qty_afford = (self.available / (margin_per_qty * FEE_BUF)) if self.available > 0 else qty_risk

        # 3) итог
        raw = max(0.0, min(qty_risk, qty_afford))
        qty = self._round_step(raw, f["qtyStep"])
        if qty < f["minQty"]:
            return 0.0
        min_notional = f.get("minNotional", 0.0) or 0.0
        if min_notional > 0 and (qty * price) < min_notional:
            return 0.0
        return qty

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

    async def _watchdog_close_on_lastprice(self, side: str, sl_price: float, tp_price: float, check_interval: float = 0.3, max_wait: float = 3600.0) -> bool:
        """
        Сторож: если LastPrice пересекает TP/SL, закрываем позицию маркетом (fallback на случай,
        когда биржевой TP/SL с MarkPrice не срабатывает). Возвращает True, если позиция стала flat.
        """
        deadline = time.monotonic() + max_wait
        side = "Buy" if side == "Buy" else "Sell"
        opp = self._opposite(side)

        while time.monotonic() < deadline:
            try:
                # если уже flat — всё, выходим
                ps, sz = self._position_side_and_size()
                if not ps or sz <= 0:
                    return True

                last = self._last_price()
                if last <= 0:
                    await asyncio.sleep(check_interval)
                    continue

                trigger = False
                if side == "Buy":
                    # SL: last <= sl ; TP: last >= tp
                    if (sl_price and last <= sl_price) or (tp_price and last >= tp_price):
                        trigger = True
                else:
                    # Sell: SL: last >= sl ; TP: last <= tp
                    if (sl_price and last >= sl_price) or (tp_price and last <= tp_price):
                        trigger = True

                if trigger:
                    log.info(f"[WATCH][CROSS] Last={self._fmt(last)} vs SL={self._fmt(sl_price)} / TP={self._fmt(tp_price)} -> force close")
                    # stop any TP/SL realigner before force close
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

    def _cancel_realigner(self):
        """Cancel TPSL realigner task if running."""
        t = getattr(self, "_realign_task", None)
        if t and not t.done():
            try:
                t.cancel()
            except Exception:
                pass
        self._realign_task = None

    async def _realign_tpsl(self, side: str, desired_sl: float, desired_tp: float, tick: float, debounce: float = 0.8, max_tries: int = 30):
        """
        Periodically try to re-apply TP/SL to desired levels, but normalized
        against the *current* anchor (Last/Mark) to satisfy Bybit constraints.
        Stops automatically when position becomes flat or after max_tries.
        """
        try:
            side = "Buy" if side == "Buy" else "Sell"
            tries = 0
            while tries < max_tries:
                tries += 1

                # if flat — stop
                ps, sz = self._position_side_and_size()
                if not ps or sz <= 0:
                    log.debug("[REALIGN] flat — stop")
                    break

                # pick dynamic anchor (use last/base inside normalization)
                # and bring desired targets into allowed ranges
                # NOTE: we pass base_price=None; _normalize_tpsl_with_anchor will use LastPrice
                sl_norm, tp_norm = self._normalize_tpsl_with_anchor(side, base_price=0.0, sl=desired_sl, tp=desired_tp, tick=tick)

                # apply
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
                        # If we successfully set within 2 ticks of desired, consider done
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
        """
        Ждём исполнения текущего ордера. Если он отменён без фила — делаем до двух ретраев
        с умным маркетом и повышением допусков.
        Возвращает True, если позиция открылась; иначе False.
        """
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
                oid = r.get("result", {}).get("orderId", "")
                log.info(f"[ORDER][RETRY1] OK id={oid}")
                if await self._wait_position_open(timeout=10.0, interval=0.3):
                    return True

            r = self.client.place_market_safe(self.symbol, side, qty, position_idx=0, slip_percent=0.20)
            if r.get("retCode") == 0:
                oid = r.get("result", {}).get("orderId", "")
                log.info(f"[ORDER][RETRY2] OK id={oid}")
                if await self._wait_position_open(timeout=10.0, interval=0.3):
                    return True

            log.error("[ORDER][FAIL] no fill after retries")
            return False

        for _ in range(20):
            if await self._wait_position_open(timeout=1.0, interval=0.3):
                return True
        return False

    # ---------- ТИКИ / РЫНОЧНАЯ ЦЕНА ----------
    def _last_price(self) -> float:
        """Быстрый опрос последней цены через /v5/market/tickers (без лишних полей)."""
        try:
            r = self.client._request("GET", "/v5/market/tickers", params={"category": "linear", "symbol": self.symbol})
            it = (r.get("result", {}) or {}).get("list", [])
            if it:
                self.__lp_fail = 0
                return float(it[0].get("lastPrice") or 0.0)
        except Exception:
            pass
        self.__lp_fail += 1
        if self.__lp_fail % 20 == 0:
            log.warning(f"[EXT][LP] lastPrice=0 (#{self.__lp_fail})")
        return 0.0

    def _position_side_and_size(self) -> tuple[str | None, float]:
        """Возвращает (side, size) по текущей позиции или (None, 0.0)."""
        try:
            pl = self.client.position_list(self.symbol)
            lst = pl.get("result", {}).get("list", [])
            for it in lst:
                size = float(it.get("size") or 0.0)
                if size > 0:
                    return (it.get("side"), size)
        except Exception:
            pass
        return (None, 0.0)

    def _opposite(self, side: str) -> str:
        return "Sell" if side == "Buy" else "Buy"

    # ---------- ЛИМИТНЫЕ ОРДЕРА ПО ЭКСТРЕМАМ ----------
    async def _enter_extremes_with_limits(self, side: str, prev_high: float, prev_low: float, qty: float, sl: float, tp: float):
        """
        Экстрем-режим через ЛИМИТНЫЕ ордера с динамической перестановкой:
          • Каждую минуту получаем экстремумы предыдущей ЗАКРЫТОЙ 1m-свечи через REST /v5/market/kline.
          • Ставим лимит на вход чуть внутри уровня (ε = EXT_EPS_TICKS * tickSize).
          • Ждём до 60с. Если не исполнился — отменяем, переставляем по новым экстремам и повторяем.
          • После фила — ставим TP/SL через trading-stop (Full) и ждём закрытия позиции, затем ставим новый лимитник.
        """
        if qty <= 0:
            log.info("[EXT][LIM][SKIP] qty<=0")
            self._cancel_realigner()
            return

        f = self.ensure_filters()
        tick = float(f.get("tickSize", 0.1) or 0.1)
        eps_ticks = max(1, int(self.ext_eps_ticks))
        eps = max(tick * eps_ticks, tick)

        async def _prev_hl_rest() -> tuple[float | None, float | None]:
            """Быстро получаем экстремумы предыдущей ЗАКРЫТОЙ 1m-свечи через /v5/market/kline."""
            try:
                r = self.client._request(
                    "GET",
                    "/v5/market/kline",
                    params={"category": "linear", "symbol": self.symbol, "interval": "1", "limit": 3},
                )
                kl = (r.get("result", {}) or {}).get("list", [])
                if not kl:
                    return (None, None)
                if len(kl) >= 2:
                    last = kl[-2]
                else:
                    last = kl[-1]

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
        max_cycles = 120  # максимум 2 часа перестановок по 1 минуте

        for cycle in range(max_cycles):
            # 1) HL предыдущей свечи
            if cur_prev_high <= 0 or cur_prev_low <= 0:
                ph, pl = await _prev_hl_rest()
                if ph and pl:
                    cur_prev_high, cur_prev_low = ph, pl
                else:
                    log.info("[EXT][LIM][SKIP] no prev HL from REST; retry in 5s")
                    await asyncio.sleep(5)
                    continue

            # 2) Рассчитать вход и целевые SL/TP
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

            # 3) Баланс и qty
            try:
                if self.available <= 0 or self.equity <= 0:
                    self.refresh_equity()
            except Exception:
                pass
            attempt_qty = min(qty, self._calc_qty(side, entry_price, sl_adj))
            f_loc = self.ensure_filters()
            min_qty = float(f_loc.get("minQty", 0.001))
            if attempt_qty < min_qty:
                # try to use min_qty if affordable
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

            # 4) Постановка лимитника с ретраями 110007
            active_oid = None
            for _try in range(6):
                r = self.client.place_order(
                    self.symbol,
                    side,
                    attempt_qty,
                    order_type="Limit",
                    price=entry_price,
                    timeInForce="GTC",
                    position_idx=0,
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

            # 5) Ждать до 60с исполнения
            end_ts = time.time() + 60
            filled = False
            while time.time() < end_ts:
                if await self._wait_position_open(timeout=1.0, interval=0.25):
                    filled = True
                    break
                await asyncio.sleep(0.5)

            if not filled:
                try:
                    self._cancel_order(order_id=active_oid)
                except Exception:
                    pass
                log.warning("[EXT][LIM][REPLACE] no fill in 60s — re-evaluate last closed HL")
                ph, pl = await _prev_hl_rest()
                cur_prev_high = float(ph or 0.0)
                cur_prev_low = float(pl or 0.0)
                continue

            # 6) Исполнилось — определить фактический side/price и поставить TP/SL
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

            # Apply TPSL only if position is not flat (guard against race right after fill)
            ps, sz = self._position_side_and_size()
            if not ps or sz <= 0:
                log.debug("[EXT][LIM][TPSL][SKIP] position flat right after fill")
            else:
                tr = self.client.trading_stop(
                    self.symbol,
                    side=actual_side,
                    stop_loss=sl_final,
                    take_profit=tp_final,
                    tpslMode="Full",
                    tpTriggerBy="LastPrice",
                    slTriggerBy="MarkPrice",
                    tpOrderType="Market",
                    positionIdx=0,
                )
                rc = tr.get("retCode")
                if rc in (0, None, 34040):  # 34040 = not modified
                    tag = "OK" if rc in (0, None) else "UNCHANGED"
                    log.info(f"[EXT][LIM][TPSL][{tag}] sl={self._fmt(sl_final)} tp={self._fmt(tp_final)}")
                else:
                    log.warning(f"[EXT][LIM][TPSL][FAIL] {tr}")

            # Start background realigner to move TP/SL toward desired (normalized on the fly)
            self._cancel_realigner()
            try:
                self._realign_task = asyncio.create_task(
                    self._realign_tpsl(actual_side, sl_adj, tp_adj, tick),
                    name="tpsl_realign"
                )
            except Exception:
                self._realign_task = None

            # --- Ждём закрытия позиции TP/SL и перезапускаем цикл с новым лимитником ---
            log.info("[EXT][LIM][WATCH] arming TP/SL watchdog (LastPrice cross)")
            # Не останавливаем рабочий цикл из-за таймаута.
            # Следим за TP/SL батчами по 5 минут; если таймаут — просто продолжаем мониторить,
            # пока позиция не станет flat.
            while True:
                ok_flat = await self._watchdog_close_on_lastprice(
                    actual_side, sl_final, tp_final, check_interval=0.25, max_wait=300.0
                )
                if ok_flat:
                    break
                log.warning("[EXT][LIM][WATCH][TIMEOUT] position still open; keep monitoring…")
            # Ensure realigner is stopped once flat
            self._cancel_realigner()

            # На всякий случай снимем все оставшиеся ордера по символу
            try:
                ca = self._cancel_all_orders()
                rc = ca.get("retCode")
                if rc in (0, None):
                    log.info("[EXT][LIM][CANCEL_ALL][OK]")
                else:
                    log.warning(f"[EXT][LIM][CANCEL_ALL][WARN] rc={rc} msg={ca.get('retMsg')}")
            except Exception as e:
                log.warning(f"[EXT][LIM][CANCEL_ALL][EXC] {e}")

            # Обновим экстремумы на следующей итерации и продолжим цикл
            cur_prev_high = 0.0
            cur_prev_low = 0.0
            active_oid = None
            continue

        log.warning("[EXT][LIM][ABORT] max cycles reached; stop")

    # вспомогательная проверка статуса лимитника по orderId
    def _order_status_brief(self, order_id: str) -> str:
        try:
            st = self.client.get_order_status(self.symbol, order_id)
            s = (st.get("status") or "").lower()
            ce = float(st.get("cumExecQty") or 0.0)
            q = float(st.get("qty") or 0.0)
            return f"{s or 'n/a'} {ce}/{q}"
        except Exception:
            return "n/a"

    # NOTE: старая ветка с условными заявками (не используется в live-follow)
    def _place_conditional(self, side: str, trigger_price: float, qty: float, trigger_direction: int) -> Dict[str, Any]:
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

    def _cancel_order(self, order_id: Optional[str] = None, order_link_id: Optional[str] = None) -> Dict[str, Any]:
        body = {
            "category": "linear",
            "symbol": self.symbol,
        }
        if order_id:
            body["orderId"] = order_id
        if order_link_id:
            body["orderLinkId"] = order_link_id
        return self.client._request("POST", "/v5/order/cancel", body=body)

    def _cancel_all_orders(self) -> Dict[str, Any]:
        """
        Отмена всех активных/условных ордеров по символу.
        """
        body = {
            "category": "linear",
            "symbol": self.symbol,
        }
        return self.client._request("POST", "/v5/order/cancel-all", body=body)

    # ---------- ОРДЕРА ----------

    async def _enter_by_extremes(self, side: str, prev_high: float, prev_low: float, qty: float, sl_r: float, tp_r: float):
        """
        (не используется в live-follow) Ставит две условные заявки.
        """
        if qty <= 0 or prev_high <= 0 or prev_low <= 0:
            log.info("[COND][SKIP] bad params")
            return

        r_sell = self._place_conditional("Sell", prev_high, qty, trigger_direction=1)
        r_buy = self._place_conditional("Buy", prev_low, qty, trigger_direction=2)

        oid_sell = r_sell.get("result", {}).get("orderId") if r_sell.get("retCode") == 0 else None
        oid_buy = r_buy.get("result", {}).get("orderId") if r_buy.get("retCode") == 0 else None

        if r_sell.get("retCode") != 0:
            log.warning(f"[COND][ERR] sell {r_sell}")
        if r_buy.get("retCode") != 0:
            log.warning(f"[COND][ERR] buy {r_buy}")

        if self.notifier:
            try:
                await self.notifier.notify(f"⏳ Cond placed: Sell@{self._fmt(prev_high)} / Buy@{self._fmt(prev_low)} qty={self._fmt(qty)}")
            except Exception:
                pass

        ok = await self._wait_position_open(timeout=300.0, interval=0.5)
        if not ok:
            log.warning("[COND][TIMEOUT] no fill within 5m — cancel both")
            if oid_sell:
                self._cancel_order(order_id=oid_sell)
            if oid_buy:
                self._cancel_order(order_id=oid_buy)
            return

        if oid_sell:
            self._cancel_order(order_id=oid_sell)
        if oid_buy:
            self._cancel_order(order_id=oid_buy)

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

    async def open_market(self, side: str, signal: Dict[str, Any]):
        """
        side: "Buy" | "Sell"
        signal: { 'sl': float, 'tp': float, 'atr': float, 'ts_ms': int, ... }
        """
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
            await self._stop_minute_logger()
            return
        # --- ЕДИНСТВЕННЫЙ ОРДЕР/ПОЗИЦИЯ В СТОРОНУ ИИ ---
        can_enter = await self._enforce_single_exposure(side)
        if not can_enter:
            # Уже есть позиция в сторону ИИ — только realign TP/SL и выходим.
            try:
                f = self.ensure_filters()
                tick = f["tickSize"]
                # Если есть текущая позиция — подтянуть TP/SL к желаемым
                ps, sz = self._position_side_and_size()
                if ps and sz > 0:
                    base_price = float(self._last_price()) or float(price)
                    sl_r = self._ceil_step(sl, tick) if side == "Buy" else self._round_step(sl, tick)
                    tp_r = self._round_step(tp, tick) if side == "Buy" else self._ceil_step(tp, tick)
                    sl_adj, tp_adj = self._normalize_tpsl_with_anchor(ps, base_price, sl_r, tp_r, tick)
                    # попытка применить
                    r2 = self.client.trading_stop(
                        self.symbol,
                        side=ps,
                        stop_loss=sl_adj,
                        take_profit=tp_adj,
                        tpslMode="Full",
                        tpTriggerBy="LastPrice",
                        slTriggerBy="MarkPrice",
                        tpOrderType="Market",
                        positionIdx=0,
                    )
                    rc2 = r2.get("retCode")
                    tag = "OK" if rc2 in (0, None) else ("UNCHANGED" if rc2 == 34040 else f"RC={rc2}")
                    log.info(f"[TPSL][REALIGN] sl={self._fmt(sl_adj)} tp={self._fmt(tp_adj)} {tag}")
                    # запустить мягкий реалайнер
                    self._cancel_realigner()
                    try:
                        self._realign_task = asyncio.create_task(
                            self._realign_tpsl(ps, sl_r, tp_r, tick),
                            name="tpsl_realign"
                        )
                    except Exception:
                        self._realign_task = None
            except Exception as e:
                log.warning(f"[ONE][REALIGN][EXC] {e}")
            return

        self.set_minute_status("normal", float(sl) if sl else None)

        qty = self._calc_qty(side, price, sl)
        log.info(f"[QTY] risk%={self.risk_pct*100:.2f} stop={abs(price-sl):.2f} equity={self.equity:.2f} avail={self.available:.2f} -> qty={self._fmt(qty)}")
        if qty <= 0:
            log.info("[ENTER][SKIP] qty=0")
            self._cancel_realigner()
            await self._stop_minute_logger()
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
            self.set_minute_status("ext" if use_ext else "normal", float(sl) if sl else None)
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
            # жёстко соблюдаем «один ордер» и «в сторону ИИ»
            can_enter_ext = await self._enforce_single_exposure(side)
            if not can_enter_ext:
                log.info("[EXT][SKIP] already in AI side; no new limit order")
                await self._stop_minute_logger()
                return

            log.info(
                f"[EXT][LIM][FLOW] side={side} prevH={self._fmt(prev_high)} prevL={self._fmt(prev_low)} qty={self._fmt(qty)}")
            try:
                await self._enter_extremes_with_limits(side, float(prev_high), float(prev_low), qty, sl=float(sl), tp=float(tp))
                log.info("[EXT][RETURN] _enter_extremes_with_limits finished")
            except Exception as e:
                log.exception(f"[EXT][CRASH] _enter_extremes_with_limits exception: {e}")
            finally:
                await self._stop_minute_logger()
            return

        # --- Маркет-вход (обычный) ---
        log.info(f"[ENTER] {side} qty={self._fmt(qty)}")
        attempt_qty = qty
        order_id = None
        for attempt in range(6):
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
                    await self._stop_minute_logger()
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
                await self._stop_minute_logger()
                return
            log.error(f"[ENTER][FAIL] {r}")
            self._cancel_realigner()
            await self._stop_minute_logger()
            return

        if not order_id:
            log.error("[ENTER][FAIL] no orderId")
            self._cancel_realigner()
            await self._stop_minute_logger()
            return

        if not await self._await_fill_or_retry(order_id, side, attempt_qty):
            log.warning("[ENTER][ABORT] no fill")
            self._cancel_realigner()
            await self._stop_minute_logger()
            return

        # Сразу SL в Full-режиме (без TP) — но только если позиция реально открыта
        if sl:
            try:
                ps0, sz0 = self._position_side_and_size()
                if not ps0 or sz0 <= 0:
                    log.debug("[SL][NORM][SKIP] flat right after entry")
                else:
                    tr = self.client.trading_stop(
                        self.symbol,
                        side=side,
                        stop_loss=float(sl),
                        tpslMode="Full",
                        slTriggerBy="MarkPrice",
                        positionIdx=0,
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
                self.symbol,
                side=actual_side,
                stop_loss=sl_adj,
                take_profit=tp_adj,
                tpslMode="Full",
                tpTriggerBy="LastPrice",
                slTriggerBy="MarkPrice",
                tpOrderType="Market",
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
        # Start background realigner to migrate TP/SL toward desired values as the anchor allows
        self._cancel_realigner()
        try:
            self._realign_task = asyncio.create_task(
                self._realign_tpsl(actual_side, sl_r, tp_r, tick),
                name="tpsl_realign"
            )
        except Exception:
            self._realign_task = None

    async def close_market(self, side: str, qty: float):
        """
        Закрытие позиции маркетом (на твой выбор side / qty).
        """
        if qty <= 0:
            log.info("[EXIT][SKIP] qty=0")
            return
        log.info(f"[EXIT] {side} qty={self._fmt(qty)}")
        r = self.client.place_order(
            self.symbol,
            side,
            qty,
            order_type="Market",
            preferSmart=True,
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
                self.symbol,
                side,
                qty,
                order_type="Market",
                slippageToleranceType="Percent",
                slippageTolerance="0.05",
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

    # ---- SINGLE-EXPOSURE HELPERS ----
    def _active_orders_count(self) -> int:
        """Сколько активных (и conditional) ордеров по символу."""
        try:
            r = self.client._request("GET", "/v5/order/realtime", params={"category": "linear", "symbol": self.symbol})
            lst = (r.get("result", {}) or {}).get("list", []) or []
            return len(lst)
        except Exception:
            return 0

    async def _enforce_single_exposure(self, ai_side: str) -> bool:
        """
        Гарантирует: нет активных ордеров и позиция либо плоская, либо совпадает со стороной ИИ.
        Возвращает True, если можно продолжать постановку нового входа (flat или совпадает).
        Возвращает False, если уже открыта позиция В СТОРОНУ ИИ и вход НЕ НУЖЕН.
        """
        # 1) снять все отложенные ордера
        try:
            ca = self._cancel_all_orders()
            rc = ca.get("retCode")
            if rc not in (0, None):
                log.warning(f"[ONE][CANCEL_ALL][WARN] rc={rc} msg={ca.get('retMsg')}")
            else:
                log.info("[ONE][CANCEL_ALL][OK]")
        except Exception as e:
            log.warning(f"[ONE][CANCEL_ALL][EXC] {e}")

        # 2) проверить позицию
        ps, sz = self._position_side_and_size()
        if ps and sz > 0:
            if ps == ai_side:
                log.info(f"[ONE][KEEP] already in {ps} size={self._fmt(sz)} -> skip new entry; will realign TP/SL only")
                return False  # ничего нового открывать не надо
            else:
                # противоположная — закрываем
                log.info(f"[ONE][FLIP] have {ps} size={self._fmt(sz)} -> closing to follow AI={ai_side}")
                await self.close_market(self._opposite(ps), sz)
                ok = await self._wait_position_flat(timeout=30.0, interval=0.25)
                if not ok:
                    log.warning("[ONE][FLIP][WARN] position is not flat after close attempt")
        else:
            log.info("[ONE] flat — ok")

        # 3) убедиться, что нет активных ордеров
        for _ in range(10):
            if self._active_orders_count() == 0:
                break
            await asyncio.sleep(0.2)
        return True
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

    def _start_minute_logger(self, mode: str, sl_price: float | None):
        """Запускает компактный логгер раз в минуту: px, позиция, режим, SL."""
        if self._minute_task and not self._minute_task.done():
            return
        import asyncio, time
        async def _loop():
            last_min = int(time.time() // 60)
            while True:
                try:
                    m = int(time.time() // 60)
                    if m != last_min:
                        last_min = m
                        px = self._last_price()
                        side, sz = self._position_side_and_size()
                        log.info(f"[STAT][MIN] mode={mode} px={self._fmt(px)} pos={side or 'Flat'} size={self._fmt(sz)} SL={(self._fmt(sl_price) if sl_price else 'None')}")
                    await asyncio.sleep(0.25)
                except asyncio.CancelledError:
                    break
                except Exception:
                    await asyncio.sleep(0.5)
        self._minute_task = asyncio.create_task(_loop(), name="minute_stat")

    async def _stop_minute_logger(self):
        t = self._minute_task
        if t and not t.done():
            t.cancel()
            try:
                await t
            except Exception:
                pass
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
            # SL должен быть строго ниже цены, TP — строго выше
            if sl_f >= p:
                sl_f = p - t
            if tp_f <= p:
                tp_f = p + t
            # округление в сторону, совместимую с правилами биржи
            sl_f = self._round_step(sl_f, t)  # вниз по тику
            tp_f = self._ceil_step(tp_f, t)   # вверх по тику
            # дополнительная страховка: разнести хотя бы на 1 тик
            if sl_f >= p:
                sl_f = p - 2 * t
            if tp_f <= p:
                tp_f = p + 2 * t
        else:  # Sell
            if sl_f <= p:
                sl_f = p + t
            if tp_f >= p:
                tp_f = p - t
            sl_f = self._ceil_step(sl_f, t)   # вверх по тику
            tp_f = self._round_step(tp_f, t)  # вниз по тику
            if sl_f <= p:
                sl_f = p + 2 * t
            if tp_f >= p:
                tp_f = p - 2 * t

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
                    import asyncio
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
            # Bybit может отдавать пустой список, если позиции нет
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

        # 1) по риску (леверидж не влияет на риск в $)
        risk_amt = max(self.equity * self.risk_pct, 0.0)
        qty_risk = risk_amt / stop_dist

        # 2) по доступной марже (учтём буфер на комиссии/плавающие требования)
        FEE_BUF = 1.003
        margin_per_qty = price / max(self.leverage, 1.0)
        if margin_per_qty <= 0:
            return 0.0
        qty_afford = (self.available / (margin_per_qty * FEE_BUF)) if self.available > 0 else qty_risk

        # 3) итог
        raw = max(0.0, min(qty_risk, qty_afford))
        qty = self._round_step(raw, f["qtyStep"])
        # Минимальные ограничения
        if qty < f["minQty"]:
            return 0.0
        # Минимальная нотация (если биржа требует)
        min_notional = f.get("minNotional", 0.0) or 0.0
        if min_notional > 0 and (qty * price) < min_notional:
            return 0.0
        return qty

    async def _wait_position_open(self, timeout: float = 10.0, interval: float = 0.3) -> bool:
        """
        Короткое ожидание появления позиции (size&gt;0), чтобы fallback trading-stop не падал rc=10001.
        Возвращает True, если позиция открылась в течение timeout.
        """
        import time
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                pl = self.client.position_list(self.symbol)
                lst = pl.get("result", {}).get("list", [])
                # Пройдём по всем позициям (на случай hedge-mode/двух сторон)
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

    async def _await_fill_or_retry(self, order_id: str, side: str, qty: float) -> bool:
        """
        Ждём исполнения текущего ордера. Если он отменён без фила — делаем до двух ретраев
        с умным маркетом и повышением допусков.
        Возвращает True, если позиция открылась; иначе False.
        """
        # 1) Подождать быстрое появление позиции
        ok = await self._wait_position_open(timeout=10.0, interval=0.3)
        if ok:
            return True

        # 2) Узнать статус исходного ордера
        try:
            st = self.client.get_order_status(self.symbol, order_id)
        except Exception:
            st = {"status": None, "cumExecQty": 0.0, "qty": 0.0}
        status = (st.get("status") or "").lower()
        filled = float(st.get("cumExecQty") or 0.0) > 0.0

        if filled or status in ("filled", "partiallyfilled"):
            # иногда позиция появится с задержкой — проверим ещё чуть-чуть
            ok2 = await self._wait_position_open(timeout=5.0, interval=0.3)
            return ok2

        if status in ("cancelled", "rejected") or not status:
            log.warning(f"[ORDER][CANCELLED] status={status or 'n/a'} -> retry smart market")

            # Ретрай #1: умный маркет с 0.10% допуска
            r = self.client.place_market_safe(self.symbol, side, qty, position_idx=0, slip_percent=0.10)
            if r.get("retCode") == 0:
                oid = r.get("result", {}).get("orderId", "")
                log.info(f"[ORDER][RETRY1] OK id={oid}")
                if await self._wait_position_open(timeout=10.0, interval=0.3):
                    return True

            # Ретрай #2: 0.20%
            r = self.client.place_market_safe(self.symbol, side, qty, position_idx=0, slip_percent=0.20)
            if r.get("retCode") == 0:
                oid = r.get("result", {}).get("orderId", "")
                log.info(f"[ORDER][RETRY2] OK id={oid}")
                if await self._wait_position_open(timeout=10.0, interval=0.3):
                    return True

            log.error("[ORDER][FAIL] no fill after retries")
            return False

        # Если статус «new/created» — подождём еще немного
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

    def _position_side_and_size(self) -> tuple[str|None, float]:
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

    # ---------- УСЛОВНЫЕ ОРДЕРА ПО ЭКСТРЕМАМ ----------

    # ---------- ЖИВОЕ СОПРОВОЖДЕНИЕ ПО ЭКСТРЕМАМ (БЕЗ ОТЛОЖЕК) ----------
    async def _follow_extremes(self, side: str, prev_high: float, prev_low: float, qty: float):
        """
        НЕ ставим отложки. Следим за ценой в непрерывном цикле:
          • side == "Sell": при достижении текущего high-ε ⇒ Market Sell; далее при достижении текущего low+ε ⇒ Market Buy (фиксация), и снова цикл.
          • side == "Buy":  при достижении текущего low+ε  ⇒ Market Buy;  далее при достижении текущего high-ε ⇒ Market Sell (фиксация), и снова цикл.
        Экстремы текущей незакрытой 1m-свечи обновляются на каждом тике. ε — смещение внутрь уровней на EXT_EPS_TICKS * tickSize.
        Сигнал LLM задаёт только вектор (Buy/Sell). Цикл продолжается, пока включён экстрем-режим.
        """
        if qty <= 0 or prev_high <= 0 or prev_low <= 0:
            log.info("[EXT][SKIP] bad params")
            return

        log.info("[EXT][ENTER_FN] _follow_extremes started")

        # базовые величины
        f = self.ensure_filters()
        tick = float(f.get("tickSize", 0.1) or 0.1)
        eps = max(tick * max(1, int(self.ext_eps_ticks)), tick)  # минимум 1 тик внутрь

        log.info(f"[EXT][START] vec={side} prevH={self._fmt(prev_high)} prevL={self._fmt(prev_low)} qty={self._fmt(qty)} eps={self._fmt(eps)}")

        # Бесконечный цикл: вход -> выход -> вход ...
        while True:
            # если режим выключили снаружи — выходим
            if not self.entry_extremes:
                log.info("[EXT][STOP] flag off")
                return

            # --- Ожидание входа ---
            # Инициализация «живых» экстремов от переданных prev_high/low (сужаем внутрь на eps)
            cur_high = float(prev_high)
            cur_low = float(prev_low)
            want_entry_on = "high" if side == "Sell" else "low"
            want_exit_on = "low" if side == "Sell" else "high"

            cur_minute = int(time.time() // 60)
            last_minute_logged = cur_minute

            log.info(f"[EXT][WAIT_ENTER] vec={side} qty={self._fmt(qty)} startH={self._fmt(cur_high)} startL={self._fmt(cur_low)} eps={self._fmt(eps)} want={want_entry_on}")

            px = 0.0
            while True:
                px = self._last_price()
                if px <= 0:
                    await asyncio.sleep(0.2)
                    continue

                # обновляем «живые» экстремы в рамках минуты
                m = int(time.time() // 60)
                if m != cur_minute:
                    cur_minute = m
                    cur_high = px
                    cur_low = px
                else:
                    if px > cur_high:
                        cur_high = px
                    if px < cur_low:
                        cur_low = px

                # минутный лог
                if m != last_minute_logged:
                    last_minute_logged = m
                    log.info(f"[EXT][MINUTE][ENTER] curH={self._fmt(cur_high)} curL={self._fmt(cur_low)} px={self._fmt(px)} want={want_entry_on} eps={self._fmt(eps)}")

                # условия входа (сузим уровни внутрь)
                if want_entry_on == "high" and px >= (cur_high - eps):
                    log.info(f"[EXT][ENTER] Sell @ {self._fmt(px)} (trigger≥{self._fmt(cur_high - eps)})")
                    r = self.client.place_order(self.symbol, "Sell", qty, order_type="Market", preferSmart=True)
                    if r.get("retCode") != 0:
                        log.error(f"[EXT][ENTER][FAIL] {r}")
                        return
                    oid = r.get("result", {}).get("orderId", "")
                    if not await self._await_fill_or_retry(oid, "Sell", qty):
                        log.warning("[EXT][ENTER][ABORT] no fill")
                        return
                    # страховой SL сразу после входа — за внешний уровень
                    try:
                        sl_price = cur_high + 2 * tick
                        tr = self.client.trading_stop(self.symbol, side="Sell", stop_loss=float(sl_price), tpslMode="Full", positionIdx=0)
                        if tr.get("retCode") in (0, None):
                            log.info(f"[EXT][SL] set {self._fmt(sl_price)} OK")
                        else:
                            log.warning(f"[EXT][SL][FAIL] {tr}")
                    except Exception as e:
                        log.warning(f"[EXT][SL][EXC] {e}")
                    break

                if want_entry_on == "low" and px <= (cur_low + eps):
                    log.info(f"[EXT][ENTER] Buy @ {self._fmt(px)} (trigger≤{self._fmt(cur_low + eps)})")
                    r = self.client.place_order(self.symbol, "Buy", qty, order_type="Market", preferSmart=True)
                    if r.get("retCode") != 0:
                        log.error(f"[EXT][ENTER][FAIL] {r}")
                        return
                    oid = r.get("result", {}).get("orderId", "")
                    if not await self._await_fill_or_retry(oid, "Buy", qty):
                        log.warning("[EXT][ENTER][ABORT] no fill")
                        return
                    # страховой SL сразу после входа — за внешний уровень
                    try:
                        sl_price = cur_low - 2 * tick
                        tr = self.client.trading_stop(self.symbol, side="Buy", stop_loss=float(sl_price), tpslMode="Full", positionIdx=0)
                        if tr.get("retCode") in (0, None):
                            log.info(f"[EXT][SL] set {self._fmt(sl_price)} OK")
                        else:
                            log.warning(f"[EXT][SL][FAIL] {tr}")
                    except Exception as e:
                        log.warning(f"[EXT][SL][EXC] {e}")
                    break

                await asyncio.sleep(0.12)

            # --- Ожидание выхода (фиксации) ---
            # После входа обнулим экстремы от текущей цены и начнём следить заново
            cur_high = px
            cur_low = px
            cur_minute = int(time.time() // 60)
            last_minute_logged = cur_minute
            side_exit = self._opposite(side)
            log.info(f"[EXT][WAIT_EXIT] want={want_exit_on} sideExit={side_exit} eps={self._fmt(eps)}")

            while True:
                # если позицию закрыли снаружи — начинаем новый оборот цикла без входа (ждать новый вход)
                ps, sz = self._position_side_and_size()
                if not ps or sz <= 0:
                    log.info("[EXT][DONE] position closed externally — restart loop")
                    break

                px = self._last_price()
                if px <= 0:
                    await asyncio.sleep(0.2)
                    continue

                m = int(time.time() // 60)
                if m != cur_minute:
                    cur_minute = m
                    cur_high = px
                    cur_low = px
                else:
                    if px > cur_high:
                        cur_high = px
                    if px < cur_low:
                        cur_low = px

                # минутный лог
                if m != last_minute_logged:
                    last_minute_logged = m
                    log.info(f"[EXT][MINUTE][EXIT] curH={self._fmt(cur_high)} curL={self._fmt(cur_low)} px={self._fmt(px)} want={want_exit_on} eps={self._fmt(eps)}")

                # условия выхода (сузим уровни внутрь)
                if want_exit_on == "low" and px <= (cur_low + eps):
                    log.info(f"[EXT][EXIT] Buy @ {self._fmt(px)} (trigger≤{self._fmt(cur_low + eps)})")
                    await self.close_market("Buy", sz)
                    break

                if want_exit_on == "high" and px >= (cur_high - eps):
                    log.info(f"[EXT][EXIT] Sell @ {self._fmt(px)} (trigger≥{self._fmt(cur_high - eps)})")
                    await self.close_market("Sell", sz)
                    break

                await asyncio.sleep(0.12)

            # подготовим уровни следующего оборота
            prev_high = cur_high
            prev_low = cur_low

    # NOTE: ниже — старая ветка с условными заявками; для live-follow она не используется.
    def _place_conditional(self, side: str, trigger_price: float, qty: float, trigger_direction: int) -> Dict[str, Any]:
        """
        Ставит условный маркет-ордер (IOC) по достижению trigger_price.
        trigger_direction: 1 = триггер при росте до цены, 2 = при падении до цены.
        Возвращает ответ API.
        """
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
        # Уникальный orderLinkId для удобной отмены/отслеживания
        body["orderLinkId"] = f"ext-{int(time.time()*1000)}-{side[0].lower()}"
        return self.client._request("POST", "/v5/order/create", body=body)

    def _cancel_order(self, order_id: Optional[str] = None, order_link_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Отмена одного активного/условного ордера.
        """
        body = {
            "category": "linear",
            "symbol": self.symbol,
        }
        if order_id:
            body["orderId"] = order_id
        if order_link_id:
            body["orderLinkId"] = order_link_id
        return self.client._request("POST", "/v5/order/cancel", body=body)

    # ---------- ОРДЕРА ----------

    async def _enter_by_extremes(self, side: str, prev_high: float, prev_low: float, qty: float, sl_r: float, tp_r: float):
        """
        Ставит две условные заявки:
          • Sell при достижении prev_high (triggerDirection=1),
          • Buy при достижении prev_low (triggerDirection=2).
        Как только одна исполняется — вторая отменяется. Затем ставим TP/SL через trading-stop.
        """
        # sanity
        if qty <= 0 or prev_high <= 0 or prev_low <= 0:
            log.info("[COND][SKIP] bad params")
            return

        # 1) Ставим оба условных
        r_sell = self._place_conditional("Sell", prev_high, qty, trigger_direction=1)
        r_buy  = self._place_conditional("Buy",  prev_low,  qty, trigger_direction=2)

        oid_sell = r_sell.get("result", {}).get("orderId") if r_sell.get("retCode") == 0 else None
        oid_buy  = r_buy.get("result", {}).get("orderId")  if r_buy.get("retCode") == 0 else None

        if r_sell.get("retCode") != 0:
            log.warning(f"[COND][ERR] sell {r_sell}")
        if r_buy.get("retCode") != 0:
            log.warning(f"[COND][ERR] buy {r_buy}")

        if self.notifier:
            try:
                await self.notifier.notify(f"⏳ Cond placed: Sell@{self._fmt(prev_high)} / Buy@{self._fmt(prev_low)} qty={self._fmt(qty)}")
            except Exception:
                pass

        # 2) Ждём, пока появится позиция (значит одна из заявок сработала)
        ok = await self._wait_position_open(timeout=300.0, interval=0.5)
        if not ok:
            log.warning("[COND][TIMEOUT] no fill within 5m — cancel both")
            if oid_sell:
                self._cancel_order(order_id=oid_sell)
            if oid_buy:
                self._cancel_order(order_id=oid_buy)
            return

        # 3) Позиция появилась — снимаем вторую заявку (если ещё активна)
        if oid_sell:
            self._cancel_order(order_id=oid_sell)
        if oid_buy:
            self._cancel_order(order_id=oid_buy)

        # 4) Определяем фактическую сторону/цену позиции и нормализуем TP/SL
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

        # 5) Ставим TP/SL для открытой позиции
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
        # Унифицированный доступ к полям сигнала (dict или dataclass/объект)
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

        # обновим equity/available (если 0) и фильтры/плечо
        if self.equity <= 0 or self.available <= 0:
            self.refresh_equity()
        self.ensure_filters()
        self.ensure_leverage()

        price = float(_sg_multi(["price", "close"]) or 0.0)
        if price <= 0:
            # берём близкую оценку — без отдельного REST-запроса
            price = float(_sg("tp") or 0.0) or 1.0

        sl = float(_sg_multi(["sl", "stop_loss", "stopLoss"]) or 0.0)
        tp = float(_sg_multi(["tp", "take_profit", "takeProfit"]) or 0.0)
        if sl <= 0 or tp <= 0:
            log.info("[ENTER][SKIP] bad SL/TP")
            await self._stop_minute_logger()
            return

        # старт минутного логгера (режим уточним ниже)
        self._start_minute_logger("normal", float(sl) if sl else None)

        qty = self._calc_qty(side, price, sl)
        log.info(f"[QTY] risk%={self.risk_pct*100:.2f} stop={abs(price-sl):.2f} equity={self.equity:.2f} avail={self.available:.2f} -> qty={self._fmt(qty)}")
        if qty <= 0:
            log.info("[ENTER][SKIP] qty=0")
            await self._stop_minute_logger()
            return

        # Подготовим округлённые SL/TP заранее (для передачи в order/create)
        f = self.ensure_filters()
        tick = f["tickSize"]
        sl_r = self._ceil_step(sl, tick) if side == "Buy" else self._round_step(sl, tick)
        tp_r = self._round_step(tp, tick) if side == "Buy" else self._ceil_step(tp, tick)
        sl_r, tp_r = self._fix_tpsl(side, price, sl_r, tp_r, tick)

        # --- Диагностика режима экстремов ---
        prev_high = _sg_multi(["prev_high", "prevHigh", "prevH", "previous_high", "prev_high_price"])
        prev_low  = _sg_multi(["prev_low", "prevLow", "prevL", "previous_low", "prev_low_price"])
        use_ext = bool(self.entry_extremes and prev_high and prev_low)
        if use_ext:
            log.info(f"[EXT][MODE] ON  prevH={self._fmt(prev_high)} prevL={self._fmt(prev_low)} qty={self._fmt(qty)}")
        else:
            reason = []
            if not self.entry_extremes:
                reason.append("flag_off")
            if not prev_high or not prev_low:
                reason.append("no_prev_hl")
            # Показать доступные поля сигнала для дебага (без значений)
            try:
                fields = list(signal.keys()) if isinstance(signal, dict) else list(vars(signal).keys())
            except Exception:
                fields = ["?"]
            log.info(f"[EXT][MODE] OFF ({','.join(reason) if reason else 'n/a'}) fields={fields}")

        # обновим режим минутного логгера
        await self._stop_minute_logger()
        self._start_minute_logger("ext" if use_ext else "normal", float(sl) if sl else None)

        # (3) Всегда форсируем flat перед новым входом
        ps, sz = self._position_side_and_size()
        if ps and sz > 0:
            log.info(f"[ENTER][FLAT] close existing {ps} size={self._fmt(sz)} before new entry")
            await self.close_market(self._opposite(ps), sz)
            for _ in range(20):
                p2, s2 = self._position_side_and_size()
                if not p2 or s2 <= 0:
                    break
                await asyncio.sleep(0.25)

        if use_ext:
            log.info(f"[EXT][FOLLOW] side={side} prevH={self._fmt(prev_high)} prevL={self._fmt(prev_low)} qty={self._fmt(qty)}")
            try:
                await self._follow_extremes(side, float(prev_high), float(prev_low), qty)
                log.info("[EXT][RETURN] _follow_extremes finished normally")
            except Exception as e:
                log.exception(f"[EXT][CRASH] _follow_extremes exception: {e}")
            finally:
                await self._stop_minute_logger()
            return

        # Маркет-вход: БЕЗ TP/SL, с жёстким ретраем 110007 (уменьшаем qty до минимума)
        log.info(f"[ENTER] {side} qty={self._fmt(qty)}")
        attempt_qty = qty
        order_id = None
        # максимум 6 попыток, уменьшая до минимального шага
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
                f = self.ensure_filters()
                step = f.get("qtyStep", 0.001)
                min_qty = f.get("minQty", 0.001)
                new_qty = self._round_down_qty(attempt_qty * 0.75)
                if new_qty >= attempt_qty:
                    new_qty = self._round_down_qty(attempt_qty - step)
                if new_qty < min_qty:
                    log.error(f"[ENTER][FAIL] insufficient balance for minimal qty; last={self._fmt(attempt_qty)}")
                    await self._stop_minute_logger()
                    return
                log.info(f"[ENTER][RETRY_QTY] 110007 -> {self._fmt(attempt_qty)} -> {self._fmt(new_qty)}")
                attempt_qty = new_qty
                continue
            if rc == 30208:
                # прайс-защита: используем мягкую толерантность
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
                # ещё попытка с TickSize=5
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
                await self._stop_minute_logger()
                return
            log.error(f"[ENTER][FAIL] {r}")
            await self._stop_minute_logger()
            return

        if not order_id:
            log.error("[ENTER][FAIL] no orderId")
            await self._stop_minute_logger()
            return

        # Дождаться фактического фила или ретраиться
        if not await self._await_fill_or_retry(order_id, side, attempt_qty):
            log.warning("[ENTER][ABORT] no fill")
            await self._stop_minute_logger()
            return

        # ← NEW: сразу ставим SL после обычного входа по рынку
        if sl:
            try:
                tr = self.client.trading_stop(
                    self.symbol,
                    side=side,
                    stop_loss=float(sl),
                    tpslMode="Full",
                    positionIdx=0,
                )
                if tr.get("retCode") in (0, None):
                    log.info(f"[SL][NORM] set SL={self._fmt(sl)} OK")
                else:
                    log.warning(f"[SL][NORM][FAIL] {tr}")
            except Exception as e:
                log.warning(f"[SL][NORM][EXC] {e}")

        # Обновим qty на фактический
        qty = attempt_qty

        # Нормализуем TP/SL относительно ФАКТИЧЕСКОЙ стороны и базовой цены позиции
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

        await self._stop_minute_logger()

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
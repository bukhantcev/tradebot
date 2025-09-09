import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("PM")


@dataclass
class PositionState:
    side: str = "FLAT"   # LONG / SHORT / FLAT
    size: float = 0.0
    entry_price: float = 0.0
    sl_price: Optional[float] = None    # Биржевой SL (установленный через executor)
    tp_price: Optional[float] = None    # ВИРТУАЛЬНЫЙ TP (для трейлинга/каскада)


class PositionManager:
    """
    Управляет локальным состоянием позиции и постановкой реального SL на бирже.
    - Входы: маркет-ордера через Executor.market_buy/sell
    - Сразу после входа: считаем R = max_loss_usdt / qty и ставим БИРЖЕВОЙ SL (через ex.set_stop_loss),
      TP ведём виртуально (в state.tp_price), без отправки на биржу.
    - Каскад/трейлинг: при достижении виртуального TP двигаем биржевой SL на уровень достигнутого TP
      и пересчитываем следующий виртуальный TP (+R для LONG, −R для SHORT).
    - Выход по SL исполняет биржа; мы лишь помечаем FLAT, чтобы не задвоить вход.
    - close_all(): сперва убирает SL/TP на бирже (ex.cancel_trading_stop()), затем маркет-выход.
    """

    def __init__(self, symbol: str, base_order_usdt: float, max_loss_usdt: float, ex, category: str):
        self.symbol = symbol
        self.base_usdt = base_order_usdt
        self.max_loss = max_loss_usdt
        self.ex = ex              # Executor (обёртка над BybitClient)
        self.category = category
        self.state = PositionState()
        self.loss_streak = 0

    # ===== Helpers =====
    def is_open(self) -> bool:
        return self.state.side in ("LONG", "SHORT")

    async def _round_to_tick(self, price: float) -> float:
        filters = await self.ex.get_filters()
        tick = float(filters.get("tickSize") or 0.01)
        tick = tick if tick > 0 else 0.01
        # округляем к ближайшей «сетке» тика
        return round(price / tick) * tick

    async def _qty_from_usdt(self, price: float) -> float:
        filters = await self.ex.get_filters()
        price = max(float(price), 1e-9)
        qty_raw = self.base_usdt / price

        step = float(filters.get("qtyStep") or 0.001)
        minq = float(filters.get("minOrderQty") or step)

        # приведение к шагу
        qty = (int(qty_raw / step)) * step
        # минимум
        if qty < minq:
            qty = minq

        log.info("[PM] qty calc: usdt=%.2f price=%.2f -> raw=%.10f -> qty=%.6f (min=%.6f step=%.6f)",
                 self.base_usdt, price, qty_raw, qty, minq, step)
        return float(qty)

    def _risk_R(self) -> float:
        qty = max(self.state.size, 1e-9)
        return self.max_loss / qty

    def _mark_assumed_closed(self, last_price: float, reason: str):
        side = self.state.side
        qty = self.state.size

        pnl = 0.0
        if last_price and qty:
            if side == "LONG":
                pnl = (last_price - self.state.entry_price) * qty
            else:
                pnl = (self.state.entry_price - last_price) * qty

        self.loss_streak = (self.loss_streak + 1) if pnl < 0 else 0
        log.info("[PM] Assume %s closed by exchange (%s) @%.2f | PnL≈%.2f | loss_streak=%d",
                 side, reason, last_price or -1.0, pnl, self.loss_streak)
        self.state = PositionState()

    # ===== Entries =====
    async def open_long(self, last_price: float):
        if self.is_open():
            log.info("[PM] skip open_long — already open: %s", self.state)
            return

        qty = await self._qty_from_usdt(last_price)
        if not qty:
            log.error("[PM] open_long aborted: qty=0")
            return

        if await self.ex.market_buy(qty):
            # зафиксировали локально
            self.state.side = "LONG"
            self.state.size = qty
            self.state.entry_price = last_price

            R = self.max_loss / qty
            sl = await self._round_to_tick(last_price - R)
            tp_virtual = await self._round_to_tick(last_price + 2 * R)

            ok_sl = await self.ex.set_stop_loss(stop_loss=sl)

            self.state.sl_price = sl                 # реальный SL (биржа)
            self.state.tp_price = tp_virtual         # виртуальный TP

            log.info("[PM] LONG opened @ %.2f qty=%.6f | R=%.2f | SL(exch)=%.2f | TP(virtual)=%.2f | set_sl_ok=%s",
                     last_price, qty, R, sl, tp_virtual, ok_sl)
        else:
            log.error("[PM] open_long failed: market_buy rejected")

    async def open_short(self, last_price: float):
        if self.is_open():
            log.info("[PM] skip open_short — already open: %s", self.state)
            return

        qty = await self._qty_from_usdt(last_price)
        if not qty:
            log.error("[PM] open_short aborted: qty=0")
            return

        if await self.ex.market_sell(qty):
            # зафиксировали локально
            self.state.side = "SHORT"
            self.state.size = qty
            self.state.entry_price = last_price

            R = self.max_loss / qty
            sl = await self._round_to_tick(last_price + R)
            tp_virtual = await self._round_to_tick(last_price - 2 * R)

            ok_sl = await self.ex.set_stop_loss(stop_loss=sl)

            self.state.sl_price = sl                 # реальный SL (биржа)
            self.state.tp_price = tp_virtual         # виртуальный TP

            log.info("[PM] SHORT opened @ %.2f qty=%.6f | R=%.2f | SL(exch)=%.2f | TP(virtual)=%.2f | set_sl_ok=%s",
                     last_price, qty, R, sl, tp_virtual, ok_sl)
        else:
            log.error("[PM] open_short failed: market_sell rejected")

    # ===== Trailing / Virtual TP logic =====
    async def check_sl_tp(self, last_price: float) -> bool:
        """
        Возвращает True если мы считаем позицию закрытой (по SL биржи).
        Логика:
          1) «Достигнут виртуальный TP»:
             - двигаем БИРЖЕВОЙ SL на уровень достигнутого виртуального TP
             - пересчитываем следующий виртуальный TP (ещё +R для LONG, −R для SHORT)
          2) «Цена пересекла текущий SL» → предполагаем, что биржа закрыла позицию:
             - помечаем FLAT локально (assume closed), чтобы не удвоить входы
        """
        if not self.is_open():
            return False

        side = self.state.side
        sl = self.state.sl_price
        tp = self.state.tp_price
        R = self._risk_R()

        # 1) Виртуальный TP сработал → трейлим SL на TP и считаем следующий виртуальный TP.
        if side == "LONG" and tp is not None and last_price >= tp:
            new_sl = await self._round_to_tick(max(sl if sl is not None else tp, tp))
            ok = await self.ex.set_stop_loss(stop_loss=new_sl)
            next_tp = await self._round_to_tick(tp + R)
            self.state.sl_price = new_sl
            self.state.tp_price = next_tp
            log.info("[PM] TRAIL LONG: hit virtual TP=%.2f → move SL to %.2f (ok=%s); next virtual TP=%.2f",
                     tp, new_sl, ok, next_tp)
            return False

        if side == "SHORT" and tp is not None and last_price <= tp:
            new_sl = await self._round_to_tick(min(sl if sl is not None else tp, tp))
            ok = await self.ex.set_stop_loss(stop_loss=new_sl)
            next_tp = await self._round_to_tick(tp - R)
            self.state.sl_price = new_sl
            self.state.tp_price = next_tp
            log.info("[PM] TRAIL SHORT: hit virtual TP=%.2f → move SL to %.2f (ok=%s); next virtual TP=%.2f",
                     tp, new_sl, ok, next_tp)
            return False

        # 2) Пересечение SL (закрытие сделает биржа) → помечаем, что мы FLAT,
        #    чтобы не пытаться открыть новую позицию поверх закрывающейся.
        if side == "LONG" and sl is not None and last_price <= sl:
            self._mark_assumed_closed(last_price, "exchange SL LONG")
            return True

        if side == "SHORT" and sl is not None and last_price >= sl:
            self._mark_assumed_closed(last_price, "exchange SL SHORT")
            return True

        return False

    # ===== Manual close =====
    async def close_all(self, last_price: float | None = None):
        if not self.is_open():
            log.info("[PM] close_all: already FLAT")
            return

        # 1) Сброс биржевых SL/TP (на всякий)
        await self.ex.cancel_trading_stop()

        # 2) Рыночный выход
        qty = self.state.size
        side = self.state.side
        if side == "LONG":
            ok = await self.ex.market_sell(qty)
        else:
            ok = await self.ex.market_buy(qty)

        # 3) Учёт приблизительного PnL (для streak)
        pnl = 0.0
        if last_price and qty:
            if side == "LONG":
                pnl = (last_price - self.state.entry_price) * qty
            else:
                pnl = (self.state.entry_price - last_price) * qty
        self.loss_streak = (self.loss_streak + 1) if pnl < 0 else 0

        log.info("[PM] Close_all %s qty=%.6f @%.2f ok=%s | PnL≈%.2f | loss_streak=%d",
                 side, qty, last_price or -1.0, ok, pnl, self.loss_streak)

        # 4) reset
        self.state = PositionState()
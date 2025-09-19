# trader_mod/tpsl.py
import time
import asyncio
import logging
from .utils import round_step, ceil_step
log = logging.getLogger("TRADER")

class TPSL:
    def __init__(self, account, get_last_price, get_pos_side_size, close_market, cancel_realigner, notifier=None):
        self.account = account
        self.get_last_price = get_last_price
        self.get_pos_side_size = get_pos_side_size
        self.close_market = close_market
        self.cancel_realigner = cancel_realigner
        self.notifier = notifier
        self._task = None
        self._sl_streak = 0
        self._cooldown_until = 0.0
        self._cooldown_minutes = 10.0
        self.paused = False

    # wire-ups for external state (Trader reads/writes эти поля напрямую)
    def reset_sl_streak(self):
        self._sl_streak = 0

    def set_cooldown_minutes(self, v: float):
        self._cooldown_minutes = float(v)

    def is_on_cooldown(self) -> bool:
        return self._cooldown_until > time.time()

    def cooldown_left(self) -> int:
        return int(max(0.0, self._cooldown_until - time.time()))

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False
        self._sl_streak = 0
        self._cooldown_until = 0.0

    # --- math ---
    def fix_tpsl(self, side: str, price: float, sl: float, tp: float, tick: float) -> tuple[float, float]:
        p = float(price)
        sl_f, tp_f = float(sl), float(tp)
        t = max(float(tick), 0.0) or 0.1
        if side == "Buy":
            if sl_f >= p: sl_f = p - t
            if tp_f <= p: tp_f = p + t
            sl_f = round_step(sl_f, t)
            tp_f = ceil_step(tp_f, t)
            if sl_f >= p: sl_f = p - 2*t
            if tp_f <= p: tp_f = p + 2*t
        else:
            if sl_f <= p: sl_f = p + t
            if tp_f >= p: tp_f = p - t
            sl_f = ceil_step(sl_f, t)
            tp_f = round_step(tp_f, t)
            if sl_f <= p: sl_f = p + 2*t
            if tp_f >= p: tp_f = p - 2*t
        return sl_f, tp_f

    def normalize_with_anchor(self, side: str, base_price: float, sl: float, tp: float, tick: float) -> tuple[float, float]:
        last = float(self.get_last_price() or 0.0)
        anchor = float(base_price or 0.0)
        if side == "Buy":
            if last > 0: anchor = max(anchor, last)
            if tp <= anchor: tp = ceil_step(anchor + tick, tick)
            if sl >= anchor: sl = round_step(anchor - tick, tick)
        else:
            if last > 0: anchor = min(anchor, last) if anchor > 0 else last
            if tp >= anchor: tp = round_step(anchor - tick, tick)
            if sl <= anchor: sl = ceil_step(anchor + tick, tick)
        return self.fix_tpsl(side, anchor if anchor > 0 else (base_price or last or tp), sl, tp, tick)

    async def realign_tpsl(self, side: str, desired_sl: float, desired_tp: float, tick: float, debounce: float = 0.8, max_tries: int = 30, client=None, symbol:str=""):
        tries = 0
        while tries < max_tries:
            tries += 1
            ps, sz = self.get_pos_side_size()
            if not ps or sz <= 0:
                break
            sl_norm, tp_norm = self.normalize_with_anchor(side, base_price=0.0, sl=desired_sl, tp=desired_tp, tick=tick)
            try:
                r = client.trading_stop(
                    symbol,
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
                tag = "OK" if rc in (0, None) else ("UNCHANGED" if rc == 34040 else f"RC{rc}")
                log.info(f"[REALIGN][{tag}] sl={sl_norm:.2f} tp={tp_norm:.2f} (try {tries})")
                if rc in (0, None) and abs(tp_norm - desired_tp) <= 2*max(tick, 1e-9) and abs(sl_norm - desired_sl) <= 2*max(tick, 1e-9):
                    break
            except Exception:
                pass
            await asyncio.sleep(max(0.1, float(debounce)))

    async def watchdog_close_on_last(self, side: str, sl_price: float, tp_price: float, close_side: str, check_interval: float, max_wait: float):
        deadline = time.monotonic() + max_wait
        while time.monotonic() < deadline:
            try:
                ps, sz = self.get_pos_side_size()
                if not ps or sz <= 0:
                    return True
                last = float(self.get_last_price() or 0.0)
                if last <= 0:
                    await asyncio.sleep(check_interval); continue

                crossed = None
                if side == "Buy":
                    if sl_price and last <= sl_price: crossed = "SL"
                    elif tp_price and last >= tp_price: crossed = "TP"
                else:
                    if sl_price and last >= sl_price: crossed = "SL"
                    elif tp_price and last <= tp_price: crossed = "TP"

                if crossed:
                    log.info(f"[WATCH][CROSS] Last={last:.2f} vs SL={sl_price:.2f} / TP={tp_price:.2f} -> {crossed} force close")
                    self.cancel_realigner()
                    try:
                        await self.close_market(close_side, sz)
                    except Exception:
                        pass
                    ok_flat = True
                    # streak / cooldown logic (упрощённый): управляется снаружи Trader-ом,
                    # но оставим здесь хук при необходимости расширить
                    return ok_flat
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            await asyncio.sleep(check_interval)
        return False
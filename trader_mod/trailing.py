import asyncio
import logging
import time
from typing import Optional

log = logging.getLogger("TRADER")


async def start_trailing(
    self,
    side: str,
    atr: float,
    entry: float,
    gap_mode: str = "atr",
    atr_mult: float = 1.5,
    check_sec: float = 1.0,
    min_step_ticks: float = 1.0,
) -> None:
    """
    Trailing stop loop for trend mode.

    Rules
    -----
    • gap = ATR * atr_mult (default) or ticks-based if gap_mode == 'ticks'.
    • For Buy:  new_sl = max(cur_sl, last - gap)
      For Sell: new_sl = min(cur_sl, last + gap)
    • Applies exchange stop only (no TP) via trading_stop.
    • Stops on: position flat, regime != 'trend', cancellation.
    """
    try:
        side = "Buy" if side == "Buy" else "Sell"
        f = self.ensure_filters()
        tick = float(f.get("tickSize", 0.1) or 0.1)
        min_step = max(tick * float(min_step_ticks), tick)

        # compute gap
        if gap_mode == "ticks":
            try:
                ticks = float(atr)  # user passes ticks in `atr` var in this mode
            except Exception:
                ticks = 10.0
            gap = max(tick * ticks, min_step)
        else:
            # default ATR-based gap
            try:
                gap = max(float(atr) * float(atr_mult), min_step)
            except Exception:
                gap = max(50 * tick, min_step)

        # initialize current SL around entry, respecting favorable direction only later
        try:
            cur_sl: Optional[float]
            if side == "Buy":
                cur_sl = entry - gap
            else:
                cur_sl = entry + gap
        except Exception:
            cur_sl = None

        log.info(
            f"[TRAIL][START] side={side} entry={self._fmt(entry)} gap={self._fmt(gap)} (mode={gap_mode} mult={atr_mult})"
        )
        if getattr(self, "notifier", None):
            try:
                asyncio.create_task(self.notifier.notify(f"🏃‍♂️ Трейлинг стартовал: {side} • entry {self._fmt(entry)} • gap {self._fmt(gap)} (mode={gap_mode}, x{atr_mult})"))
            except Exception:
                pass

        # Loop
        while True:
            # cancellation or regime switch
            if getattr(self, "_regime", None) != "trend":
                log.info("[TRAIL][STOP] regime switched out of trend")
                if getattr(self, "notifier", None):
                    try:
                        asyncio.create_task(self.notifier.notify("⏹ Трейлинг остановлен: смена режима"))
                    except Exception:
                        pass
                break

            # if flat — stop
            ps, sz = self._position_side_and_size()
            if not ps or sz <= 0:
                log.info("[TRAIL][STOP] flat position")
                if getattr(self, "notifier", None):
                    try:
                        asyncio.create_task(self.notifier.notify("⏹ Трейлинг остановлен: позиция закрыта"))
                    except Exception:
                        pass
                break

            # compute new target SL
            last = self._last_price()
            if last > 0:
                if side == "Buy":
                    target = last - gap
                    if cur_sl is None:
                        cur_sl = target
                    # only tighten upwards
                    new_sl = max(cur_sl, target)
                else:
                    target = last + gap
                    if cur_sl is None:
                        cur_sl = target
                    # only tighten downwards
                    new_sl = min(cur_sl, target)

                # apply only if moved at least a tick
                if cur_sl is None or abs(new_sl - cur_sl) >= min_step:
                    try:
                        r = self.client.trading_stop(
                            self.symbol,
                            side=side,
                            stop_loss=float(new_sl),
                            tpslMode="Full",
                            slTriggerBy="MarkPrice",
                            positionIdx=0,
                        )
                        rc = r.get("retCode")
                        tag = "OK" if rc in (0, None, 34040) else f"rc={rc} msg={r.get('retMsg')}"
                        log.info(f"[TRAIL][SET] SL={self._fmt(new_sl)} {tag}")
                        if rc in (0, None, 34040):
                            cur_sl = new_sl
                    except Exception as e:
                        log.warning(f"[TRAIL][EXC] {e}")

            await asyncio.sleep(max(0.2, float(check_sec)))

    except asyncio.CancelledError:
        log.info("[TRAIL][CANCEL]")
        if getattr(self, "notifier", None):
            try:
                asyncio.create_task(self.notifier.notify("⏹ Трейлинг отменён"))
            except Exception:
                pass
        raise
    except Exception as e:
        log.warning(f"[TRAIL][STOP][EXC] {e}")
    finally:
        # Tell facade that task finished
        try:
            self._trail_task = None
        except Exception:
            pass
from bybit_client import BybitClient
from trader_mod import trailing as trail

class Trader:
    # ... existing methods ...

    def _start_trailing(self, side: str, atr: float, entry: float, gap_mode: str = "atr", atr_mult: float = 1.5, check_sec: float = 1.0):
        """Start trailing stop task (trend mode)."""
        # stop previous if any
        t = getattr(self, "_trail_task", None)
        if t and not t.done():
            try:
                t.cancel()
            except Exception:
                pass
        try:
            self._trail_task = asyncio.create_task(
                trail.start_trailing(self, side, atr, entry, gap_mode=gap_mode, atr_mult=atr_mult, check_sec=check_sec),
                name="trail_stop_loop",
            )
        except Exception:
            self._trail_task = None

    async def _stop_trailing(self):
        """Stop trailing task if running."""
        t = getattr(self, "_trail_task", None)
        if t and not t.done():
            try:
                t.cancel()
                try:
                    await asyncio.wait_for(t, timeout=1.0)
                except Exception:
                    pass
            finally:
                self._trail_task = None
        else:
            self._trail_task = None

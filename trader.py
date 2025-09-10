import asyncio
import json
import random
import ssl
import time
from typing import Optional, Dict, Any

import websockets
from websockets.exceptions import ConnectionClosed

try:
    import certifi
except Exception:
    certifi = None

from config import CFG, Config
from bybit_client import BybitClient
from indicators import OnlineIndicators
from log import log
from regime import classify_regime, MarketRegime

class Trader:
    def __init__(self, cfg: Config, bot, bybit: BybitClient):
        self.cfg = cfg
        self.bot = bot
        self.bybit = bybit

        self.ind = OnlineIndicators(cfg.ema_fast, cfg.ema_slow, cfg.atr_len, cfg.channel_lookback)
        self.running = False
        self.ws_task: Optional[asyncio.Task] = None
        self.trade_task: Optional[asyncio.Task] = None

        self.last_regime: Optional[str] = None
        self.regime_streak = 0
        self.required_streak = 3
        self.loss_streak = 0
        self.cooldown_until = 0.0

        self.position_side: Optional[str] = None
        self.position_entry: Optional[float] = None
        self.position_qty: float = 0.0
        self.position_sl: Optional[float] = None
        self.virt_tp_price: Optional[float] = None

    async def notify(self, text: str):
        try:
            log(text)
            if self.cfg.tg_chat_id:
                await self.bot.send_message(self.cfg.tg_chat_id, text)
        except Exception as e:
            log(f"[TG ERROR] {e}: {text}")

    async def load_filters_and_set_leverage(self):
        info = await self.bybit.get_instruments_info()
        it = info.get("result", {}).get("list", [])
        if it:
            f = it[0].get("lotSizeFilter", {})
            pf = it[0].get("priceFilter", {})
            self.bybit.qty_step = float(f.get("qtyStep", self.bybit.qty_step))
            self.bybit.min_qty = float(f.get("minOrderQty", self.bybit.min_qty))
            self.bybit.tick_size = float(pf.get("tickSize", self.bybit.tick_size))

        # Однократная установка плеча с корректной диагностикой
        try:
            res = await self.bybit.set_leverage(self.cfg.leverage, self.cfg.leverage)
            if isinstance(res, dict):
                rc = res.get("retCode", 0)
                if rc == 0:
                    await self.notify("✅ Leverage set successfully.")
                elif rc == 110043:
                    # leverage not modified — уже стоит нужное плечо; это не ошибка
                    log("[INFO] leverage not modified — already set to desired value")
                else:
                    await self.notify(f"❌ set_leverage error: {res}")
        except Exception as e:
            await self.notify(f"❌ set_leverage exception: {e}")

    async def _seed_history(self, bars: int = 200):
        """Подтянуть историю 1m свечей и прогреть индикаторы, чтобы режим был доступен сразу."""
        try:
            resp = await self.bybit.get_kline(interval="1", limit=bars)
            li = resp.get("result", {}).get("list", []) if isinstance(resp, dict) else []
            if not li:
                await self.notify("ℹ️ История свечей не получена (пусто). Продолжаю без прогрева.")
                return
            # По Bybit v5 kline list обычно отсортирован от новой к старой. Прогоним в хронологическом порядке.
            for itm in reversed(li):
                # Формат: [start, open, high, low, close, volume, turnover]
                try:
                    high = float(itm[2]); low = float(itm[3]); close = float(itm[4]); vol = float(itm[5])
                    self.ind.push_candle(high, low, close, vol)
                except Exception:
                    continue
            # После прогрева попробуем классифицировать режим
            if len(self.ind.closes) >= max(self.cfg.ema_slow, self.cfg.atr_len) + 5:
                regime, _ = classify_regime(self.ind, self.cfg)
                self.last_regime = regime
            else:
                need = max(self.cfg.ema_slow, self.cfg.atr_len) + 5
                self.last_regime = f"WARMUP {len(self.ind.closes)}/{need}"
            await self.notify(f"🧪 Прогрев индикаторов: получено {len(li)} свечей, буфер={len(self.ind.closes)} | режим={self.last_regime}")
        except Exception as e:
            await self.notify(f"⚠️ Ошибка прогрева истории: {e}")

    async def start(self):
        log("[ENGINE] start trading requested")
        if self.running:
            await self.notify("⚠️ Торговля уже запущена.")
            return
        await self.load_filters_and_set_leverage()
        # Прогрев индикаторов историей, чтобы не ждать 20–30 минут
        await self._seed_history(200)
        self.running = True
        self.ws_task = asyncio.create_task(self._ws_loop())
        self.trade_task = asyncio.create_task(self._trade_loop())
        await self.notify("▶️ Торговля запущена.")

    async def stop(self):
        log("[ENGINE] stop trading requested")
        self.running = False
        if self.ws_task:
            self.ws_task.cancel()
        if self.trade_task:
            self.trade_task.cancel()
        try:
            await self.bybit.cancel_all_orders()
        except Exception:
            pass
        try:
            res = await self.bybit.close_position_market()
            await self.notify(f"🛑 Торговля остановлена. Позиции закрыты: {res}")
        except Exception as e:
            await self.notify(f"🛑 Торговля остановлена. Ошибка при закрытии позиций: {e}")

    async def _ws_loop(self):
        sub_msg = {"op": "subscribe", "args": [f"kline.1.{self.cfg.symbol}"]}
        url = self.cfg.ws_public_url
        while self.running:
            try:
                if self.cfg.ws_verify_ssl:
                    if self.cfg.ws_use_certifi and certifi is not None:
                        ctx = ssl.create_default_context(cafile=certifi.where())
                    else:
                        ctx = ssl.create_default_context()
                else:
                    ctx = ssl._create_unverified_context()

                async with websockets.connect(
                    url,
                    ssl=ctx,
                    ping_interval=10,
                    ping_timeout=self.cfg.ws_timeout,
                ) as ws:
                    await ws.send(json.dumps(sub_msg))
                    log(f"[WS] subscribed to kline.1 {self.cfg.symbol}")
                    await self.notify(f"📡 Подключен к WS {url}, подписка kline 1m {self.cfg.symbol}")
                    while self.running:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=self.cfg.ws_timeout + 5)
                        except asyncio.TimeoutError:
                            continue
                        data: Dict[str, Any] = json.loads(raw)
                        if data.get("topic", "").startswith("kline.1."):
                            arr = data.get("data", [])
                            for k in arr:
                                if not k.get("confirm", False):
                                    continue
                                high = float(k["high"]); low = float(k["low"]); close = float(k["close"]); vol = float(k.get("volume", "0"))
                                self.ind.push_candle(high, low, close, vol)
                                if self.cfg.log_ws_raw:
                                    log(f"[WS<-] kline close={close} hi={high} lo={low} vol={vol}")
                        elif data.get("op") == "ping":
                            await ws.send(json.dumps({"op": "pong"}))
                        elif "success" in data and data.get("success") is True:
                            continue
            except ConnectionClosed:
                await asyncio.sleep(1.0)
            except Exception as e:
                await self.notify(f"WS ошибка: {e}")
                await asyncio.sleep(2.0)

    async def _trade_loop(self):
        import math
        while self.running:
            try:
                await asyncio.sleep(2.0)
                if time.time() < self.cooldown_until:
                    await self._manage_open_position()
                    continue
                if len(self.ind.closes) < max(self.cfg.ema_slow, self.cfg.atr_len) + 5:
                    continue

                regime, metrics = classify_regime(self.ind, self.cfg)
                if self.cfg.log_signals:
                    log(f"[REGIME] {regime} atrz={metrics['atr_z']:.2f} ema_spread={metrics['ema_spread']:.4f} vwap={metrics['vwap']:.2f}")

                if random.random() < 0.05:
                    await self.notify(f"ℹ️ Режим: {regime} | ATRz={metrics['atr_z']:.2f} | EMAspread={metrics['ema_spread']:.4f}")

                await self._manage_open_position()
                if self.position_side is not None:
                    continue

                if self.regime_streak and self.last_regime == regime:
                    self.regime_streak += 1
                else:
                    self.regime_streak = 1
                    self.last_regime = regime
                if self.regime_streak < 3:
                    continue

                close = self.ind.closes[-1]
                vwap = self.ind.vwap or close

                if regime == MarketRegime.TRND_UP:
                    if abs(close - max(self.ind.ema_slow, vwap)) / close <= 0.002:
                        await self._enter_position("Buy", close, regime, metrics)
                elif regime == MarketRegime.TRND_DN:
                    if abs(close - min(self.ind.ema_slow, vwap)) / close <= 0.002:
                        await self._enter_position("Sell", close, regime, metrics)
                elif regime == MarketRegime.RNG:
                    low, high = self.ind.channel()
                    if low is not None and high is not None and high > low:
                        if abs(close - high) / close <= 0.0015:
                            await self._enter_position("Sell", close, regime, metrics)
                        elif abs(close - low) / close <= 0.0015:
                            await self._enter_position("Buy", close, regime, metrics)
                elif regime == MarketRegime.IMP_UP:
                    await self._enter_position("Buy", close, regime, metrics)
                elif regime == MarketRegime.IMP_DN:
                    await self._enter_position("Sell", close, regime, metrics)

            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.notify(f"⚠️ Trade loop error: {e}")

    async def _enter_position(self, side: str, px: float, regime: str, metrics):
        atr = self.ind.atr or (px * 0.003)
        if side == "Buy":
            sl = px - max(self.cfg.atr_mult * atr, px * 0.0015)
        else:
            sl = px + max(self.cfg.atr_mult * atr, px * 0.0015)

        bal = await self.bybit.get_wallet_balance()
        total_equity = 0.0
        for a in bal.get("result", {}).get("list", []):
            for c in a.get("coin", []):
                if c.get("coin") == "USDT":
                    total_equity += float(c.get("equity", "0"))

        if total_equity <= 0:
            await self.notify("❌ Нет equity для расчёта позиции.")
            return

        risk_usd = total_equity * self.cfg.risk_pct
        stop_dist = abs(px - sl)
        if stop_dist <= 0:
            return
        qty_raw = risk_usd / stop_dist   # qty in base coin (linear)
        qty = max(0.0, qty_raw)
        if qty <= 0:
            await self.notify("❌ Размер позиции <= 0.")
            return

        order_res = await self.bybit.create_order(side, qty, order_type="Market", reduce_only=False)
        if isinstance(order_res, dict) and order_res.get("retCode", 0) != 0:
            await self.notify(f"❌ Ошибка создания ордера: {order_res}")
            return

        self.position_side = "Long" if side == "Buy" else "Short"
        self.position_qty = qty
        self.position_entry = px

        # Округление SL по шагу цены биржи
        def round_to_tick(val: float, tick: float) -> float:
            if tick <= 0:
                return val
            return round(val / tick) * tick
        sl = round_to_tick(sl, self.bybit.tick_size)
        self.position_sl = sl

        # Получить фактическую позицию (avgPrice, positionIdx) и корректно задать SL/Trailing
        avg_price = px
        position_idx = 0
        try:
            plist = await self.bybit.get_position_list()
            li = plist.get("result", {}).get("list", []) if isinstance(plist, dict) else []
            for p in li:
                if p.get("symbol") == self.cfg.symbol and p.get("side") == ("Buy" if self.position_side == "Long" else "Sell"):
                    avg_price = float(p.get("avgPrice") or avg_price)
                    position_idx = int(p.get("positionIdx") or 0)
                    break
        except Exception:
            pass

        # === Реальный TP по avg_price и tp_pct, округление к шагу цены ===
        tp_price = None
        try:
            tp_pct = getattr(self.cfg, "tp_pct", None)
            if tp_pct is not None and tp_pct > 0:
                if side == "Buy":
                    tp_raw = avg_price * (1.0 + tp_pct)
                else:  # side == "Sell"
                    tp_raw = avg_price * (1.0 - tp_pct)
                tp_price = round_to_tick(tp_raw, self.bybit.tick_size)
            # Если tp_pct не задан или <= 0, НЕ выставляем биржевой TP, только виртуальный
        except Exception:
            pass

        if self.cfg.use_trailing:
            # Преобразуем trailing параметры в абсолютные цены (Bybit ждет абсолюты, не проценты)
            # Если distance/activation заданы малыми числами (<1), трактуем как процент от цены
            ta = self.cfg.trailing_activation
            td = self.cfg.trailing_distance
            if ta is None or ta <= 0:
                ta_abs = avg_price  # активировать возле средней цены входа
            else:
                ta_abs = avg_price * ta if ta < 1 else ta
            if td is None or td <= 0:
                # по умолчанию половина ATR, но не менее одного тика
                base = max(self.ind.atr or (avg_price * 0.003), self.bybit.tick_size)
                td_abs = base * 0.5
            else:
                td_abs = avg_price * td if td < 1 else td
            ta_abs = round_to_tick(ta_abs, self.bybit.tick_size)
            td_abs = max(round_to_tick(td_abs, self.bybit.tick_size), self.bybit.tick_size)

            await self.bybit.set_trading_stop(
                sl=sl,
                take_profit=str(tp_price) if (tp_price is not None and getattr(self.cfg, "tp_pct", 0) > 0) else None,
                side=side,
                position_idx=position_idx,
                trailing_activation=str(ta_abs),
                trailing_distance=str(td_abs),
            )
        else:
            await self.bybit.set_trading_stop(
                sl=sl,
                take_profit=str(tp_price) if (tp_price is not None and getattr(self.cfg, "tp_pct", 0) > 0) else None,
                side=side,
                position_idx=position_idx,
            )

        # Биржевой TP установлен через trading-stop только если tp_pct > 0, иначе виртуальный TP
        if getattr(self.cfg, "tp_pct", 0) > 0 and tp_price is not None:
            self.virt_tp_price = None
            tp_str = f"TP(биржевой)={tp_price:.2f}"
        else:
            # fallback to R-multiplier virtual TP
            r = abs(px - sl)
            if side == "Buy":
                virt_tp = px + self.cfg.tp_r_mult * r
            else:
                virt_tp = px - self.cfg.tp_r_mult * r
            self.virt_tp_price = round_to_tick(virt_tp, self.bybit.tick_size)
            tp_str = f"TP(вирт)={self.virt_tp_price:.2f}"

        await self.notify(
            f"✅ ВХОД {self.position_side} qty={self.position_qty} @ {px:.2f} | regime={regime}\n"
            f"SL(биржевой)={sl:.2f} | {tp_str}\n"
            f"metrics: ATRz={metrics.get('atr_z',0):.2f} EMAspread={metrics.get('ema_spread',0):.4f}"
        )

    async def _manage_open_position(self):
        if self.position_side is None or not self.ind.closes:
            return
        last = self.ind.closes[-1]
        hit_tp = (self.position_side == "Long" and self.virt_tp_price and last >= self.virt_tp_price) or \
                 (self.position_side == "Short" and self.virt_tp_price and last <= self.virt_tp_price)

        if hit_tp:
            log(f"[EXIT] virtual TP hit @ {last:.2f}")
            side = "Sell" if self.position_side == "Long" else "Buy"
            try:
                await self.bybit.create_order(side, self.position_qty, order_type="Market", reduce_only=True)
                await self.bybit.cancel_all_orders()
            except Exception as e:
                await self.notify(f"⚠️ Ошибка при закрытии по TP: {e}")
            pnl = (last - (self.position_entry or last)) * (1 if self.position_side == "Long" else -1)
            await self.notify(f"🎯 TP закрыт @ {last:.2f} | pnl/px={pnl:.2f}")
            self._clear_position()
            self.loss_streak = 0
            return

        if random.random() < 0.05:
            try:
                plist = await self.bybit.get_position_list()
                li = plist.get("result", {}).get("list", [])
                size_now = 0.0
                for p in li:
                    size_now += abs(float(p.get("size", "0")))
                if size_now == 0.0:
                    exit_px = last
                    pnl = (exit_px - (self.position_entry or exit_px)) * (1 if self.position_side == "Long" else -1)
                    log(f"[EXIT] position gone (likely SL). last={last:.2f}")
                    if pnl < 0:
                        self.loss_streak += 1
                    else:
                        self.loss_streak = 0
                    await self.notify(f"⛔ Позиция закрыта (вероятно SL) @ {exit_px:.2f} | pnl/px={pnl:.2f}")
                    self._clear_position()
                    if self.loss_streak >= 2:
                        self.cooldown_until = time.time() + 60 * self.cfg.cooldown_after_2_losses_min
                        await self.notify(f"🧊 Кулдаун {self.cfg.cooldown_after_2_losses_min} мин после {self.loss_streak} стопов.")

            except Exception:
                pass

    def _clear_position(self):
        self.position_side = None
        self.position_entry = None
        self.position_qty = 0.0
        self.position_sl = None
        self.virt_tp_price = None
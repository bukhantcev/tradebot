import asyncio
import logging
from aiogram import Router, F, types, Bot
from aiogram.filters import CommandStart, Command
from bot.keyboards import main_kb
from core.bybit.client import BybitClient
from core.bybit.stream import BybitPublicStream
from core.ai.openai_client import ask_strategy
from core.params.calc import build_params
from core.trade.executor import Executor
from core.trade.position_manager import PositionManager
from core.strategies.momentum import Momentum
from core.strategies.reversal import Reversal
from core.strategies.breakout import Breakout
from core.strategies.density import Density
from core.strategies.knife import Knife

log = logging.getLogger("LOOP")

# --- WATCHDOG STATE (strict) ---
_last_entry_ts = 0.0
_last_signal_ts = 0.0
_last_exit_ts = 0.0

def _wd_mark_signal(now_ts: float):
    global _last_signal_ts
    _last_signal_ts = now_ts

def _wd_mark_entry(now_ts: float):
    global _last_entry_ts
    _last_entry_ts = now_ts

def _wd_mark_exit(now_ts: float):
    global _last_exit_ts
    _last_exit_ts = now_ts

def _wd_now():
    import time
    return time.time()

def _wd_idle_seconds() -> int:
    import time
    last = max(_last_entry_ts, _last_signal_ts, _last_exit_ts)
    if last <= 0:
        return 10**9
    return int(time.time() - last)

# Глобальная ссылка на последнее состояние для сервисных задач завершения
LAST_STATE = None

def setup_routes(dp, cfg):
    router = Router()
    state = {
        "cfg": cfg,
        "bybit": BybitClient(cfg["bybit_key"], cfg["bybit_secret"], testnet=cfg["testnet"]),
        "executor": None,
        "pm": None,
        "ws": None,
        "strategy": None,
        "strategy_name": None,
        "params": build_params(),
        "last_trade_ts": None,
        "idle_task": None,
        "loss_streak": 0,
        "stats": {"recent_high": 0, "recent_low": 0, "atr": 1.0},
        "ws_start_ts": None,
        "last_activity_ts": None,
    }

    # Сохраняем ссылку на state для shutdown_cleanup
    global LAST_STATE
    LAST_STATE = state

    async def _rotate_strategy(reason: str, bot: Bot, chat_id: int):
        cfg = state["cfg"]
        import time
        client = state["bybit"]
        log.info("[ROTATE] reason=%s -> collecting snapshot…", reason)
        state["last_activity_ts"] = time.time()

        j1 = await client.klines(cfg["category"], cfg["symbol"], "1", cfg["snap_1m"])
        j5 = await client.klines(cfg["category"], cfg["symbol"], "5", cfg["snap_5m"])
        ob = await client.orderbook(cfg["category"], cfg["symbol"], 50)
        bal = await client.wallet_balance(cfg["account_type"])

        snapshot = {
            "kline_1m": j1,
            "kline_5m": j5,
            "orderbook": ob,
            "wallet": bal,
        }
        ai = await ask_strategy(cfg["openai_model"], cfg["openai_key"], snapshot)
        name = ai["strategy"]
        reason = ai["reason"]
        log.info("[AI] chosen=%s | reason=%s", name, reason)
        await bot.send_message(chat_id, f"ИИ выбрал стратегию: <b>{name}</b>\nПричина: {reason}", parse_mode="HTML")

        state["strategy"], state["strategy_name"] = _make_strategy(name, state["params"])
        # сброс внутренней метрики
        state["loss_streak"] = 0
        state["last_trade_ts"] = None
        state["last_activity_ts"] = time.time()

    def _make_strategy(name: str, params: dict):
        name = name.strip()
        if name.lower().startswith("momentum"):
            return Momentum(params.get("Momentum", {})), "Momentum"
        if name.lower().startswith("reversal"):
            return Reversal(params.get("Reversal", {})), "Reversal"
        if name.lower().startswith("breakout"):
            return Breakout(params.get("Breakout", {})), "Breakout"
        if name.lower().startswith("orderbook"):
            return Density(params.get("Orderbook Density", {})), "Orderbook Density"
        if name.lower().startswith("knife"):
            return Knife(params.get("Knife", {})), "Knife"
        # fallback
        return Momentum(params.get("Momentum", {})), "Momentum"

    async def _start_ws(bot: Bot, chat_id: int):
        cfg = state["cfg"]
        import time
        cat = cfg["category"]; sym = cfg["symbol"]
        state["executor"] = Executor(state["bybit"], cat, sym)
        state["pm"] = PositionManager(sym, cfg["base_order_usdt"], cfg["max_loss_usdt"], state["executor"], cat)
        state["ws"] = BybitPublicStream(sym, cfg["ws_kline_interval"], cfg["testnet"])
        await state["ws"].start()
        state["ws_start_ts"] = time.time()
        state["last_activity_ts"] = state["ws_start_ts"]
        log.info("[WATCHDOG] WS started, timestamps initialized")
        asyncio.create_task(_tick_loop(bot, chat_id))

    async def _stop_all(bot: Bot, chat_id: int):
        try:
            if state["pm"] and state["pm"].is_open():
                # Сначала снимаем биржевые SL/TP, затем закрываем позицию маркетом
                try:
                    await state["pm"].ex.cancel_trading_stop()
                    log.info("[STOP] Exchange SL/TP cancelled")
                except Exception as e:
                    log.warning("[STOP] cancel_trading_stop error: %s", e)
                await state["pm"].close_all(state["ws"].last_price if state["ws"] else None)
                _wd_mark_exit(_wd_now())
        finally:
            if state["ws"]:
                await state["ws"].stop()
            await bot.send_message(chat_id, "Торговля остановлена. Все позиции закрыты.", parse_mode="HTML")

    async def _idle_watchdog(bot: Bot, chat_id: int):
        cfg = state["cfg"]
        import asyncio, logging
        log = logging.getLogger("WATCHDOG")
        ttl = int(cfg["no_trade_timeout_sec"])
        loss_requery = int(cfg["loss_streak_requery"])
        log.info("[WATCHDOG] armed (timeout=%ds, loss_streak=%d)", ttl, loss_requery)

        while True:
            try:
                await asyncio.sleep(10)
                pm = state.get("pm")
                open_flag = pm.is_open() if pm else False
                loss_streak = pm.loss_streak if pm else 0
                idle = _wd_idle_seconds()
                log.info("[WATCHDOG] tick idle=%ds open=%s loss_streak=%d", idle, open_flag, loss_streak)

                # Idle re-query: только когда FLAT
                if not open_flag and idle >= ttl:
                    log.warning("[WATCHDOG] idle timeout %ds reached → rotate strategy", idle)
                    await _rotate_strategy("idle_timeout", bot, chat_id)
                    # Сбросить idle‑счётчик: считаем, что активность была
                    _wd_mark_exit(_wd_now())

                # Loss streak re-query
                if open_flag is False and loss_streak >= loss_requery:
                    log.warning("[WATCHDOG] loss streak %d → rotate strategy", loss_streak)
                    await _rotate_strategy("loss_streak", bot, chat_id)
                    if pm:
                        pm.loss_streak = 0
                    _wd_mark_exit(_wd_now())
            except asyncio.CancelledError:
                log.info("[WATCHDOG] cancelled")
                break
            except Exception:
                log.exception("[WATCHDOG] error loop iteration")

    async def _tick_loop(bot: Bot, chat_id: int):
        q = state["ws"].queue()
        cfg = state["cfg"]
        import time
        log.info("[TICK] loop started.")
        while True:
            tick = await q.get()
            # price heartbeat + SL/TP check
            if tick["type"] == "ticker":
                last = state["ws"].last_price
                if last is not None and state["pm"]:
                    closed = await state["pm"].check_sl_tp(last)
                    if closed:
                        state["last_trade_ts"] = time.time()
                        _wd_mark_exit(_wd_now())
                        await bot.send_message(chat_id, f"Позиция закрыта по {'SL' if state['pm'].loss_streak>0 else 'TP'} @ {last:.2f}")

                continue

            if tick["type"] == "kline" and tick["data"].get("confirm"):
                # обновим статистику диапазона/экстремумов
                d = tick["data"]
                h = float(d.get("high") or d.get("h") or 0)
                l = float(d.get("low") or d.get("l") or 0)
                s = state["stats"]
                s["recent_high"] = max(s.get("recent_high", 0), h)
                s["recent_low"] = min(s.get("recent_low", h), l) if s.get("recent_low") else l
                s["atr"] = 0.9 * s.get("atr", 1.0) + 0.1 * (h - l)

                import time
                state["last_activity_ts"] = time.time()
                log.debug("[WATCHDOG] activity at bar close ts=%s", d.get("start") or d.get("timestamp"))

                # стратегия даёт сигнал
                strat = state["strategy"]
                if strat:
                    signal = await strat.on_tick(tick, ctx={"pm": state["pm"], "stats": state["stats"], "orderbook": {}})
                    log.info("[SIGNAL] %s -> %s", state["strategy_name"], signal)
                    # Исполнение: только если FLAT и сигнал enter_*
                    if signal in ("enter_long", "enter_short") and state["pm"] and (not state["pm"].is_open()):
                        _wd_mark_signal(_wd_now())
                        px = float(d.get("close") or d.get("c") or 0)
                        if signal == "enter_long":
                            await state["pm"].open_long(px)
                            await bot.send_message(chat_id, f"Открыт <b>LONG</b> @ {px:.2f}", parse_mode="HTML")
                        else:
                            await state["pm"].open_short(px)
                            await bot.send_message(chat_id, f"Открыт <b>SHORT</b> @ {px:.2f}", parse_mode="HTML")
                        state["last_trade_ts"] = time.time()
                        _wd_mark_entry(_wd_now())

    @router.message(CommandStart())
    async def cmd_start(message: types.Message):
        # Показать меню сразу и не запускать торговлю
        await message.answer("Меню готово", reply_markup=main_kb())
        await message.answer("Бот онлайн. Нажми \u00abСтарт\u00bb, чтобы запустить торговлю.")

    @router.message((F.text.lower() == "старт") | (F.text.lower() == "start"))
    async def cmd_trade_start(message: types.Message):
        await message.answer("Запускаю торговлю…")
        await _rotate_strategy("manual_start", message.bot, message.chat.id)
        await _start_ws(message.bot, message.chat.id)
        if not state["idle_task"] or state["idle_task"].done():
            state["idle_task"] = asyncio.create_task(_idle_watchdog(message.bot, message.chat.id))
            log.info("[WATCHDOG] task launched")

    @router.message(Command("старт"))
    async def cmd_trade_start_ru(message: types.Message):
        await message.answer("Запускаю торговлю…")
        await _rotate_strategy("manual_start", message.bot, message.chat.id)
        await _start_ws(message.bot, message.chat.id)
        if not state["idle_task"] or state["idle_task"].done():
            state["idle_task"] = asyncio.create_task(_idle_watchdog(message.bot, message.chat.id))
            log.info("[WATCHDOG] task launched")

    @router.message(F.text.lower() == "стоп")
    async def cmd_stop(message: types.Message):
        await _stop_all(message.bot, message.chat.id)

    @router.message(F.text.lower() == "баланс")
    async def cmd_balance(message: types.Message):
        cfg = state["cfg"]
        data = await state["bybit"].wallet_balance(cfg["account_type"])

        # v5 wallet-balance structure: result.list[0].coin -> [{coin, equity, availableToWithdraw, ...}]
        acct = (data or {}).get("result", {}).get("list", [])
        usdt_equity = None
        if acct:
            coins = acct[0].get("coin", []) or []
            for c in coins:
                if (c.get("coin") or "").upper() == "USDT":
                    val = c.get("equity") or c.get("walletBalance") or "0"
                    try:
                        usdt_equity = float(val)
                    except Exception:
                        usdt_equity = 0.0
                    break

        if usdt_equity is None:
            # fallback: show 0.00 if parsing failed
            usdt_equity = 0.0

        await message.answer(f"usdt - {usdt_equity:.2f}")

    @router.message(F.text.lower() == "ордера")
    async def cmd_orders(message: types.Message):
        await message.answer("В этой версии см. логи о выставлении ордеров.")

    @router.message(F.text.lower() == "статистика")
    async def cmd_stats(message: types.Message):
        s = state["stats"]; pm = state["pm"]
        await message.answer(
            f"Стратегия: <b>{state.get('strategy_name')}</b>\n"
            f"ATR≈ {s.get('atr'):.2f}\nhi={s.get('recent_high'):.2f} lo={s.get('recent_low'):.2f}\n"
            f"Позиция: {pm.state if pm else 'n/a'}\n"
            f"Лосс-стрик: {pm.loss_streak if pm else 0}",
            parse_mode="HTML"
        )

    dp.include_router(router)
    log.info("[ROUTES] handlers registered")

async def shutdown_cleanup(bot: Bot):
    """Грациозная остановка при завершении процесса: снять биржевые SL/TP, закрыть позицию и остановить WS.
    Ничего не отправляет в Telegram.
    """
    state = LAST_STATE
    if not state:
        logging.getLogger("LOOP").info("[SHUTDOWN] no state to cleanup")
        return
    try:
        pm = state.get("pm")
        ws = state.get("ws")
        if pm and pm.is_open():
            try:
                await pm.ex.cancel_trading_stop()
                log.info("[SHUTDOWN] Exchange SL/TP cancelled")
            except Exception as e:
                log.warning("[SHUTDOWN] cancel_trading_stop error: %s", e)
            try:
                await pm.close_all(ws.last_price if ws else None)
                log.info("[SHUTDOWN] Position closed by market")
            except Exception as e:
                log.warning("[SHUTDOWN] close_all error: %s", e)
    finally:
        try:
            ws = state.get("ws")
            if ws:
                await ws.stop()
                log.info("[SHUTDOWN] WS stopped")
        except Exception as e:
            log.warning("[SHUTDOWN] WS stop error: %s", e)
        log.info("[SHUTDOWN] cleanup done")
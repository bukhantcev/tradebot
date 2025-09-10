import asyncio
import logging
import contextlib
from aiogram import Router, F, types, Bot
from aiogram.filters import CommandStart, Command, or_f
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
        "ws_task": None,
        "ws_running": False,
        "strategy": None,
        "strategy_name": None,
        "params": build_params(),
        "last_trade_ts": None,
        "idle_task": None,
        "loss_streak": 0,
        "stats": {"recent_high": 0, "recent_low": 0, "atr": 1.0},
        "ws_start_ts": None,
        "last_activity_ts": None,
        "start_lock": asyncio.Lock(),
        "last_bar_id": None,      # для дебаунса сигналов по бару
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

        snapshot = {"kline_1m": j1, "kline_5m": j5, "orderbook": ob, "wallet": bal}
        ai = await ask_strategy(cfg["openai_model"], cfg["openai_key"], snapshot)
        name = ai["strategy"]; reason = ai["reason"]
        log.info("[AI] chosen=%s | reason=%s", name, reason)
        await bot.send_message(chat_id, f"ИИ выбрал стратегию: <b>{name}</b>\nПричина: {reason}", parse_mode="HTML")

        state["strategy"], state["strategy_name"] = _make_strategy(name, state["params"])
        # сброс внутренней метрики
        state["loss_streak"] = 0
        state["last_trade_ts"] = None
        state["last_activity_ts"] = time.time()

    def _make_strategy(name: str, params: dict):
        name = name.strip().lower()
        if name.startswith("momentum"):
            return Momentum(params.get("Momentum", {})), "Momentum"
        if name.startswith("reversal"):
            return Reversal(params.get("Reversal", {})), "Reversal"
        if name.startswith("breakout"):
            return Breakout(params.get("Breakout", {})), "Breakout"
        if name.startswith("orderbook"):
            return Density(params.get("Orderbook Density", {})), "Orderbook Density"
        if name.startswith("knife"):
            return Knife(params.get("Knife", {})), "Knife"
        return Momentum(params.get("Momentum", {})), "Momentum"

    async def _start_ws(bot: Bot, chat_id: int):
        cfg = state["cfg"]

        # Если уже запущено — не стартуем второй раз
        if state["ws_running"]:
            log.warning("[WS] already running, skip _start_ws()")
            return

        # Инициализация экзекьютора/PM/WS
        cat = cfg["category"]; sym = cfg["symbol"]
        state["executor"] = Executor(state["bybit"], cat, sym)
        state["pm"] = PositionManager(sym, cfg["base_order_usdt"], cfg["max_loss_usdt"], state["executor"], cat)

        # На всякий случай аккуратно убедимся, что PM в FLAT, не ломая тип state
        try:
            pm = state["pm"]
            if hasattr(pm, "state") and hasattr(pm.state, "side"):
                pm.state.side = "FLAT"
            if hasattr(pm, "qty"):
                pm.qty = 0
            log.info("[FORCE-FLAT] PM side=FLAT before WS start")
        except Exception as e:
            log.warning("[FORCE-FLAT] unable to force flat safely: %s", e)

        state["ws"] = BybitPublicStream(sym, cfg["ws_kline_interval"], cfg["testnet"])
        await state["ws"].start()
        state["ws_running"] = True

        import time
        state["ws_start_ts"] = time.time()
        state["last_activity_ts"] = state["ws_start_ts"]
        state["last_bar_id"] = None

        log.info("[WATCHDOG] WS started, timestamps initialized")

        # Запуск единственного тик-цикла
        if state["ws_task"] and not state["ws_task"].done():
            log.warning("[TICK] previous task still alive — cancelling")
            state["ws_task"].cancel()
            with contextlib.suppress(Exception):
                await state["ws_task"]
        state["ws_task"] = asyncio.create_task(_tick_loop(bot, chat_id))
        log.info("[TICK] loop task created")

    async def _stop_all(bot: Bot, chat_id: int):
        try:
            # Сначала отменяем стопы и закрываем позицию, если она вдруг открыта
            if state["pm"] and state["pm"].is_open():
                try:
                    await state["pm"].ex.cancel_trading_stop()
                    log.info("[STOP] Exchange SL/TP cancelled")
                except Exception as e:
                    log.warning("[STOP] cancel_trading_stop error: %s", e)
                await state["pm"].close_all(state["ws"].last_price if state["ws"] else None)
                log.info("[STOP] Position market-closed")

            # Останавливаем тик-цикл
            if state["ws_task"] and not state["ws_task"].done():
                state["ws_task"].cancel()
                with contextlib.suppress(Exception):
                    await state["ws_task"]
                log.info("[STOP] tick task cancelled")
                state["ws_task"] = None

        finally:
            # Останавливаем WS
            if state["ws"]:
                try:
                    await state["ws"].stop()
                except Exception as e:
                    log.warning("[STOP] WS stop error: %s", e)
                state["ws"] = None
            state["ws_running"] = False
            state["last_bar_id"] = None
            await bot.send_message(chat_id, "Торговля остановлена. Все позиции закрыты.", parse_mode="HTML")

    async def _idle_watchdog(bot: Bot, chat_id: int):
        cfg = state["cfg"]
        import time
        log.info("[WATCHDOG] started (idle=%ds, loss_streak=%d)", cfg["no_trade_timeout_sec"], cfg["loss_streak_requery"])
        while True:
            try:
                await asyncio.sleep(10)
                now = time.time()
                base_ts = state.get("last_trade_ts") or state.get("last_activity_ts") or state.get("ws_start_ts")
                pm = state.get("pm")
                open_flag = pm.is_open() if pm else None
                loss_streak = pm.loss_streak if pm else 0
                idle = int(now - base_ts) if base_ts else -1
                log.info("[WATCHDOG] tick idle=%ds open=%s loss_streak=%d", idle, open_flag, loss_streak)

                # Idle re-query (только когда FLAT)
                if base_ts and idle >= cfg["no_trade_timeout_sec"]:
                    log.warning("[WATCHDOG] idle %ds >= %d -> requery AI", idle, cfg["no_trade_timeout_sec"])
                    await _rotate_strategy("idle_timeout", bot, chat_id)
                    state["last_activity_ts"] = now

                # Loss streak re-query (только когда FLAT)
                if pm and pm.loss_streak >= cfg["loss_streak_requery"]:
                    log.warning("[WATCHDOG] loss streak %d -> requery AI", pm.loss_streak)
                    await _rotate_strategy("loss_streak", bot, chat_id)
                    pm.loss_streak = 0

            except asyncio.CancelledError:
                log.info("[WATCHDOG] cancelled")
                raise
            except Exception as e:
                log.exception("[WATCHDOG] error: %s", e)

    async def _tick_loop(bot: Bot, chat_id: int):
        q = state["ws"].queue()
        cfg = state["cfg"]
        import time

        log.info("[TICK] loop started.")
        while True:
            try:
                tick = await q.get()

                # ticker heartbeat + SL/TP trailing check
                if tick["type"] == "ticker":
                    last = state["ws"].last_price
                    if last is not None and state["pm"]:
                        closed = await state["pm"].check_sl_tp(last)
                        if closed:
                            state["last_trade_ts"] = time.time()
                            state["last_activity_ts"] = state["last_trade_ts"]
                            # не шлём лишнего в чат, позиция закроется биржей
                    continue

                if tick["type"] == "kline" and tick["data"].get("confirm"):
                    d = tick["data"]

                    # --- дебаунс по бару ---
                    bar_id = d.get("start") or d.get("timestamp") or d.get("startTime")
                    if bar_id is None:
                        # бэкап: используем close цену + секундный таймштамп
                        bar_id = f"{d.get('close')}-{int(time.time()//cfg.get('ws_kline_interval',1))}"
                    if state["last_bar_id"] == bar_id:
                        log.debug("[DEBOUNCE] skip duplicate bar %s", bar_id)
                        continue
                    state["last_bar_id"] = bar_id

                    # обновим статистику диапазона/экстремумов
                    h = float(d.get("high") or d.get("h") or 0)
                    l = float(d.get("low") or d.get("l") or 0)
                    s = state["stats"]
                    s["recent_high"] = max(s.get("recent_high", 0), h)
                    s["recent_low"] = min(s.get("recent_low", h), l) if s.get("recent_low") else l
                    s["atr"] = 0.9 * s.get("atr", 1.0) + 0.1 * (h - l)

                    # state["last_activity_ts"] = time.time()

                    # стратегия даёт сигнал
                    strat = state["strategy"]
                    if not strat:
                        log.warning("[TICK] no strategy set")
                        continue

                    signal = await strat.on_tick(tick, ctx={"pm": state["pm"], "stats": state["stats"], "orderbook": {}})
                    log.info("[SIGNAL] %s -> %s", state["strategy_name"], signal)

                    # Исполнение: только если FLAT и сигнал enter_*
                    if signal in ("enter_long", "enter_short") and state["pm"] and (not state["pm"].is_open()):
                        px = float(d.get("close") or d.get("c") or 0)
                        try:
                            if signal == "enter_long":
                                await state["pm"].open_long(px)
                                await bot.send_message(chat_id, f"Открыт <b>LONG</b> @ {px:.2f}", parse_mode="HTML")
                            else:
                                await state["pm"].open_short(px)
                                await bot.send_message(chat_id, f"Открыт <b>SHORT</b> @ {px:.2f}", parse_mode="HTML")
                            state["last_trade_ts"] = time.time()
                            state["last_activity_ts"] = state["last_trade_ts"]
                        except Exception as e:
                            log.exception("[EXEC] open error: %s", e)
                    # иначе игнорируем

            except asyncio.CancelledError:
                log.info("[TICK] cancelled")
                raise
            except Exception as e:
                log.exception("[TICK] error: %s", e)

    @router.message(CommandStart())
    async def cmd_start(message: types.Message):
        # Показать меню сразу и не запускать торговлю
        await message.answer("Меню готово", reply_markup=main_kb())
        await message.answer("Бот онлайн. Нажми «Старт», чтобы запустить торговлю.")

    @router.message(or_f(Command("старт"), F.text.lower().in_({"старт", "start"})))
    async def cmd_trade_start(message: types.Message):
        async with state["start_lock"]:
            await message.answer("Запускаю торговлю…")
            # Прежде чем запускать — остановим предыдущее (если вдруг висит)
            if state["ws_running"]:
                log.warning("[START] WS already running -> stopping previous first")
                await _stop_all(message.bot, message.chat.id)

            await _rotate_strategy("manual_start", message.bot, message.chat.id)
            await _start_ws(message.bot, message.chat.id)

            if not state["idle_task"] or state["idle_task"].done():
                state["idle_task"] = asyncio.create_task(_idle_watchdog(message.bot, message.chat.id))
                log.info("[WATCHDOG] task launched")

    @router.message(F.text.lower() == "стоп")
    async def cmd_stop(message: types.Message):
        async with state["start_lock"]:
            await _stop_all(message.bot, message.chat.id)

    @router.message(F.text.lower() == "баланс")
    async def cmd_balance(message: types.Message):
        cfg = state["cfg"]
        data = await state["bybit"].wallet_balance(cfg["account_type"])
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
            usdt_equity = 0.0
        await message.answer(f"usdt - {usdt_equity:.2f}")

    @router.message(F.text.lower() == "ордера")
    async def cmd_orders(message: types.Message):
        await message.answer("В этой версии см. логи о выставлении ордеров.")

    @router.message(F.text.lower() == "статистика")
    async def cmd_stats(message: types.Message):
        s = state["stats"]; pm = state["pm"]
        open_flag = pm.is_open() if pm else False
        await message.answer(
            f"Стратегия: <b>{state.get('strategy_name')}</b>\n"
            f"ATR≈ {s.get('atr'):.2f}\nhi={s.get('recent_high'):.2f} lo={s.get('recent_low'):.2f}\n"
            f"Открыта ли позиция: {open_flag}\n"
            f"Позиция (raw): {getattr(pm, 'state', 'n/a')}\n"
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
        open_now = False
        try:
            open_now = bool(pm and hasattr(pm, "is_open") and pm.is_open())
        except Exception as e:
            log.warning("[SHUTDOWN] is_open check failed: %s", e)
            open_now = False
        if open_now:
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
            # стоп тик-таска
            if state.get("ws_task") and not state["ws_task"].done():
                state["ws_task"].cancel()
                with contextlib.suppress(Exception):
                    await state["ws_task"]
            # стоп WS
            ws = state.get("ws")
            if ws:
                await ws.stop()
                log.info("[SHUTDOWN] WS stopped")
            state["ws_running"] = False
            state["last_bar_id"] = None
        except Exception as e:
            log.warning("[SHUTDOWN] WS stop error: %s", e)
        log.info("[SHUTDOWN] cleanup done")
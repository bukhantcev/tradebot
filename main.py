# main.py
import os
import asyncio

import logging
from dataclasses import dataclass

from datetime import datetime, timezone

from typing import Optional, Any, List

from bybit_client import bybit_set_leverage, read_market, _kline_to_df, bybit_cancel_entry_orders, bybit_cancel_all, \
    market_close_all, hold_guard_sl, enforce_sl_must_have, sweep_after_sl, bybit_wallet_usdt
from clients import BOT, RETRY_ATTEMPTS, RETRY_DELAY_SEC, \
    SL_TICKS, TREND_SL_MULT, \
    LEVERAGE, BOOTSTRAP_HOURS, REENTER_AFTER_SL_SEC, \
    POLL_TICK_MS, AI_POLL_SEC, DP, USE_LOCAL_DECIDER
from helpers import tg_send
from logger import log
from reshalka import get_decision
from trader import apply_trend_confirmation, do_trend, do_flat, do_hold
from models import STATE, Regime, AIDecision, Side

# ---- Optional SSL relax (env BYBIT_VERIFY_SSL=false) ----
if os.getenv("BYBIT_VERIFY_SSL", "true").lower() == "false":
    os.environ["PYTHONHTTPSVERIFY"] = "0"

from dotenv import load_dotenv

from aiogram import F
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from aiogram.enums import ParseMode

# ------------------ .env & logging ------------------
load_dotenv()




# ------------------ Trading primitives with retries ------------------
RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", str(RETRY_ATTEMPTS)))
RETRY_DELAY_SEC = int(os.getenv("RETRY_DELAY_SEC", str(RETRY_DELAY_SEC)))



# ------------------ Main trading loop ------------------
async def trading_loop():
    symbol = STATE.symbol
    STATE.prev_regime_was_trend = False
    await tg_send("🚀 Старт торговли")

    # leverage
    try:
        bybit_set_leverage(symbol, LEVERAGE)
        log.info("[LEV] set leverage=%dx", LEVERAGE)
    except Exception as e:
        log.warning("[LEV] fail set leverage: %s", e)

    # initial bootstrap (5 hours history) with fallback
    try:
        md = read_market(symbol, BOOTSTRAP_HOURS)
        df = _kline_to_df(md.kline_1m)
        await tg_send("📥 Подгрузка данных за 5 часов завершена")
    except Exception as e:
        log.exception("[BOOTSTRAP] failed 5h load: %s", e)
        await tg_send("❌ Ошибка загрузки истории за 5 часов. Пробую загрузить 60 минут…")
        md = read_market(symbol, 1)
        df = _kline_to_df(md.kline_1m)
        await tg_send("📥 Подгружено за 60 минут (fallback)")

    # Инициализация маркера последней закрытой 1m свечи
    if len(df) >= 2:
        STATE.last_ai_prev_ts = int(df.iloc[-2]["ts"])

    # Первое решение — по закрытой свече
    dec = await get_decision(symbol, df, md)
    STATE.current_regime = dec.regime
    STATE.current_side = dec.side
    STATE.last_decision_sl_ticks = dec.sl_ticks
    await tg_send(f"🧭 Режим: <b>{dec.regime.value}</b> | Направление: <b>{dec.side.value}</b>")

    last_ai_time = 0.0
    while STATE.is_trading:
        try:
            now_mono = asyncio.get_event_loop().time()

            # Читаем рынок
            history_hours = 4 if USE_LOCAL_DECIDER else 1
            md = read_market(symbol, history_hours)
            df = _kline_to_df(md.kline_1m)

            # ---------- РАННИЙ КУЛДАУН ПОСЛЕ ВЫХОДА В НОЛЬ ----------
            # Если на прошлом шаге позиция была >0, а сейчас 0 — считаем, что был полный выход (в т.ч. SL).
            if STATE.last_pos_size > 0 and md.position.size == 0:
                STATE.trail_anchor = None
                STATE.last_sl_hit_at = datetime.now(timezone.utc)
                STATE.reenter_block_until = now_mono + REENTER_AFTER_SL_SEC
                log.info("[COOLDOWN] (EARLY) блок входа до %.1f (now=%.1f) после выхода в 0",
                         STATE.reenter_block_until, now_mono)
                # перестраховка: снимаем только входные лимитки (SL/TP уже неактуальны без позиции)
                try:
                    bybit_cancel_entry_orders(symbol)
                    log.info("[CANCEL-ENTRY] cancelled entry orders after exit")
                except Exception:
                    pass
                # обновляем маркер и уходим в следующий тик, чтобы в этом не было входов
                STATE.last_pos_size = md.position.size
                await asyncio.sleep(POLL_TICK_MS / 1000.0)
                continue
            # --------------------------------------------------------

            # --- Получаем решение ТОЛЬКО после закрытия новой минутки (+ резервная страховка) ---
            if len(df) >= 2:
                prev_ts_now = int(df.iloc[-2]["ts"])
            else:
                prev_ts_now = None

            need_decision = False
            if prev_ts_now is not None and prev_ts_now != STATE.last_ai_prev_ts:
                need_decision = True
                STATE.last_ai_prev_ts = prev_ts_now
            elif (now_mono - last_ai_time) >= AI_POLL_SEC * 3:
                # страховка: если рынок «залип» и свеча не закрывается долго
                need_decision = True

            if need_decision:
                dec_new = await get_decision(symbol, df, md)
                STATE.last_decision_sl_ticks = dec_new.sl_ticks
                regime_changed = dec_new.regime != STATE.current_regime
                side_changed = (dec_new.side != STATE.current_side) and dec_new.regime != Regime.HOLD

                # Подтверждение разворота тренда
                if dec_new.regime == Regime.TREND:
                    flipped = await apply_trend_confirmation(dec_new, md, now_mono)
                    if flipped:
                        last_ai_time = now_mono
                        # после flip сразу в следующий тик (пускай гистерезис/кулдаун отработают)
                        await asyncio.sleep(POLL_TICK_MS / 1000.0)
                        continue

                # Спец-логика для FLAT
                if STATE.current_regime == Regime.FLAT and dec_new.regime == Regime.FLAT:
                    pos_now = md.position
                    if pos_now.size > 0:
                        if dec_new.side == pos_now.side:
                            STATE.current_side = dec_new.side
                            STATE.current_regime = Regime.FLAT
                            last_ai_time = now_mono
                        else:
                            await tg_send("🔁 FLAT: смена направления при открытой позиции → закрываю по рынку")
                            # РАНЕЕ ставим короткий кулдаун, чтобы в этот же тик не войти обратно
                            STATE.reenter_block_until = now_mono + 3
                            try:
                                bybit_cancel_all(symbol)
                            except Exception:
                                pass
                            await market_close_all(symbol)
                            log.info("[COOLDOWN] блок входа до %.1f (now=%.1f) после смены стороны FLAT",
                                     STATE.reenter_block_until, now_mono)
                            STATE.current_regime = Regime.FLAT
                            STATE.current_side = dec_new.side
                            last_ai_time = now_mono
                            await tg_send(
                                "🤖 Обновление:\n"
                                f"Режим: <b>{dec_new.regime.value}</b>\nНаправление: <b>{dec_new.side.value}</b>"
                                + (f"\nКомментарий: {dec_new.comment}" if dec_new.comment else "")
                            )
                    else:
                        if side_changed:
                            try:
                                bybit_cancel_all(symbol)
                            except Exception:
                                pass
                            STATE.current_side = dec_new.side
                            STATE.current_regime = Regime.FLAT
                            last_ai_time = now_mono
                            await tg_send(
                                "🤖 Обновление:\n"
                                f"Режим: <b>{dec_new.regime.value}</b>\nНаправление: <b>{dec_new.side.value}</b>"
                                + (f"\nКомментарий: {dec_new.comment}" if dec_new.comment else "")
                            )
                else:
                    if regime_changed or side_changed:
                        if dec_new.regime == Regime.HOLD:
                            # Переход в HOLD: позиции НЕ закрываем, снимаем только входные лимитки
                            await tg_send("⏸ Переход в HOLD → позицию оставляю, снимаю только входные лимит-заявки")
                            try:
                                bybit_cancel_entry_orders(symbol)
                            except Exception:
                                pass
                            STATE.prev_regime_was_trend = (STATE.current_regime == Regime.TREND)
                            STATE.current_regime = Regime.HOLD
                            await tg_send(
                                "🤖 Обновление:\n"
                                f"Режим: <b>{dec_new.regime.value}</b>\nНаправление: <b>{STATE.current_side.value}</b>"
                                + (f"\nКомментарий: {dec_new.comment}" if dec_new.comment else "")
                            )
                        elif dec_new.regime == Regime.FLAT:
                            # Переход в FLAT: ВСЕГДА закрываем позиции и удаляем ВСЕ заявки
                            await tg_send("↔️ Переход в FLAT → снимаю все заявки и закрываю позиции")
                            # ставим короткий кулдаун заранее, чтобы вход не произошёл в этот же тик
                            STATE.reenter_block_until = now_mono + 3
                            try:
                                bybit_cancel_all(symbol)
                            except Exception:
                                pass
                            await market_close_all(symbol)
                            log.info("[COOLDOWN] блок входа до %.1f (now=%.1f) при переходе во FLAT",
                                     STATE.reenter_block_until, now_mono)
                            STATE.current_regime = Regime.FLAT
                            STATE.current_side = dec_new.side
                            STATE.last_flat_prev_ts = None  # принудим свежую перекотировку лимиток на следующей минуте
                            await tg_send(
                                "🤖 Обновление:\n"
                                f"Режим: <b>{dec_new.regime.value}</b>\nНаправление: <b>{dec_new.side.value}</b>"
                                + (f"\nКомментарий: {dec_new.comment}" if dec_new.comment else "")
                            )
                        else:
                            # Переход в TREND или смена стороны вне HOLD/FLAT-спецкейса
                            await tg_send("🔁 Переключение режима/направления → Закрываю позиции")
                            # РАНЕЕ ставим кулдаун перед закрытием
                            STATE.reenter_block_until = now_mono + 3
                            try:
                                bybit_cancel_all(symbol)
                            except Exception:
                                pass
                            await market_close_all(symbol)
                            log.info("[COOLDOWN] блок входа до %.1f (now=%.1f) после переключения режима",
                                     STATE.reenter_block_until, now_mono)
                            STATE.current_regime = dec_new.regime
                            STATE.current_side = dec_new.side
                            await tg_send(
                                "🤖 Обновление:\n"
                                f"Режим: <b>{dec_new.regime.value}</b>\nНаправление: <b>{dec_new.side.value}</b>"
                                + (f"\nКомментарий: {dec_new.comment}" if dec_new.comment else "")
                            )
                    last_ai_time = now_mono

            # Если идёт кулдаун — вообще не трогаем исполнение режимов
            if now_mono < getattr(STATE, "reenter_block_until", 0.0):
                await asyncio.sleep(POLL_TICK_MS / 1000.0)
                # обновим маркер размера и продолжим
                STATE.last_pos_size = md.position.size
                continue

            # --- Исполнение по текущему режиму ---
            if STATE.current_regime == Regime.TREND:
                await do_trend(md, AIDecision(STATE.current_regime, STATE.current_side))
            elif STATE.current_regime == Regime.FLAT:
                await do_flat(md, AIDecision(STATE.current_regime, STATE.current_side))
            else:
                await do_hold(md, AIDecision(STATE.current_regime, STATE.current_side))
                await hold_guard_sl(symbol, md)  # в HOLD следим, что SL живой

            # --- Если позиция только что появилась (0 → >0), 10с на установку SL ---
            if STATE.last_pos_size <= 0 and md.position.size > 0:
                try:
                    base_ticks = STATE.last_decision_sl_ticks if STATE.last_decision_sl_ticks is not None else SL_TICKS
                    try:
                        base_ticks = int(base_ticks)
                    except Exception:
                        base_ticks = int(SL_TICKS)
                    sl_ticks_enforce = int(base_ticks * (TREND_SL_MULT if STATE.current_regime == Regime.TREND else 1))
                    STATE.trail_anchor = md.position.avg_price
                    await enforce_sl_must_have(
                        symbol,
                        md.position.side,
                        md.filters,
                        sl_ticks=sl_ticks_enforce,
                        timeout_sec=10,
                    )
                except Exception as e:
                    log.exception("[ENFORCE-SL] непредвиденная ошибка: %s", e)

            # --- Добивание после SL, если остаток висит ---
            await sweep_after_sl(symbol, md, md.filters)

            # Обновляем маркер размера позиции (для ранней ветки выше)
            STATE.last_pos_size = md.position.size

            await asyncio.sleep(POLL_TICK_MS / 1000.0)
        except Exception as e:
            log.exception("[LOOP] error: %s", e)
            await asyncio.sleep(1)

    await tg_send("⏹️ Торговля остановлена")
# ------------------ Telegram UI ------------------
def kb():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="▶️ Старт"), KeyboardButton(text="⏹️ Стоп")],
            [KeyboardButton(text="💰 Баланс"), KeyboardButton(text="🚫 Закрыть всё")],
        ],
        resize_keyboard=True,
        one_time_keyboard=False,
        input_field_placeholder="Выберите действие…",
        selective=False,
    )

@DP.message(F.text.regexp(r"^/start$"))
async def start_cmd(msg: Message):
    await msg.answer(
        "Бот готов. Кнопки снизу.\n\n"
        "▶️ Старт — запустить торговлю\n"
        "⏹️ Стоп — остановить и закрыть всё\n"
        "💰 Баланс — показать баланс\n"
        "🚫 Закрыть всё — закрыть позиции и отменить заявки",
        reply_markup=kb(),
    )

# --- Reply keyboard handlers ---
@DP.message(F.text == "▶️ Старт")
async def btn_start(msg: Message):
    if STATE.is_trading:
        await msg.answer("⚠️ Торговля уже запущена", reply_markup=kb())
        return
    STATE.is_trading = True
    STATE.current_regime = Regime.HOLD
    STATE.current_side = Side.NONE
    if STATE.loop_task and not STATE.loop_task.done():
        STATE.loop_task.cancel()
    STATE.loop_task = asyncio.create_task(trading_loop())
    await msg.answer(f"✅ Торговля запущена. Источник сигналов: {'ЛОКАЛЬНЫЙ' if USE_LOCAL_DECIDER else 'ИИ'}", reply_markup=kb())

@DP.message(F.text == "⏹️ Стоп")
async def btn_stop(msg: Message):
    STATE.is_trading = False
    try:
        bybit_cancel_all(STATE.symbol)
    except Exception:
        pass
    await market_close_all(STATE.symbol)
    await msg.answer("⏹️ Торговля остановлена и позиции закрыты", reply_markup=kb())

@DP.message(F.text == "💰 Баланс")
async def btn_balance(msg: Message):
    bal = bybit_wallet_usdt()
    await msg.answer(f"💰 Баланс: <b>{bal:.2f} USDT</b>", parse_mode=ParseMode.HTML, reply_markup=kb())

@DP.message(F.text == "🚫 Закрыть всё")
async def btn_close_all(msg: Message):
    try:
        bybit_cancel_all(STATE.symbol)
    except Exception:
        pass
    await market_close_all(STATE.symbol)
    await msg.answer("🚫 Все позиции закрыты", reply_markup=kb())

# ------------------ Entry ------------------
async def main():
    log.info("Starting Bybit AI Trading Bot…")
    if not BOT:
        log.error("No TELEGRAM_BOT_TOKEN provided")
        return
    await tg_send("🤖 Бот запущен. Нажмите Старт для начала торговли.\n" +
                  f"Символ: <b>{STATE.symbol}</b>, Леверидж: <b>{LEVERAGE}x</b>\n" +
                  ("ТЕСТНЕТ" if os.getenv("BYBIT_TESTNET","true").lower()=="true" else "РЕАЛ") +
                  f"\nИсточник сигналов: <b>{'ЛОКАЛЬНЫЙ' if USE_LOCAL_DECIDER else 'ИИ'}</b>")
    await DP.start_polling(BOT)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        log.info("Stopped.")
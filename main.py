from __future__ import annotations

import time
from typing import Optional

from logger import get_logger
from pybit.unified_trading import HTTP
from config import Config
from bybit_request import get_min_qty, get_open_positions
from norm_price import make_qty
from preloader import load_history_candles

log = get_logger("MAIN")
cfg = Config()

# === Swing (Daily) config ===
WATCH_SYMBOLS = [cfg.symbol]  # только символ из конфигурации
SWING_INTERVAL = "D"                 # дневные свечи
SWING_RSI_PERIOD = 14
SWING_LOOKBACK_PEAK = int(getattr(cfg, "swing_lookback_peak", 200))  # дней для поиска локального максимума
SWING_DD_MIN = float(getattr(cfg, "swing_dd_min", 15.0))             # мин. просадка от максимума, %
SWING_DD_MAX = float(getattr(cfg, "swing_dd_max", 20.0))             # макс. просадка от максимума, %
SWING_SL_PCT = float(getattr(cfg, "swing_sl_pct", 8.0))              # стоп-лосс от цены входа, %
SWING_TP_PCT = float(getattr(cfg, "swing_tp_pct", 25.0))             # тейк-профит, %
SWING_CHECK_SECONDS = int(getattr(cfg, "swing_check_seconds", 3600)) # проверка раз в час

# ---- Static pre-calcs ----
is_testnet = True if str(cfg.testnet).lower() == "true" else False
# qty вычисляем для каждого символа отдельно при выставлении ордера


def _client() -> HTTP:
    return HTTP(
        testnet=is_testnet,
        api_key=cfg.bybit_api_key_test if is_testnet else cfg.bybit_api_key_main,
        api_secret=cfg.bybit_api_secret_test if is_testnet else cfg.bybit_api_secret_main,
        recv_window=20_000,
        timeout=30,
        max_retries=3,
    )


def close_market():
    """Закрыть позицию рынком (если вдруг понадобится)."""
    try:
        pos = get_open_positions()[0]
    except Exception:
        pos = {}
    side = pos.get("side", "")
    size = float(pos.get("size", "0") or 0)
    if not side or size <= 0:
        return

    try:
        cl = _client()
        opposite = "Sell" if side == "Buy" else "Buy"
        payload = {
            "category": cfg.category,
            "symbol": cfg.symbol,
            "side": opposite,
            "orderType": "Market",
            "qty": str(size),
            "reduceOnly": True,
        }
        resp = cl.place_order(**payload)
        log.info(f"POSITION CLOSED | {payload} | resp={resp}")
    except Exception as e:
        log.error(f"Failed to close position: {e}")


# ---- Helpers ----

def _rsi_wilder(closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, period + 1):
        d = closes[i] - closes[i-1]
        gains.append(d if d > 0 else 0.0)
        losses.append(-d if d < 0 else 0.0)
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i-1]
        gain = d if d > 0 else 0.0
        loss = -d if d < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _get_position_for(symbol: str) -> dict:
    """Позиция по конкретному символу (или пустая)."""
    try:
        positions = get_open_positions()
        if isinstance(positions, list):
            for p in positions:
                if str(p.get("symbol", "")).upper() == symbol.upper():
                    return p
            if positions:
                return positions[0]
        return {}
    except Exception:
        return {}


def _place_market_buy_with_sl_tp(symbol: str, sl_price: float, tp_price: float) -> bool:
    """Рыночная покупка + сразу SL/TP."""
    try:
        cl = _client()
        # Пересчитываем qty для КОНКРЕТНОГО символа
        try:
            # Пытаемся вызвать get_min_qty со сменой символа (если функция поддерживает аргумент)
            try:
                min_info = get_min_qty(symbol)
            except TypeError:
                # Фоллбэк: если сигнатура без аргумента, просто вызываем и используем как есть
                min_info = get_min_qty()
            sym_min_qty = float(min_info.get("min_qty", 0.0))
            sym_decimals = int(min_info.get("decimals", 3))
            mult = float(getattr(cfg, "mult_qty", 1))
            qty = round(sym_min_qty * mult, sym_decimals)
        except Exception:
            # На крайний случай — минимальный безопасный лот
            qty = 0.001

        tick_size = float(getattr(cfg, "tick_size", 0.1))
        sl_price = round(sl_price / tick_size) * tick_size
        tp_price = round(tp_price / tick_size) * tick_size
        payload = {
            "category": cfg.category,
            "symbol": symbol,
            "side": "Buy",
            "orderType": "Market",
            "qty": qty,
            "stopLoss": str(sl_price),
            "takeProfit": str(tp_price),
        }
        log.info(f"QTY RESOLVED | {symbol} → qty={qty}")
        resp = cl.place_order(**payload)
        log.info(f"SWING BUY → {symbol} | qty={qty} SL={sl_price} TP={tp_price} | resp={resp}")
        return True
    except Exception as e:
        log.error(f"Failed to place swing market order for {symbol}: {e}")
        return False


# ---- Swing (daily) detector ----

def _check_daily_swing_and_trade(symbol: str, candles: list[dict]) -> None:
    """Свинг на дневках:
    1) просадка 15–20% от локального максимума (за SWING_LOOKBACK_PEAK дней)
    2) RSI(14) < 30
    3) Покупка, SL -8%, TP +25%
    """
    if len(candles) < max(SWING_LOOKBACK_PEAK, SWING_RSI_PERIOD + 1):
        log.info(f"SWING {symbol}: not enough candles")
        return

    closes = [float(c["c"]) for c in candles]
    highs = [float(c["h"]) for c in candles]

    last = candles[-1]
    close_price = float(last["c"])

    recent_high = max(highs[-SWING_LOOKBACK_PEAK:])
    if recent_high <= 0:
        return
    drawdown_pct = (recent_high - close_price) / recent_high * 100.0

    rsi_val = _rsi_wilder(closes, SWING_RSI_PERIOD)
    rsi_str = f"{rsi_val:.1f}" if rsi_val is not None else "NaN"

    log.info(f"SWING {symbol} | close={close_price:.2f} peak={recent_high:.2f} dd={drawdown_pct:.2f}% RSI={rsi_str}")

    # уже есть позиция?
    pos = _get_position_for(symbol)
    has_pos = (pos.get("side") in ("Buy", "Sell")) and float(pos.get("size", 0) or 0) > 0
    if has_pos:
        log.info(f"SWING {symbol}: position already open → waiting (SL/TP on exchange)")
        return

    # вход
    if rsi_val is not None and rsi_val < 30.0 and SWING_DD_MIN <= drawdown_pct <= SWING_DD_MAX:
        entry = close_price
        sl = entry * (1.0 - SWING_SL_PCT / 100.0)
        tp = entry * (1.0 + SWING_TP_PCT / 100.0)
        _place_market_buy_with_sl_tp(symbol, sl, tp)
    else:
        log.info(f"SWING {symbol}: no setup | need dd∈[{SWING_DD_MIN},{SWING_DD_MAX}] & RSI<30")

def run_candle_close_loop():
    """Основной цикл: ТОЛЬКО свинг. Проверка каждый час."""
    log.info("START SWING DAILY STRATEGY | symbols=%s | check_every=%ds", WATCH_SYMBOLS, SWING_CHECK_SECONDS)

    next_check_ts = 0.0
    while True:
        try:
            now = time.time()
            if now >= next_check_ts:
                for sym in WATCH_SYMBOLS:
                    daily_need = max(SWING_LOOKBACK_PEAK + 5, SWING_RSI_PERIOD + 20)
                    d_candles = load_history_candles(
                        limit=daily_need,
                        interval=SWING_INTERVAL,
                        symbol=sym,
                        category=cfg.category
                    )
                    if not d_candles:
                        log.warning(f"SWING {sym}: no candles returned")
                        continue
                    _check_daily_swing_and_trade(sym, d_candles)

                next_check_ts = now + SWING_CHECK_SECONDS

            time.sleep(1.0)  # не грузим CPU

        except KeyboardInterrupt:
            log.info("STOPPED by user")
            break
        except Exception as e:
            log.error(f"ERROR in main loop: {e}")
            time.sleep(5.0)


if __name__ == "__main__":
    run_candle_close_loop()
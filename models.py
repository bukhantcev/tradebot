
import os
import asyncio
from enum import Enum

from dataclasses import dataclass

from datetime import datetime

from typing import Optional, Any, List




if os.getenv("BYBIT_VERIFY_SSL", "true").lower() == "false":
    os.environ["PYTHONHTTPSVERIFY"] = "0"

from dotenv import load_dotenv

load_dotenv()


class Regime(str, Enum):
    TREND = "trend"
    FLAT = "flat"
    HOLD = "hold"

class Side(str, Enum):
    BUY = "Buy"
    SELL = "Sell"
    NONE = "None"


@dataclass
class AIDecision:
    regime: Regime
    side: Side          # For HOLD, may be NONE
    sl_ticks: Optional[int] = None   # optional, can be from .env instead
    comment: str = ""

@dataclass
class MdFilters:
    tick_size: float
    qty_step: float
    min_qty: float

@dataclass
class PositionInfo:
    size: float
    side: Side
    avg_price: float

@dataclass
class MarketData:
    last_price: float
    filters: MdFilters
    kline_1m: List[List[Any]]  # raw bybit kline list
    position: PositionInfo
    balance_usdt: float



class BotState:
    def __init__(self):
        self.last_decision_sl_ticks: Optional[int] = None  # последний sl_ticks от локалки/ИИ
        self.trail_anchor: Optional[float] = None  # максимум/минимум с момента входа
        self.is_trading = False
        self.current_regime: Regime = Regime.HOLD
        self.current_side: Side = Side.NONE
        self.loop_task: Optional[asyncio.Task] = None
        self.last_ai_text: str = ""
        self.flat_entry_order_id: Optional[str] = None
        self.sl_order_id: Optional[str] = None
        self.tp_order_id: Optional[str] = None
        self.last_sl_hit_at: Optional[datetime] = None
        self.symbol = os.getenv("SYMBOL", "BTCUSDT")
        self.category = "linear"
        self.last_flat_prev_ts: Optional[int] = None

        # Новые поля:
        self.last_ai_prev_ts: Optional[int] = None   # последняя обработанная ЗАКРЫТАЯ 1m свеча (ts)
        self.last_pos_size: float = 0.0              # размер позиции на предыдущем тике (детект открытия)
        self.last_sl_price: Optional[float] = None   # последний установленный SL (цена)
        self.reenter_block_until: float = 0.0        # запрет входа до этого времени (monotonic сек)

STATE = BotState()
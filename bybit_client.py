from dataclasses import dataclass
from typing import Optional

@dataclass
class Orders:
    category: str
    symbol: str
    side: str
    orderType: str
    qty: str
    timeInForce: Optional[str] = None
    price: Optional[float] = None
    triggerPrice: Optional[float] = None
    triggerBy: Optional[str] = None
    reduceOnly: Optional[bool] = None
    closeOnTrigger: Optional[bool] = None
    orderFilter: Optional[str] = None
    takeProfit: Optional[float] = None
    stopLoss: Optional[float] = None
    tpTriggerBy: Optional[str] = None
    slTriggerBy: Optional[str] = None
    trailingStop: Optional[float] = None
    trailingActive: Optional[float] = None
    orderLinkId: Optional[str] = None
    positionIdx: Optional[int] = None
    leverage: Optional[int] = None
    externalOid: Optional[str] = None
    remark: Optional[str] = None
    tpSlMode: Optional[str] = None
    closeOnTrigger: Optional[bool] = None
    takeProfitTriggerBy: Optional[str] = None
    stopLossTriggerBy: Optional[str] = None

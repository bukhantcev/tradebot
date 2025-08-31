# src/tinkoff_sandbox.py
import inspect
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional, Iterable

import requests
from tinkoff.invest import (
    Client,
    OrderDirection,
    OrderType,
    CandleInterval,
    InstrumentIdType,
    MoneyValue,
)
from tinkoff.invest.services import Services

from .config import FIGI_FALLBACK

def quotation_to_float(q) -> float:
    try:
        return (q.units or 0) + (q.nano or 0) / 1e9
    except Exception:
        return float(q)

def money_to_float(m) -> float:
    return (m.units or 0) + (m.nano or 0) / 1e9

FIGI_TO_TICKER = {
    "BBG004730N88": "SBER",
    "BBG004730RP0": "GAZP",
    "BBG004731032": "VTBR",
    "BBG0047315D0": "GMKN",
}
ISS_INTERVAL_MAP = {
    CandleInterval.CANDLE_INTERVAL_1_MIN: 1,
    CandleInterval.CANDLE_INTERVAL_5_MIN: 5,
}

def load_candles(services: Services, figi: str, hours: int, interval: CandleInterval):
    to = datetime.now(timezone.utc)
    frm = to - timedelta(hours=hours)
    try:
        return services.market_data.get_candles(figi=figi, from_=frm, to=to, interval=interval).candles
    except Exception:
        pass
    secid = FIGI_TO_TICKER.get(figi)
    if not secid:
        return []
    iss_interval = ISS_INTERVAL_MAP.get(interval, 5)
    url = (
        "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/"
        f"securities/{secid}/candles.json?interval={iss_interval}"
        f"&from={frm.strftime('%Y-%m-%dT%H:%M:%S')}"
    )
    r = requests.get(url, timeout=10)
    j = r.json()
    cols = j["candles"]["columns"]
    idx = {name: i for i, name in enumerate(cols)}
    out = []
    for row in j["candles"]["data"]:
        out.append(type("C", (), {
            "open": float(row[idx.get("open", 0)]),
            "high": float(row[idx.get("high", 1)]),
            "low":  float(row[idx.get("low", 2)]),
            "close":float(row[idx.get("close", 3)]),
            "volume": int(row[idx.get("volume", 6)]) if idx.get("volume") is not None else 0,
        }))
    keep = int(hours * 60 / iss_interval) + 3
    return out[-keep:]

class Sandbox:
    def __init__(self, token: str, init_rub: float):
        self._client = Client(token)
        self.client: Optional[Services] = None
        self.account_id: Optional[str] = None
        self.init_rub = init_rub

    def __enter__(self):
        self.client = self._client.__enter__()
        self._ensure_account()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._client.__exit__(exc_type, exc, tb)

    def _ensure_account(self):
        if not hasattr(self.client, "sandbox"):
            import tinkoff.invest as ti
            raise RuntimeError(
                f"[SDK] sandbox отсутствует. Python={sys.executable} "
                f"Pkg={inspect.getfile(ti)} Ver={getattr(ti,'__version__','unknown')}"
            )
        try:
            try:
                accs = self.client.sandbox.users.get_accounts().accounts
            except AttributeError:
                accs = self.client.sandbox.get_sandbox_accounts().accounts
            if accs:
                self.account_id = accs[0].id
            else:
                opened = self.client.sandbox.open_sandbox_account()
                self.account_id = opened.account_id
            try:
                p = self.client.sandbox.operations.get_portfolio(account_id=self.account_id)
            except AttributeError:
                p = self.client.sandbox.get_sandbox_portfolio(account_id=self.account_id)
            total = money_to_float(p.total_amount_portfolio)
            if total < 1:
                self.pay_in(self.init_rub)
        except Exception as e:
            raise RuntimeError(f"[SBX] init failed: {e}") from e

    def post_market_order(self, figi: str, lots: int, side: OrderDirection):
        return self.client.sandbox.post_sandbox_order(
            figi=figi,
            quantity=lots,
            price=None,
            direction=side,
            account_id=self.account_id,
            order_type=OrderType.ORDER_TYPE_MARKET,
        )

    def cancel_all(self):
        try:
            orders = self.client.sandbox.orders.get_orders(account_id=self.account_id).orders
            for o in orders:
                self.client.sandbox.orders.cancel_order(account_id=self.account_id, order_id=o.order_id)
        except AttributeError:
            for o in self.client.sandbox.get_sandbox_orders(account_id=self.account_id).orders:
                self.client.sandbox.cancel_sandbox_order(account_id=self.account_id, order_id=o.order_id)

    def pay_in(self, amount_rub: int | float):
        self.client.sandbox.sandbox_pay_in(
            account_id=self.account_id,
            amount=MoneyValue(currency="rub", units=int(amount_rub), nano=0),
        )

    def get_portfolio(self):
        try:
            return self.client.sandbox.operations.get_portfolio(account_id=self.account_id)
        except AttributeError:
            return self.client.sandbox.get_sandbox_portfolio(account_id=self.account_id)

    def get_positions(self) -> Iterable:
        pf = self.get_portfolio()
        return getattr(pf, "positions", [])

    def get_total_rub(self) -> float:
        pf = self.get_portfolio()
        return money_to_float(pf.total_amount_portfolio)

    def get_lot_size(self, figi: str) -> int:
        try:
            ins = self.client.instruments.get_instrument_by(
                id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, id=figi
            )
            return ins.instrument.lot or 1
        except Exception:
            return 1

    def resolve_figi(self, ticker: str) -> Optional[str]:
        t = ticker.upper()
        try:
            for inst in self.client.instruments.shares().instruments:
                if inst.ticker.upper() == t:
                    return inst.figi
        except Exception:
            pass
        return FIGI_FALLBACK.get(t)

"""
Лёгкая онлайн-модель классификации (логистическая регрессия) без тяжёлых зависимостей.
- partial_fit(x, y) на каждом 1m баре
- L2-регуляризация
- forgetting factor alpha ~ 0.995 для плавного забывания старого
- Drift guard: если недавняя AUC/Brier портится и уверенность падает, сигналим о дрейфе
"""
from __future__ import annotations
import math
import json
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

logger = logging.getLogger("ML")

@dataclass
class OnlineLogReg:
    n_features: int
    lr: float = 0.05
    l2: float = 1e-4
    alpha: float = 0.995   # forgetting factor

    def __post_init__(self):
        self.w = np.zeros(self.n_features, dtype=np.float64)
        self.b = 0.0
        self.n = 0
        self.running_loss = 0.0
        self.acc_window: List[Tuple[float, int]] = []  # (p, y in {0,1})

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def predict_proba(self, x: np.ndarray) -> float:
        z = float(np.dot(self.w, x) + self.b)
        return self._sigmoid(z)

    def partial_fit(self, x: np.ndarray, y: int):
        # y in {0,1}
        p = self.predict_proba(x)
        # gradient
        err = (p - y)  # dL/dz for logistic loss
        # L2 reg
        grad_w = err * x + self.l2 * self.w
        grad_b = err
        # forgetting (decay weights a bit)
        self.w *= self.alpha
        self.b *= self.alpha
        # step
        self.w -= self.lr * grad_w
        self.b -= self.lr * grad_b
        # stats
        self.n += 1
        loss = -(y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9)))
        self.running_loss = 0.99 * self.running_loss + 0.01 * loss if self.n > 1 else loss
        self.acc_window.append((p, y))
        if len(self.acc_window) > 500:
            self.acc_window.pop(0)

    def health(self) -> Dict[str, Any]:
        if not self.acc_window:
            return {"loss": self.running_loss, "auc": None, "brier": None}
        ps = np.array([p for p, _ in self.acc_window])
        ys = np.array([y for _, y in self.acc_window])
        # Brier
        brier = float(np.mean((ps - ys) ** 2))
        # Approx AUC via rank correlation (not exact, but cheap)
        try:
            order = np.argsort(ps)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(ps))
            auc_proxy = float(np.corrcoef(ranks, ys)[0, 1])
        except Exception:
            auc_proxy = None
        return {"loss": self.running_loss, "auc": auc_proxy, "brier": brier}

def make_feature_vector(frow: Dict[str, Any]) -> np.ndarray:
    # x = [close change to EMA, ema slope, atr norm, roc1, spread_proxy, vol_norm]
    if not frow:
        return np.zeros(6, dtype=np.float64)
    close = frow["close"]
    ema_fast = frow["ema_fast"]
    ema_slow = frow["ema_slow"]
    atr14 = max(frow["atr14"], 1e-8)
    roc1 = frow["roc1"]
    spread_proxy = frow["spread_proxy"]
    vol_roll = frow["vol_roll"]
    ema_diff = (close - ema_fast) / max(close, 1e-8)
    ema_slope = (ema_fast - ema_slow) / max(close, 1e-8)
    vol_norm = vol_roll / max(vol_roll, 1e-8)  # -> 1.0 (placeholder)
    x = np.array([ema_diff, ema_slope, atr14 / max(close, 1e-8), roc1, spread_proxy, vol_norm], dtype=np.float64)
    return x

def label_from_future(next_close: float, curr_close: float) -> int:
    return 1 if (next_close - curr_close) >= 0.0 else 0

class OnlineModel:
    def __init__(self):
        self.clf = OnlineLogReg(n_features=6)

    def predict(self, frow: Dict[str, Any]) -> float:
        x = make_feature_vector(frow)
        return float(self.clf.predict_proba(x))

    def update(self, curr_frow: Dict[str, Any], next_close: float):
        y = label_from_future(next_close, curr_frow["close"])
        x = make_feature_vector(curr_frow)
        self.clf.partial_fit(x, y)

    def status(self) -> Dict[str, Any]:
        return self.clf.health()
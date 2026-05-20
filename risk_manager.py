"""Layer 9: Risk Management Engine — position sizing, volatility, danger warnings."""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class RiskAssessment:
    recommended_risk_percent: float
    adjusted_sl_points: float
    max_position_size_lots: float
    danger_level: int
    risk_warnings: list[str]
    atr: float
    atr_pct: float
    volatility_regime: str
    spread_normal: bool
    pre_news_detected: bool
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    val = tr.rolling(period).mean().iloc[-1]
    return float(val) if not pd.isna(val) else 0.0


def _volatility_regime(atr_pct: float) -> str:
    if atr_pct < 0.5:
        return "low"
    if atr_pct > 1.2:
        return "high"
    return "normal"


def _detect_whipsaw(df: pd.DataFrame, lookback: int = 10) -> bool:
    if len(df) < lookback:
        return False
    recent = df.tail(lookback)
    direction_changes = (recent["close"].diff().apply(lambda x: 1 if x > 0 else -1).diff() != 0).sum()
    return bool(direction_changes > lookback * 0.7)


class RiskManager:
    def __init__(self, account_balance: float = 10000.0, base_risk_pct: float = 1.0):
        self.account_balance = account_balance
        self.base_risk_pct = base_risk_pct
        self._recent_trades: list[dict] = []

    def record_trade(self, trade: dict) -> None:
        self._recent_trades.append(trade)
        if len(self._recent_trades) > 20:
            self._recent_trades.pop(0)

    def assess_trade_risk(
        self,
        df: pd.DataFrame,
        trade_setup: Optional[dict] = None,
        high_impact_news_in_2h: bool = False,
        current_spread: float = 0.0,
        normal_spread: float = 0.3,
    ) -> dict:
        warnings: list[str] = []
        danger = 0

        atr_val = _atr(df)
        price = float(df["close"].iloc[-1]) if not df.empty else 2000.0
        atr_pct = atr_val / (price + 1e-9) * 100
        vol_regime = _volatility_regime(atr_pct)

        normal_atr_pct = 0.6
        vol_scaler = normal_atr_pct / (atr_pct + 1e-9)
        vol_scaler = max(0.5, min(2.0, vol_scaler))

        risk_pct = self.base_risk_pct * vol_scaler

        if high_impact_news_in_2h:
            risk_pct *= 0.5
            warnings.append("High-impact news within 2 hours — position size halved")
            danger += 2

        spread_ratio = current_spread / (normal_spread + 1e-9)
        spread_ok = spread_ratio < 2.0
        if not spread_ok:
            warnings.append(f"Spread {current_spread:.2f} is {spread_ratio:.1f}x normal — low liquidity")
            danger += 3
            risk_pct *= 0.5

        pre_news = _detect_whipsaw(df)
        if pre_news:
            warnings.append("Price whipsawing — possible pre-news manipulation. Avoid entries")
            danger += 2

        recent_count = len(self._recent_trades)
        if recent_count > 3:
            warnings.append(f"Overtrading alert: {recent_count} trades recently — take a break")
            danger += 2

        if vol_regime == "high":
            warnings.append(f"High volatility (ATR {atr_pct:.2f}%) — reduce size and widen SL")
            danger += 1
        elif vol_regime == "low":
            warnings.append(f"Low volatility (ATR {atr_pct:.2f}%) — widen SL to avoid false stops")

        sl_multiplier = 2.5 if vol_regime == "low" else 1.5
        sl_points = round(atr_val * sl_multiplier, 2)

        risk_amount = self.account_balance * (risk_pct / 100)
        lots = round(risk_amount / (sl_points * 100), 2) if sl_points > 0 else 0.01

        danger = min(10, max(1, danger))

        return RiskAssessment(
            recommended_risk_percent=round(min(risk_pct, 2.0), 2),
            adjusted_sl_points=sl_points,
            max_position_size_lots=lots,
            danger_level=danger,
            risk_warnings=warnings,
            atr=round(atr_val, 2),
            atr_pct=round(atr_pct, 3),
            volatility_regime=vol_regime,
            spread_normal=spread_ok,
            pre_news_detected=pre_news,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ).to_dict()


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from market_data_agent import get_historical_data
    df = get_historical_data("XAUUSD", period="30d", interval="1h")
    manager = RiskManager(account_balance=10000.0, base_risk_pct=1.0)
    print(json.dumps(manager.assess_trade_risk(df), indent=2, default=str))

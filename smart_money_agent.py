"""Layer 7: Liquidity & Smart Money Detector — identifies institutional footprints."""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

# ── Round numbers for XAUUSD ──────────────────────────────────────────────────
ROUND_NUMBERS = [n for n in range(1600, 3500, 50)]  # Every $50

@dataclass
class LiquidityZone:
    price: float
    zone_type: str    # equal_high / equal_low / round_number / prev_week / prev_day / fib
    magnitude: str    # high / medium / low
    swept: bool       # Has price already taken this liquidity?

@dataclass
class SmartMoneySignal:
    signal_type: str  # absorption / wick_rejection / impulse / bos_retest / stop_hunt / fvg_fill
    price: float
    timestamp: str
    confidence: float  # 0-1
    description: str

@dataclass
class SmartMoneySummary:
    liquidity_zones: list[LiquidityZone]
    sm_phase: str               # accumulation / markup / distribution / markdown
    trap_warnings: list[str]
    engineered_liquidity_score: float  # 0-100
    signals: list[SmartMoneySignal]
    vwap: Optional[float]
    price_vs_vwap: str          # premium / discount / fair_value
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)

# ── Liquidity detection ───────────────────────────────────────────────────────
def _find_equal_highs_lows(df: pd.DataFrame, lookback: int = 30, tolerance_pct: float = 0.001) -> list[LiquidityZone]:
    zones = []
    recent = df.tail(lookback * 24)  # 30 days of hourly data
    highs = recent["high"].values
    lows = recent["low"].values

    for i, h in enumerate(highs):
        cluster = [h2 for h2 in highs[i+1:] if abs(h2 - h) / (h + 1e-9) < tolerance_pct]
        if len(cluster) >= 2:
            zones.append(LiquidityZone(
                price=round(float(h), 2),
                zone_type="equal_high",
                magnitude="high" if len(cluster) >= 3 else "medium",
                swept=float(df["close"].iloc[-1]) > h,
            ))

    for i, l in enumerate(lows):
        cluster = [l2 for l2 in lows[i+1:] if abs(l2 - l) / (l + 1e-9) < tolerance_pct]
        if len(cluster) >= 2:
            zones.append(LiquidityZone(
                price=round(float(l), 2),
                zone_type="equal_low",
                magnitude="high" if len(cluster) >= 3 else "medium",
                swept=float(df["close"].iloc[-1]) < l,
            ))

    return zones[:20]

def _find_round_number_zones(price: float, radius: float = 50) -> list[LiquidityZone]:
    zones = []
    for rn in ROUND_NUMBERS:
        if abs(rn - price) <= radius * 3:
            dist = abs(rn - price)
            mag = "high" if rn % 100 == 0 else "medium" if rn % 50 == 0 else "low"
            zones.append(LiquidityZone(
                price=float(rn),
                zone_type="round_number",
                magnitude=mag,
                swept=False,
            ))
    return sorted(zones, key=lambda z: abs(z.price - price))[:5]

def _prev_high_low(df: pd.DataFrame) -> list[LiquidityZone]:
    zones = []
    if len(df) < 48:
        return zones

    # Previous day
    daily = df.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    if len(daily) >= 2:
        prev_d = daily.iloc[-2]
        cur_price = float(df["close"].iloc[-1])
        zones.append(LiquidityZone(
            price=float(prev_d["high"]),
            zone_type="prev_day",
            magnitude="high",
            swept=cur_price > float(prev_d["high"]),
        ))
        zones.append(LiquidityZone(
            price=float(prev_d["low"]),
            zone_type="prev_day",
            magnitude="high",
            swept=cur_price < float(prev_d["low"]),
        ))

    # Previous week
    weekly = df.resample("1W").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    if len(weekly) >= 2:
        prev_w = weekly.iloc[-2]
        zones.append(LiquidityZone(
            price=float(prev_w["high"]),
            zone_type="prev_week",
            magnitude="high",
            swept=cur_price > float(prev_w["high"]),
        ))
        zones.append(LiquidityZone(
            price=float(prev_w["low"]),
            zone_type="prev_week",
            magnitude="high",
            swept=cur_price < float(prev_w["low"]),
        ))

    return zones

def _find_liquidity_zones(df: pd.DataFrame) -> list[LiquidityZone]:
    price = float(df["close"].iloc[-1])
    zones = (
        _find_equal_highs_lows(df) +
        _find_round_number_zones(price) +
        _prev_high_low(df)
    )
    # Deduplicate
    seen: list[float] = []
    unique = []
    for z in zones:
        if not any(abs(z.price - s) / (z.price + 1e-9) < 0.002 for s in seen):
            seen.append(z.price)
            unique.append(z)
    return sorted(unique, key=lambda z: abs(z.price - price))[:20]

# ── Signal detection ──────────────────────────────────────────────────────────
def _detect_signals(df: pd.DataFrame) -> list[SmartMoneySignal]:
    signals = []
    if len(df) < 10:
        return signals

    recent = df.tail(50)
    for i in range(2, len(recent) - 1):
        candle = recent.iloc[i]
        prev = recent.iloc[i - 1]

        # Wick rejection
        body = abs(candle["close"] - candle["open"])
        upper_wick = candle["high"] - max(candle["close"], candle["open"])
        lower_wick = min(candle["close"], candle["open"]) - candle["low"]
        if upper_wick > body * 2 and upper_wick > 0:
            signals.append(SmartMoneySignal(
                signal_type="wick_rejection",
                price=float(candle["high"]),
                timestamp=str(recent.index[i]),
                confidence=0.7,
                description="Large upper wick — institutional rejection at high",
            ))
        if lower_wick > body * 2 and lower_wick > 0:
            signals.append(SmartMoneySignal(
                signal_type="wick_rejection",
                price=float(candle["low"]),
                timestamp=str(recent.index[i]),
                confidence=0.7,
                description="Large lower wick — institutional support / stop hunt reversal",
            ))

    # Volume spike (absorption)
    if "volume" in df.columns and df["volume"].mean() > 0:
        vol_mean = df["volume"].rolling(20).mean()
        vol_spike = df["volume"] > vol_mean * 2.5
        price_change = df["close"].pct_change().abs() < 0.002
        absorption_bars = vol_spike & price_change
        if absorption_bars.iloc[-5:].any():
            signals.append(SmartMoneySignal(
                signal_type="absorption",
                price=float(df["close"].iloc[-1]),
                timestamp=str(df.index[-1]),
                confidence=0.8,
                description="High volume + small price change — institutional absorption",
            ))

    return signals[-10:]

def _vwap(df: pd.DataFrame) -> Optional[float]:
    if "volume" not in df.columns or df["volume"].sum() == 0:
        return None
    typical = (df["high"] + df["low"] + df["close"]) / 3
    return round(float((typical * df["volume"]).sum() / df["volume"].sum()), 2)

def _sm_phase(df: pd.DataFrame, signals: list[SmartMoneySignal]) -> str:
    if len(df) < 20:
        return "unknown"
    slope = float(np.polyfit(range(20), df["close"].iloc[-20:].values, 1)[0])
    absorption = any(s.signal_type == "absorption" for s in signals)
    if slope < -0.5 and absorption:
        return "accumulation"
    if slope > 0.5:
        return "markup"
    if slope > 0.2 and not absorption:
        return "distribution"
    if slope < -0.5:
        return "markdown"
    return "accumulation"

def _engineered_liquidity_score(zones: list[LiquidityZone]) -> float:
    if not zones:
        return 0.0
    weights = {"high": 3, "medium": 2, "low": 1}
    score = sum(weights.get(z.magnitude, 1) * (1 if not z.swept else 0.3) for z in zones[:10])
    return round(min(100.0, score * 5), 1)

# ── Main class ────────────────────────────────────────────────────────────────
class SmartMoneyAnalyzer:
    def find_liquidity(self, df: pd.DataFrame) -> list[dict]:
        return [asdict(z) for z in _find_liquidity_zones(df)]

    def detect_stop_hunt(self, df: pd.DataFrame) -> list[dict]:
        signals = _detect_signals(df)
        return [asdict(s) for s in signals if s.signal_type in ("wick_rejection", "absorption")]

    def analyze(self, df: pd.DataFrame) -> dict:
        zones = _find_liquidity_zones(df)
        signals = _detect_signals(df)
        vwap_price = _vwap(df)
        current_price = float(df["close"].iloc[-1])

        price_vs_vwap = "fair_value"
        if vwap_price:
            diff_pct = (current_price - vwap_price) / (vwap_price + 1e-9) * 100
            if diff_pct > 0.5:
                price_vs_vwap = "premium"
            elif diff_pct < -0.5:
                price_vs_vwap = "discount"

        phase = _sm_phase(df, signals)
        eng_score = _engineered_liquidity_score(zones)

        trap_warnings = []
        nearby_highs = [z for z in zones if z.zone_type == "equal_high" and
                        0 < (z.price - current_price) / (current_price + 1e-9) < 0.005]
        nearby_lows = [z for z in zones if z.zone_type == "equal_low" and
                       0 < (current_price - z.price) / (current_price + 1e-9) < 0.005]
        if nearby_highs:
            trap_warnings.append(f"Equal highs at {nearby_highs[0].price:.2f} — retail longs likely trapped above")
        if nearby_lows:
            trap_warnings.append(f"Equal lows at {nearby_lows[0].price:.2f} — retail shorts likely trapped below")
        if price_vs_vwap == "premium" and phase in ("distribution", "markdown"):
            trap_warnings.append("Price at premium + distribution phase — long trap risk")

        summary = SmartMoneySummary(
            liquidity_zones=zones,
            sm_phase=phase,
            trap_warnings=trap_warnings,
            engineered_liquidity_score=eng_score,
            signals=signals,
            vwap=vwap_price,
            price_vs_vwap=price_vs_vwap,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        return summary.to_dict()


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from market_data_agent import get_historical_data
    df = get_historical_data("XAUUSD", period="30d", interval="1h")
    analyzer = SmartMoneyAnalyzer()
    print(json.dumps(analyzer.analyze(df), indent=2, default=str))

"""Layer 2: Technical Analysis Engine — indicators, structure, S/R across all timeframes."""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class TrendData:
    ema9: float
    ema20: float
    ema50: float
    ema200: float
    adx: float
    adx_strength: str
    slope: float
    trend_direction: str


@dataclass
class StructureData:
    swing_highs: list[float]
    swing_lows: list[float]
    bos_detected: bool
    choch_detected: bool
    current_structure: str


@dataclass
class SRData:
    horizontal_levels: list[float]
    order_blocks: list[dict]
    fair_value_gaps: list[dict]
    liquidity_zones: list[float]


@dataclass
class MomentumData:
    rsi: float
    rsi_signal: str
    rsi_divergence: str
    macd: float
    macd_signal: float
    macd_histogram: float
    macd_cross: str
    stoch_k: float
    stoch_d: float
    stoch_divergence: str


@dataclass
class VolatilityData:
    atr: float
    atr_pct: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_pct_b: float
    bb_bandwidth: float
    bb_squeeze: bool


@dataclass
class TimeframeAnalysis:
    timeframe: str
    trend: TrendData
    structure: StructureData
    support_resistance: SRData
    momentum: MomentumData
    volatility: VolatilityData
    bias: str
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = _atr(df, period)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    return dx.rolling(period).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    fast_ema = _ema(series, fast)
    slow_ema = _ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _stochastic(df: pd.DataFrame, k_period=14, d_period=3):
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-9)
    d = k.rolling(d_period).mean()
    return k, d


def _bollinger(series: pd.Series, period=20, std_dev=2.0):
    middle = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    pct_b = (series - lower) / (upper - lower + 1e-9)
    bandwidth = (upper - lower) / (middle + 1e-9)
    return upper, middle, lower, pct_b, bandwidth


def _swing_points(df: pd.DataFrame, lookback: int = 5) -> tuple[list, list]:
    highs, lows = [], []
    for i in range(lookback, len(df) - lookback):
        window_high = df["high"].iloc[i - lookback: i + lookback + 1]
        window_low = df["low"].iloc[i - lookback: i + lookback + 1]
        if df["high"].iloc[i] == window_high.max():
            highs.append(float(df["high"].iloc[i]))
        if df["low"].iloc[i] == window_low.min():
            lows.append(float(df["low"].iloc[i]))
    return highs[-10:], lows[-10:]


def _detect_fvg(df: pd.DataFrame) -> list[dict]:
    fvgs = []
    for i in range(2, len(df)):
        c1_high = df["high"].iloc[i - 2]
        c3_low = df["low"].iloc[i]
        if c3_low > c1_high:
            fvgs.append({"top": float(c3_low), "bottom": float(c1_high), "direction": "bullish"})
        c1_low = df["low"].iloc[i - 2]
        c3_high = df["high"].iloc[i]
        if c3_high < c1_low:
            fvgs.append({"top": float(c1_low), "bottom": float(c3_high), "direction": "bearish"})
    return fvgs[-5:]


def _detect_order_blocks(df: pd.DataFrame, swing_highs: list, swing_lows: list) -> list[dict]:
    blocks = []
    for h in swing_highs[-3:]:
        idx = df["high"].sub(h).abs().idxmin()
        pos = df.index.get_loc(idx)
        if pos > 0:
            candle = df.iloc[pos - 1]
            blocks.append({
                "price": float(h), "type": "bearish", "strength": "high",
                "zone_top": float(candle["high"]), "zone_bottom": float(candle["low"]),
            })
    for l in swing_lows[-3:]:
        idx = df["low"].sub(l).abs().idxmin()
        pos = df.index.get_loc(idx)
        if pos > 0:
            candle = df.iloc[pos - 1]
            blocks.append({
                "price": float(l), "type": "bullish", "strength": "high",
                "zone_top": float(candle["high"]), "zone_bottom": float(candle["low"]),
            })
    return blocks


def _detect_bos_choch(df: pd.DataFrame, swing_highs: list, swing_lows: list) -> tuple[bool, bool, str]:
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return False, False, "consolidation"
    last_close = df["close"].iloc[-1]
    prev_high = swing_highs[-2]
    prev_low = swing_lows[-2]
    latest_high = swing_highs[-1]
    latest_low = swing_lows[-1]
    structure = "bullish" if latest_high > prev_high and latest_low > prev_low else \
                "bearish" if latest_high < prev_high and latest_low < prev_low else "consolidation"
    bos = last_close > prev_high or last_close < prev_low
    choch = (structure == "bullish" and last_close < prev_low) or \
            (structure == "bearish" and last_close > prev_high)
    return bos, choch, structure


def _rsi_divergence(df: pd.DataFrame, rsi: pd.Series, lookback: int = 20) -> str:
    price_slice = df["close"].iloc[-lookback:]
    rsi_slice = rsi.iloc[-lookback:].dropna()
    if len(rsi_slice) < 5:
        return "none"
    price_low_idx = price_slice.idxmin()
    rsi_low_idx = rsi_slice.idxmin()
    if price_low_idx != rsi_low_idx:
        if price_slice[price_low_idx] < price_slice.iloc[-1] and \
           rsi_slice.loc[rsi_low_idx] > rsi_slice.iloc[-1]:
            return "bullish"
    price_high_idx = price_slice.idxmax()
    rsi_high_idx = rsi_slice.idxmax()
    if price_high_idx != rsi_high_idx:
        if price_slice[price_high_idx] > price_slice.iloc[-1] and \
           rsi_slice.loc[rsi_high_idx] < rsi_slice.iloc[-1]:
            return "bearish"
    return "none"


class TechnicalAnalyzer:
    def analyze(self, df: pd.DataFrame, timeframe: str = "Unknown") -> TimeframeAnalysis:
        if len(df) < 50:
            raise ValueError(f"Insufficient data for {timeframe}: need ≥50 bars, got {len(df)}")

        close = df["close"]

        e9, e20, e50, e200 = _ema(close, 9), _ema(close, 20), _ema(close, 50), _ema(close, 200)
        adx_series = _adx(df)
        adx_val = float(adx_series.iloc[-1]) if not adx_series.iloc[-1:].isna().any() else 0.0
        adx_strength = "weak" if adx_val < 20 else "strong" if adx_val < 40 else "extreme"

        y = close.iloc[-20:].values
        x = np.arange(len(y))
        slope = float(np.polyfit(x, y, 1)[0])

        last_c = float(close.iloc[-1])
        e9v, e20v, e50v, e200v = float(e9.iloc[-1]), float(e20.iloc[-1]), float(e50.iloc[-1]), float(e200.iloc[-1])
        trend_dir = "bullish" if last_c > e50v and e20v > e50v else \
                    "bearish" if last_c < e50v and e20v < e50v else "sideways"

        trend = TrendData(
            ema9=e9v, ema20=e20v, ema50=e50v, ema200=e200v,
            adx=adx_val, adx_strength=adx_strength,
            slope=round(slope, 4), trend_direction=trend_dir,
        )

        lookback = min(5, len(df) // 10)
        swing_highs, swing_lows = _swing_points(df, lookback=lookback)
        bos, choch, structure = _detect_bos_choch(df, swing_highs, swing_lows)

        struct = StructureData(
            swing_highs=swing_highs, swing_lows=swing_lows,
            bos_detected=bos, choch_detected=choch,
            current_structure=structure,
        )

        fvgs = _detect_fvg(df)
        order_blocks = _detect_order_blocks(df, swing_highs, swing_lows)
        all_extremes = swing_highs + swing_lows
        liq_zones: list[float] = []
        for p in all_extremes:
            if not any(abs(z - p) / (p + 1e-9) < 0.001 for z in liq_zones):
                liq_zones.append(round(p, 2))

        sr = SRData(
            horizontal_levels=sorted(set(round(p, 1) for p in all_extremes))[-10:],
            order_blocks=order_blocks,
            fair_value_gaps=fvgs,
            liquidity_zones=sorted(liq_zones)[-15:],
        )

        rsi_series = _rsi(close)
        rsi_val = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
        rsi_signal = "overbought" if rsi_val > 70 else "oversold" if rsi_val < 30 else "neutral"
        rsi_div = _rsi_divergence(df, rsi_series)

        macd_line, signal_line, histogram = _macd(close)
        macd_v = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0
        signal_v = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0
        hist_v = float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0
        prev_hist = float(histogram.iloc[-2]) if len(histogram) > 1 and not pd.isna(histogram.iloc[-2]) else 0.0
        macd_cross = "bullish" if macd_v > signal_v and hist_v > prev_hist else \
                     "bearish" if macd_v < signal_v and hist_v < prev_hist else "none"

        k, d = _stochastic(df)
        stoch_k_v = float(k.iloc[-1]) if not pd.isna(k.iloc[-1]) else 50.0
        stoch_d_v = float(d.iloc[-1]) if not pd.isna(d.iloc[-1]) else 50.0
        stoch_div = "bullish" if stoch_k_v < 20 and stoch_k_v > stoch_d_v else \
                    "bearish" if stoch_k_v > 80 and stoch_k_v < stoch_d_v else "none"

        momentum = MomentumData(
            rsi=round(rsi_val, 2), rsi_signal=rsi_signal, rsi_divergence=rsi_div,
            macd=round(macd_v, 4), macd_signal=round(signal_v, 4),
            macd_histogram=round(hist_v, 4), macd_cross=macd_cross,
            stoch_k=round(stoch_k_v, 2), stoch_d=round(stoch_d_v, 2),
            stoch_divergence=stoch_div,
        )

        atr_series = _atr(df)
        atr_v = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0.0
        atr_pct = atr_v / (last_c + 1e-9) * 100

        bb_up, bb_mid, bb_low, pct_b, bandwidth = _bollinger(close)
        bb_up_v = float(bb_up.iloc[-1]) if not pd.isna(bb_up.iloc[-1]) else last_c
        bb_mid_v = float(bb_mid.iloc[-1]) if not pd.isna(bb_mid.iloc[-1]) else last_c
        bb_low_v = float(bb_low.iloc[-1]) if not pd.isna(bb_low.iloc[-1]) else last_c
        pct_b_v = float(pct_b.iloc[-1]) if not pd.isna(pct_b.iloc[-1]) else 0.5
        bw_v = float(bandwidth.iloc[-1]) if not pd.isna(bandwidth.iloc[-1]) else 0.0
        bw_avg = float(bandwidth.rolling(50).mean().iloc[-1]) if len(bandwidth) >= 50 else bw_v
        squeeze = bw_v < bw_avg * 0.8

        volatility = VolatilityData(
            atr=round(atr_v, 2), atr_pct=round(atr_pct, 3),
            bb_upper=round(bb_up_v, 2), bb_middle=round(bb_mid_v, 2), bb_lower=round(bb_low_v, 2),
            bb_pct_b=round(pct_b_v, 3), bb_bandwidth=round(bw_v, 4), bb_squeeze=squeeze,
        )

        bull_signals = sum([
            trend_dir == "bullish",
            rsi_val > 50,
            macd_cross == "bullish",
            structure == "bullish",
            adx_val > 20 and slope > 0,
        ])
        bear_signals = sum([
            trend_dir == "bearish",
            rsi_val < 50,
            macd_cross == "bearish",
            structure == "bearish",
            adx_val > 20 and slope < 0,
        ])
        bias = "bullish" if bull_signals > bear_signals else \
               "bearish" if bear_signals > bull_signals else "neutral"
        confidence = max(bull_signals, bear_signals) / 5.0

        return TimeframeAnalysis(
            timeframe=timeframe, trend=trend, structure=struct,
            support_resistance=sr, momentum=momentum, volatility=volatility,
            bias=bias, confidence=round(confidence, 2),
        )


def analyze_all_timeframes(tf_data: dict[str, pd.DataFrame]) -> dict[str, dict]:
    analyzer = TechnicalAnalyzer()
    results: dict[str, dict] = {}
    for label, df in tf_data.items():
        try:
            analysis = analyzer.analyze(df, timeframe=label)
            results[label] = analysis.to_dict()
        except Exception as e:
            logger.warning(f"Technical analysis failed for {label}: {e}")
            results[label] = {"error": str(e), "timeframe": label}
    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from market_data_agent import get_multi_timeframe_data

    tf_data = get_multi_timeframe_data("XAUUSD")
    results = analyze_all_timeframes(tf_data)
    print(json.dumps({k: v for k, v in results.items()}, indent=2, default=str))

"""Layer 6: Correlation & Cross-Asset Analyzer — tracks gold correlations and divergences."""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from loguru import logger

EXPECTED_CORRELATIONS = {
    "DXY":    -0.85,
    "US10Y":  -0.70,
    "EURUSD": +0.75,
    "XAGUSD": +0.90,
    "SPX":    +0.20,
    "VIX":    +0.40,
    "BTCUSD": +0.35,
    "WTI":    +0.30,
    "USDJPY": -0.50,
    "GBPUSD": +0.50,
}


@dataclass
class CorrelationResult:
    symbol: str
    correlation_1h: Optional[float]
    correlation_4h: Optional[float]
    correlation_1d: Optional[float]
    expected_correlation: float
    divergence_detected: bool
    divergence_type: str
    gold_expected_move_per_1pct: float
    current_influence_rank: int


@dataclass
class CorrelationSummary:
    correlations: list[CorrelationResult]
    correlation_matrix: dict[str, float]
    strongest_influence: str
    divergence_warnings: list[str]
    regime: str
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


def _rolling_corr(s1: pd.Series, s2: pd.Series, window: int) -> Optional[float]:
    if len(s1) < window or len(s2) < window:
        return None
    aligned = pd.concat([s1, s2], axis=1).dropna()
    if len(aligned) < window:
        return None
    corr = aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1]).iloc[-1]
    return round(float(corr), 3) if not pd.isna(corr) else None


def _detect_divergence(actual: Optional[float], expected: float) -> tuple[bool, str]:
    if actual is None:
        return False, "na"
    if abs(actual - expected) < 0.3:
        return False, "normal"
    if actual * expected < 0:
        return True, "inverted"
    if abs(actual) < 0.2 and abs(expected) > 0.5:
        return True, "breakdown"
    return False, "normal"


SENSITIVITIES = {
    "DXY":    -15.0,
    "US10Y":  -8.0,
    "EURUSD": +10.0,
    "XAGUSD": +25.0,
    "SPX":    +3.0,
    "VIX":    +5.0,
    "BTCUSD": +2.0,
    "WTI":    +3.0,
    "USDJPY": -6.0,
    "GBPUSD": +7.0,
}


def _regime(corr_results: list[CorrelationResult]) -> str:
    vix_corr = next((r for r in corr_results if r.symbol == "VIX"), None)
    spx_corr = next((r for r in corr_results if r.symbol == "SPX"), None)
    if vix_corr and vix_corr.correlation_1d and vix_corr.correlation_1d > 0.5:
        return "risk_off"
    if spx_corr and spx_corr.correlation_1d and spx_corr.correlation_1d > 0.5:
        return "risk_on"
    return "mixed"


class CorrelationAnalyzer:
    def __init__(self):
        self._data_cache: dict[str, pd.Series] = {}

    def update_data(self, gold_df: pd.DataFrame, correlated_data: dict[str, pd.DataFrame]) -> None:
        self._data_cache["XAUUSD"] = gold_df["close"].rename("XAUUSD")
        for sym, df in correlated_data.items():
            if not df.empty:
                self._data_cache[sym] = df["close"].rename(sym)

    def analyze(self) -> dict:
        gold = self._data_cache.get("XAUUSD")
        if gold is None or gold.empty:
            return {"error": "No XAUUSD data loaded"}

        results: list[CorrelationResult] = []
        corr_matrix: dict[str, float] = {}
        divergence_warnings: list[str] = []

        for sym, expected in EXPECTED_CORRELATIONS.items():
            series = self._data_cache.get(sym)
            if series is None or series.empty:
                results.append(CorrelationResult(
                    symbol=sym, correlation_1h=None, correlation_4h=None,
                    correlation_1d=None, expected_correlation=expected,
                    divergence_detected=False, divergence_type="na",
                    gold_expected_move_per_1pct=SENSITIVITIES.get(sym, 0.0),
                    current_influence_rank=99,
                ))
                continue

            c1h = _rolling_corr(gold, series, 60)
            c4h = _rolling_corr(gold, series, 240)
            c1d = _rolling_corr(gold, series, 1440)
            best_corr = c1h or c4h or c1d
            diverged, div_type = _detect_divergence(best_corr, expected)

            if diverged:
                divergence_warnings.append(
                    f"{sym}: Expected {expected:.2f}, actual {best_corr:.2f} — {div_type}"
                )

            corr_matrix[sym] = best_corr or 0.0
            results.append(CorrelationResult(
                symbol=sym, correlation_1h=c1h, correlation_4h=c4h, correlation_1d=c1d,
                expected_correlation=expected, divergence_detected=diverged, divergence_type=div_type,
                gold_expected_move_per_1pct=SENSITIVITIES.get(sym, 0.0),
                current_influence_rank=0,
            ))

        results.sort(key=lambda r: abs(r.correlation_1h or 0), reverse=True)
        for i, r in enumerate(results):
            r.current_influence_rank = i + 1

        strongest = results[0].symbol if results else "DXY"
        regime = _regime(results)

        dxy_r = next((r for r in results if r.symbol == "DXY"), None)
        if dxy_r and dxy_r.correlation_1h and dxy_r.correlation_1h > 0:
            divergence_warnings.insert(0, "CRITICAL: Gold rising WITH DXY — rare safe haven bid in effect")

        return CorrelationSummary(
            correlations=results,
            correlation_matrix=corr_matrix,
            strongest_influence=strongest,
            divergence_warnings=divergence_warnings[:5],
            regime=regime,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ).to_dict()


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from market_data_agent import get_historical_data
    gold_df = get_historical_data("XAUUSD", period="30d", interval="1h")
    correlated = {sym: get_historical_data(sym, period="30d", interval="1h")
                  for sym in ["DXY", "EURUSD", "VIX"]}
    analyzer = CorrelationAnalyzer()
    analyzer.update_data(gold_df, correlated)
    print(json.dumps(analyzer.analyze(), indent=2, default=str))

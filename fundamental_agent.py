"""Layer 3: Macro Fundamentals Agent — economic indicators impacting gold."""
from __future__ import annotations
import time
import json
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)


def _cache_get(key: str, ttl: int) -> Optional[dict]:
    p = CACHE_DIR / f"macro_{key}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d["v"] if time.time() - d.get("t", 0) < ttl else None


def _cache_set(key: str, value: dict) -> None:
    (CACHE_DIR / f"macro_{key}.json").write_text(json.dumps({"t": time.time(), "v": value}))


@dataclass
class MacroMetric:
    name: str
    current_value: Optional[float]
    previous_value: Optional[float]
    consensus_forecast: Optional[float]
    unit: str
    deviation_impact: str
    confidence_score: float
    short_term_gold_impact: str
    medium_term_gold_impact: str
    long_term_gold_impact: str


@dataclass
class MacroSummary:
    metrics: dict[str, MacroMetric]
    usd_strength_score: float
    inflation_regime: str
    real_rate_regime: str
    gold_macro_bias: float
    geopolitical_risk_index: float
    analysis_timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

FRED_SERIES = {
    "fed_funds_rate":    "FEDFUNDS",
    "cpi_yoy":           "CPIAUCSL",
    "core_cpi":          "CPILFESL",
    "ppi":               "PPIACO",
    "unemployment":      "UNRATE",
    "gdp":               "GDP",
    "yield_10y":         "DGS10",
    "yield_2y":          "DGS2",
    "real_yield_10y":    "DFII10",
    "fed_balance_sheet": "WALCL",
}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def _fred_fetch(series_id: str, api_key: str, limit: int = 2) -> Optional[tuple[float, float]]:
    if not api_key:
        return None
    try:
        r = requests.get(FRED_BASE, params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }, timeout=10)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        vals = [float(o["value"]) for o in obs if o.get("value") not in (".", "")]
        if len(vals) >= 2:
            return vals[0], vals[1]
        elif len(vals) == 1:
            return vals[0], vals[0]
        return None
    except Exception as e:
        logger.warning(f"FRED fetch failed for {series_id}: {e}")
        return None


MOCK_DATA = {
    "fed_funds_rate":    (5.33, 5.33),
    "cpi_yoy":           (3.2,  3.5),
    "core_cpi":          (3.8,  3.9),
    "ppi":               (2.2,  2.0),
    "unemployment":      (3.9,  3.7),
    "gdp":               (2.5,  3.3),
    "yield_10y":         (4.45, 4.30),
    "yield_2y":          (4.85, 4.90),
    "real_yield_10y":    (1.80, 1.70),
    "fed_balance_sheet": (7500, 7600),
}


def _rate_impact(current: float, previous: float) -> tuple[str, str]:
    if current > previous + 0.1:
        return "hawkish", "bearish"
    if current < previous - 0.1:
        return "dovish", "bullish"
    return "neutral", "neutral"


def _cpi_impact(current: float, previous: float) -> tuple[str, str]:
    if current > previous + 0.2:
        return "hawkish", "bearish"
    if current < previous - 0.2:
        return "dovish", "bullish"
    return "neutral", "neutral"


def _real_yield_impact(current: float) -> str:
    return "bearish" if current > 1.5 else "bullish" if current < 0 else "neutral"


def _yield_curve(yield_10y: float, yield_2y: float) -> str:
    spread = yield_10y - yield_2y
    if spread < -0.5:
        return "inverted"
    if spread < 0:
        return "flat"
    return "normal"


def _inflation_regime(cpi: float, core_cpi: float) -> str:
    avg = (cpi + core_cpi) / 2
    if avg < 1.0:
        return "deflation"
    if avg < 2.5:
        return "disinflation"
    if avg <= 3.5:
        return "perfect"
    return "overheating"


def _gold_macro_bias(
    fed_rate: float, real_yield: float, cpi: float,
    usd_score: float, regime: str
) -> float:
    score = 5.0
    if real_yield < 0:
        score += 2.5
    elif real_yield > 2.0:
        score -= 2.5
    if regime == "overheating":
        score += 1.5
    elif regime == "deflation":
        score -= 1.5
    score -= usd_score * 0.3
    if fed_rate > 5.0:
        score -= 1.0
    elif fed_rate < 2.0:
        score += 1.0
    return round(max(1.0, min(10.0, score)), 1)


class FundamentalAgent:
    def __init__(self, fred_api_key: str = ""):
        self.fred_api_key = fred_api_key

    def _get_metric(self, name: str) -> tuple[float, float]:
        cached = _cache_get(name, ttl=3600)
        if cached:
            return cached["current"], cached["previous"]
        if self.fred_api_key:
            result = _fred_fetch(FRED_SERIES[name], self.fred_api_key)
            if result:
                _cache_set(name, {"current": result[0], "previous": result[1]})
                return result
        return MOCK_DATA.get(name, (0.0, 0.0))

    def get_macro_summary(self, geopolitical_risk: float = 5.0) -> MacroSummary:
        from datetime import datetime, timezone
        data = {name: self._get_metric(name) for name in FRED_SERIES}
        metrics: dict[str, MacroMetric] = {}

        fed_cur, fed_prev = data["fed_funds_rate"]
        rate_dev, rate_gold = _rate_impact(fed_cur, fed_prev)
        metrics["fed_funds_rate"] = MacroMetric(
            name="Federal Funds Rate", current_value=fed_cur, previous_value=fed_prev,
            consensus_forecast=None, unit="%", deviation_impact=rate_dev, confidence_score=9.0,
            short_term_gold_impact=rate_gold, medium_term_gold_impact=rate_gold,
            long_term_gold_impact="neutral",
        )

        cpi_cur, cpi_prev = data["cpi_yoy"]
        cpi_dev, cpi_gold = _cpi_impact(cpi_cur, cpi_prev)
        metrics["cpi_yoy"] = MacroMetric(
            name="CPI YoY", current_value=cpi_cur, previous_value=cpi_prev,
            consensus_forecast=None, unit="%", deviation_impact=cpi_dev, confidence_score=8.5,
            short_term_gold_impact=cpi_gold,
            medium_term_gold_impact="bullish" if cpi_cur > 3 else "neutral",
            long_term_gold_impact="bullish" if cpi_cur > 4 else "neutral",
        )

        real_cur, real_prev = data["real_yield_10y"]
        real_impact = _real_yield_impact(real_cur)
        metrics["real_yield_10y"] = MacroMetric(
            name="Real 10Y Yield (TIPS)", current_value=real_cur, previous_value=real_prev,
            consensus_forecast=None, unit="%",
            deviation_impact="hawkish" if real_cur > real_prev else "dovish",
            confidence_score=9.5, short_term_gold_impact=real_impact,
            medium_term_gold_impact=real_impact, long_term_gold_impact=real_impact,
        )

        y10, y10_prev = data["yield_10y"]
        y2, y2_prev = data["yield_2y"]
        metrics["yield_curve"] = MacroMetric(
            name="10Y-2Y Spread", current_value=round(y10 - y2, 3),
            previous_value=round(y10_prev - y2_prev, 3),
            consensus_forecast=None, unit="%", deviation_impact="neutral",
            confidence_score=7.0,
            short_term_gold_impact="bullish" if _yield_curve(y10, y2) == "inverted" else "neutral",
            medium_term_gold_impact="bullish" if _yield_curve(y10, y2) == "inverted" else "neutral",
            long_term_gold_impact="neutral",
        )

        for key, label in [("ppi", "PPI"), ("unemployment", "Unemployment Rate"),
                            ("gdp", "GDP Growth"), ("core_cpi", "Core CPI")]:
            cur, prev = data[key]
            metrics[key] = MacroMetric(
                name=label, current_value=cur, previous_value=prev,
                consensus_forecast=None, unit="%", deviation_impact="neutral",
                confidence_score=7.0, short_term_gold_impact="neutral",
                medium_term_gold_impact="neutral", long_term_gold_impact="neutral",
            )

        usd_score = (fed_cur - 2.0) * 0.5 + (real_cur) * 0.8 - (cpi_cur - 2.0) * 0.2
        usd_score = round(max(-10.0, min(10.0, usd_score)), 2)

        inflation_reg = _inflation_regime(cpi_cur, data["core_cpi"][0])
        real_reg = "negative" if real_cur < 0 else "flattening" if abs(real_cur) < 0.5 else "positive"
        gold_bias = _gold_macro_bias(fed_cur, real_cur, cpi_cur, usd_score, inflation_reg)

        return MacroSummary(
            metrics=metrics,
            usd_strength_score=usd_score,
            inflation_regime=inflation_reg,
            real_rate_regime=real_reg,
            gold_macro_bias=gold_bias,
            geopolitical_risk_index=geopolitical_risk,
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def get_impact_summary(self) -> dict:
        return self.get_macro_summary().to_dict()


if __name__ == "__main__":
    agent = FundamentalAgent()
    print(json.dumps(agent.get_impact_summary(), indent=2, default=str))

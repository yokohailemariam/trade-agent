"""Layer 12: Confidence & Uncertainty Quantifier — probabilistic outputs and calibration."""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from loguru import logger


@dataclass
class UncertaintyEstimate:
    point_estimate: float
    confidence_interval_80: tuple[float, float]
    confidence_interval_95: tuple[float, float]
    entropy: float
    confidence_label: str
    confidence_pct: float


@dataclass
class MonteCarloResult:
    simulations: int
    prob_hit_tp1: float
    prob_hit_tp2: float
    prob_hit_tp3: float
    prob_hit_sl: float
    expected_return_pct: float
    max_adverse_excursion_avg: float


@dataclass
class ConfidenceSummary:
    overall_confidence: float
    confidence_label: str
    price_estimate: UncertaintyEstimate
    monte_carlo: Optional[MonteCarloResult]
    ensemble_agreement: float
    layer_disagreements: list[str]
    trading_recommendation: str
    uncertainty_statement: str
    calibration_warning: Optional[str]
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


def _entropy(probs: list[float]) -> float:
    probs = [max(p, 1e-9) for p in probs]
    total = sum(probs)
    probs = [p / total for p in probs]
    return round(-sum(p * np.log2(p) for p in probs) / np.log2(len(probs)), 3)


def _confidence_label(conf: float) -> str:
    if conf >= 0.80:
        return "high"
    if conf >= 0.65:
        return "medium"
    if conf >= 0.50:
        return "low"
    return "avoid"


def _monte_carlo(
    current_price: float,
    atr: float,
    tp1: float,
    tp2: float,
    tp3: float,
    sl: float,
    direction: str = "long",
    n_sims: int = 5000,
    horizon_bars: int = 48,
) -> MonteCarloResult:
    sigma = atr / (current_price + 1e-9)
    tp1_hits = tp2_hits = tp3_hits = sl_hits = 0
    mae_list = []

    rng = np.random.default_rng()
    for _ in range(n_sims):
        shocks = rng.normal(0.0, sigma, horizon_bars)
        price = current_price
        peak = trough = current_price
        hit_tp1 = hit_tp2 = hit_tp3 = hit_sl = False

        for shock in shocks:
            price *= (1 + shock)
            peak = max(peak, price)
            trough = min(trough, price)

            if direction == "long":
                hit_tp1 = hit_tp1 or price >= tp1
                hit_tp2 = hit_tp2 or price >= tp2
                hit_tp3 = hit_tp3 or price >= tp3
                if price <= sl:
                    hit_sl = True
                    break
            else:
                hit_tp1 = hit_tp1 or price <= tp1
                hit_tp2 = hit_tp2 or price <= tp2
                hit_tp3 = hit_tp3 or price <= tp3
                if price >= sl:
                    hit_sl = True
                    break

        tp1_hits += hit_tp1
        tp2_hits += hit_tp2
        tp3_hits += hit_tp3
        sl_hits += hit_sl
        mae = abs((trough if direction == "long" else peak) - current_price) / (current_price + 1e-9) * 100
        mae_list.append(mae)

    return MonteCarloResult(
        simulations=n_sims,
        prob_hit_tp1=round(tp1_hits / n_sims, 3),
        prob_hit_tp2=round(tp2_hits / n_sims, 3),
        prob_hit_tp3=round(tp3_hits / n_sims, 3),
        prob_hit_sl=round(sl_hits / n_sims, 3),
        expected_return_pct=round(
            (tp2_hits / n_sims * abs(tp2 - current_price) -
             sl_hits / n_sims * abs(sl - current_price)) /
            (current_price + 1e-9) * 100, 2
        ),
        max_adverse_excursion_avg=round(float(np.mean(mae_list)), 3),
    )


class UncertaintyQuantifier:
    def quantify(self, analysis: dict, signals: list[dict]) -> dict:
        tech = analysis.get("technical_analysis", {})
        macro = analysis.get("macro_fundamentals", {})
        news = analysis.get("news_sentiment", {})
        patterns = analysis.get("historical_patterns", {})
        risk = analysis.get("risk_assessment", {})
        snapshot = analysis.get("market_snapshot", {})

        primary = snapshot.get("primary", {}) if snapshot else {}
        current_price = float(primary.get("close", 2000.0)) if primary else 2000.0
        atr = float(risk.get("atr", 15.0)) if risk else 15.0

        # Ensemble agreement across timeframes
        biases: list[str] = []
        if isinstance(tech, dict):
            for tf_d in tech.values():
                if isinstance(tf_d, dict) and "bias" in tf_d:
                    biases.append(tf_d["bias"])

        bull_pct = biases.count("bullish") / len(biases) if biases else 0.5
        bear_pct = biases.count("bearish") / len(biases) if biases else 0.5
        agreement = max(bull_pct, bear_pct)
        dominant_tech = "bullish" if bull_pct > 0.5 else "bearish" if bear_pct > 0.5 else "neutral"

        macro_bias = macro.get("gold_macro_bias", 5.0) if isinstance(macro, dict) else 5.0
        macro_direction = "bullish" if macro_bias > 6 else "bearish" if macro_bias < 4 else "neutral"
        news_sent = news.get("aggregated_sentiment", 0.0) if isinstance(news, dict) else 0.0
        news_direction = "bullish" if news_sent > 0.1 else "bearish" if news_sent < -0.1 else "neutral"

        disagreements: list[str] = []
        if dominant_tech != "neutral" and macro_direction != "neutral" and dominant_tech != macro_direction:
            disagreements.append(f"Technical ({dominant_tech}) conflicts with Macro ({macro_direction})")
        if dominant_tech != "neutral" and news_direction != "neutral" and dominant_tech != news_direction:
            disagreements.append(f"Technical ({dominant_tech}) conflicts with News ({news_direction})")

        overall_conf = round(agreement * (0.8 if not disagreements else 0.5), 2)

        std_1d = atr * 1.5
        price_est = UncertaintyEstimate(
            point_estimate=round(current_price, 2),
            confidence_interval_80=(round(current_price - std_1d, 2), round(current_price + std_1d, 2)),
            confidence_interval_95=(round(current_price - std_1d * 2, 2), round(current_price + std_1d * 2, 2)),
            entropy=_entropy([overall_conf, 1 - overall_conf]),
            confidence_label=_confidence_label(overall_conf),
            confidence_pct=round(overall_conf * 100, 1),
        )

        mc_result = None
        if signals:
            sig = signals[0]
            try:
                entry = (sig["entry_zone"][0] + sig["entry_zone"][1]) / 2
                mc_result = _monte_carlo(
                    current_price=entry, atr=atr,
                    tp1=sig["tp1"], tp2=sig["tp2"], tp3=sig["tp3"],
                    sl=sig["stop_loss"], direction=sig["direction"],
                )
            except Exception as e:
                logger.warning(f"Monte Carlo failed: {e}")

        label = _confidence_label(overall_conf)
        if label == "high":
            rec = "trade"
            stmt = f"High confidence ({overall_conf:.0%}): {dominant_tech.capitalize()} continuation expected"
        elif label == "medium":
            rec = "trade"
            stmt = f"Medium confidence ({overall_conf:.0%}): {dominant_tech.capitalize()} bias with some uncertainty"
        elif label == "low":
            rec = "wait"
            stmt = f"Low confidence ({overall_conf:.0%}): Mixed signals — wait for confirmation"
        else:
            rec = "avoid"
            stmt = f"Avoid trading: Uncertainty exceeds acceptable threshold ({overall_conf:.0%})"

        calib_warning = None
        if overall_conf > 0.85 and disagreements:
            calib_warning = "Overconfidence risk: high score despite layer disagreements"
        pattern_conf = patterns.get("pattern_confidence", 0.0) if isinstance(patterns, dict) else 0.0
        if pattern_conf < 0.3 and label in ("high", "medium") and not calib_warning:
            calib_warning = "Low historical pattern match — confidence may be overstated"

        return ConfidenceSummary(
            overall_confidence=overall_conf,
            confidence_label=label,
            price_estimate=price_est,
            monte_carlo=mc_result,
            ensemble_agreement=round(agreement, 3),
            layer_disagreements=disagreements,
            trading_recommendation=rec,
            uncertainty_statement=stmt,
            calibration_warning=calib_warning,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ).to_dict()


if __name__ == "__main__":
    sample = {
        "technical_analysis": {
            "H1": {"bias": "bullish"}, "H4": {"bias": "bullish"}, "Daily": {"bias": "neutral"},
        },
        "macro_fundamentals": {"gold_macro_bias": 7.0},
        "news_sentiment": {"aggregated_sentiment": 0.2},
        "historical_patterns": {"pattern_confidence": 0.65, "probability_of_up_move": 0.70},
        "risk_assessment": {"atr": 18.0, "danger_level": 3},
        "market_snapshot": {"primary": {"close": 2340.0}},
    }
    uq = UncertaintyQuantifier()
    print(json.dumps(uq.quantify(sample, []), indent=2, default=str))

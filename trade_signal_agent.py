"""Layer 11: Trade Signal Generator — specific setups with entries, stops, and targets."""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

from loguru import logger


@dataclass
class TradeSignal:
    direction: str
    setup_type: str
    entry_zone: tuple[float, float]
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    rr_ratio: float
    probability: float
    confidence: float
    reasoning: str
    invalidation_conditions: list[str]
    risk_percent: float
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


def _calc_rr(entry: float, sl: float, tp: float) -> float:
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    return round(reward / (risk + 1e-9), 2)


def _probability_score(
    tf_alignment: int,
    sm_confirmed: bool,
    pattern_match_pct: float,
    news_event_in_2h: bool,
    low_liquidity: bool,
) -> float:
    score = 0.50
    score += tf_alignment * 0.10
    if sm_confirmed:
        score += 0.10
    if pattern_match_pct > 0.70:
        score += 0.10
    if news_event_in_2h:
        score -= 0.10
    if low_liquidity:
        score -= 0.20
    return round(max(0.0, min(1.0, score)), 2)


class TradeSignalAgent:
    def generate_signals(self, analysis: dict) -> list[dict]:
        signals: list[TradeSignal] = []

        tech = analysis.get("technical_analysis", {})
        smart = analysis.get("smart_money", {})
        risk = analysis.get("risk_assessment", {})
        patterns = analysis.get("historical_patterns", {})
        news = analysis.get("news_sentiment", {})
        snapshot = analysis.get("market_snapshot", {})

        primary = snapshot.get("primary", {})
        current_price = primary.get("close", 2000.0)
        if isinstance(current_price, str):
            try:
                current_price = float(current_price)
            except ValueError:
                current_price = 2000.0

        atr = risk.get("atr", 15.0)
        vol_regime = risk.get("volatility_regime", "normal")
        risk_pct = risk.get("recommended_risk_percent", 1.0)
        danger = risk.get("danger_level", 5)

        if danger >= 8:
            logger.warning("Danger level too high — skipping signal generation")
            return []

        has_news_risk = bool(isinstance(news, dict) and len(news.get("high_impact_alerts", [])) > 0)
        low_liq = not risk.get("spread_normal", True)

        bullish_tfs = sum(
            1 for tf_data in tech.values()
            if isinstance(tf_data, dict) and tf_data.get("bias") == "bullish"
        ) if isinstance(tech, dict) else 0
        bearish_tfs = sum(
            1 for tf_data in tech.values()
            if isinstance(tf_data, dict) and tf_data.get("bias") == "bearish"
        ) if isinstance(tech, dict) else 0

        sm_phase = smart.get("sm_phase", "unknown") if isinstance(smart, dict) else "unknown"
        sm_bull = sm_phase in ("accumulation", "markup")
        sm_bear = sm_phase in ("distribution", "markdown")

        pattern_prob = patterns.get("probability_of_up_move", 0.5) if isinstance(patterns, dict) else 0.5
        pattern_conf = patterns.get("pattern_confidence", 0.0) if isinstance(patterns, dict) else 0.0

        liquidity_zones = smart.get("liquidity_zones", []) if isinstance(smart, dict) else []
        nearby_lows = [z for z in liquidity_zones
                       if isinstance(z, dict) and
                       0 < (current_price - z.get("price", 0)) / (current_price + 1e-9) < 0.01
                       and not z.get("swept", True)]
        nearby_highs = [z for z in liquidity_zones
                        if isinstance(z, dict) and
                        0 < (z.get("price", 0) - current_price) / (current_price + 1e-9) < 0.01
                        and not z.get("swept", True)]

        sl_mult = 2.5 if vol_regime == "low" else 1.5

        # ── Long setup ────────────────────────────────────────────────────
        if bullish_tfs >= 2 or sm_bull:
            entry_low = current_price - atr * 0.3
            entry_high = current_price + atr * 0.1
            sl = current_price - atr * sl_mult
            tp1 = current_price + atr * 1.5
            tp2 = nearby_highs[0].get("price", current_price + atr * 2.5) if nearby_highs else current_price + atr * 2.5
            tp3 = current_price + atr * 4.0

            prob = _probability_score(min(bullish_tfs, 3), sm_bull, pattern_prob, has_news_risk, low_liq)

            if prob >= 0.65:
                entry_mid = (entry_low + entry_high) / 2
                reasons = []
                if bullish_tfs >= 2:
                    reasons.append(f"{bullish_tfs} timeframes aligned bullish")
                if sm_bull:
                    reasons.append(f"Smart money in {sm_phase} phase")
                if pattern_conf > 0.5:
                    reasons.append(f"Historical analog {pattern_conf:.0%} confidence")

                signals.append(TradeSignal(
                    direction="long",
                    setup_type="continuation" if sm_phase == "markup" else "reversal",
                    entry_zone=(round(entry_low, 2), round(entry_high, 2)),
                    stop_loss=round(sl, 2),
                    tp1=round(tp1, 2), tp2=round(tp2, 2), tp3=round(tp3, 2),
                    rr_ratio=_calc_rr(entry_mid, sl, tp2),
                    probability=prob, confidence=round(pattern_conf, 2),
                    reasoning=" | ".join(reasons) or "Multi-factor bullish confluence",
                    invalidation_conditions=[
                        f"Price closes below {round(sl, 2)}",
                        "Bearish BOS confirmed on H1",
                        "DXY rallies >0.5% on volume",
                        "Hawkish Fed surprise",
                    ],
                    risk_percent=risk_pct,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))

        # ── Short setup ───────────────────────────────────────────────────
        if bearish_tfs >= 2 or sm_bear:
            entry_low = current_price - atr * 0.1
            entry_high = current_price + atr * 0.3
            sl = current_price + atr * sl_mult
            tp1 = current_price - atr * 1.5
            tp2 = nearby_lows[0].get("price", current_price - atr * 2.5) if nearby_lows else current_price - atr * 2.5
            tp3 = current_price - atr * 4.0

            prob = _probability_score(min(bearish_tfs, 3), sm_bear, 1 - pattern_prob, has_news_risk, low_liq)

            if prob >= 0.65:
                entry_mid = (entry_low + entry_high) / 2
                reasons = []
                if bearish_tfs >= 2:
                    reasons.append(f"{bearish_tfs} timeframes aligned bearish")
                if sm_bear:
                    reasons.append(f"Smart money in {sm_phase} phase")

                signals.append(TradeSignal(
                    direction="short",
                    setup_type="continuation" if sm_phase == "markdown" else "reversal",
                    entry_zone=(round(entry_low, 2), round(entry_high, 2)),
                    stop_loss=round(sl, 2),
                    tp1=round(tp1, 2), tp2=round(tp2, 2), tp3=round(tp3, 2),
                    rr_ratio=_calc_rr(entry_mid, sl, tp2),
                    probability=prob, confidence=round(pattern_conf, 2),
                    reasoning=" | ".join(reasons) or "Multi-factor bearish confluence",
                    invalidation_conditions=[
                        f"Price closes above {round(sl, 2)}",
                        "Bullish BOS confirmed on H1",
                        "DXY drops >0.5% on volume",
                        "Dovish Fed pivot news",
                    ],
                    risk_percent=risk_pct,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))

        return [s.to_dict() for s in signals]


if __name__ == "__main__":
    import asyncio, sys
    sys.path.insert(0, ".")
    from orchestrator_agent import XAUUSDAnalysisOrchestrator

    async def _test():
        orch = XAUUSDAnalysisOrchestrator()
        analysis = await orch.generate_full_analysis()
        agent = TradeSignalAgent()
        signals = agent.generate_signals(analysis)
        print(json.dumps(signals, indent=2, default=str))

    asyncio.run(_test())

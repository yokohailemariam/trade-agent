"""Layer 13: LLM Interface & Output Formatter — uses Gemini API for institutional-grade analysis."""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False
    logger.warning("google-generativeai not installed — falling back to structured formatter")

from trade_signal_agent import TradeSignalAgent
from confidence_agent import UncertaintyQuantifier

DISCLAIMER = """⚠️ DISCLAIMER: This is AI-generated analysis for research purposes only. Not financial advice. \
Past performance does not guarantee future results. Always apply proper risk management."""

REPORT_PROMPT = """\
You are a senior institutional gold analyst producing a professional XAUUSD market intelligence report.

Below is structured JSON data from a multi-layer trading analysis system. Transform it into a complete, \
authoritative report with EXACTLY these 13 sections in order:

1. Market Summary
2. Macro Fundamentals
3. News Analysis
4. Technical Analysis
5. Session Analysis
6. Historical Trend Comparison
7. Sentiment Analysis
8. Correlation Analysis
9. Trade Opportunities
10. Risk Management
11. Institutional Smart Money Analysis
12. Forecast
13. Final Trading Outlook

RULES:
- Use ## headers for each section
- Use bullet points and tables where appropriate
- Add [XX% conf] confidence tags where relevant
- Keep paragraphs to 2–4 sentences max
- NEVER say "guaranteed", "certain", or "100%"
- NEVER recommend leverage > 10x
- If probability < 60%, state "Low confidence — avoid trading"
- For Trade Opportunities: only include if probability > 65%, show exact R:R calculation
- Include invalidation conditions for every trade setup
- Use professional, neutral language — no emotional words

STRUCTURED DATA:
{data}

Begin the report now. Start directly with ## 1. Market Summary (no preamble).
"""


def _safe(val, default="N/A", fmt=None):
    if val is None or val == "":
        return default
    try:
        return fmt.format(val) if fmt else str(val)
    except Exception:
        return default


def _bias_emoji(bias: str) -> str:
    return {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}.get(str(bias).lower(), "⚪")


# ── Structured fallback formatters (used when Gemini is unavailable) ──────────

def _format_market_summary(snapshot: dict, tech: dict, risk: dict) -> str:
    primary = snapshot.get("primary", {}) if snapshot else {}
    price = _safe(primary.get("close"), fmt="{:.2f}")
    ts = _safe(primary.get("timestamp"))
    session = _safe(primary.get("session_label"))
    h1 = tech.get("H1", {}) if isinstance(tech, dict) else {}
    d1 = tech.get("Daily", {}) if isinstance(tech, dict) else {}
    h1_bias = h1.get("bias", "N/A") if isinstance(h1, dict) else "N/A"
    d1_bias = d1.get("bias", "N/A") if isinstance(d1, dict) else "N/A"
    atr = _safe(risk.get("atr") if risk else None, fmt="${:.2f}")
    vol = _safe(risk.get("volatility_regime") if risk else None)
    danger = _safe(risk.get("danger_level") if risk else None)
    return "\n".join([
        f"- **Price:** {price} USD | **Session:** {session} | **Timestamp:** {ts}",
        f"- **H1 Bias:** {_bias_emoji(h1_bias)} {h1_bias.upper()} | **Daily:** {_bias_emoji(d1_bias)} {d1_bias.upper()}",
        f"- **ATR:** {atr} | **Volatility:** {vol} | **Danger:** {danger}/10",
    ])


def _format_macro(macro: dict) -> str:
    if not macro or "error" in macro:
        return "Insufficient macro data."
    lines = [
        f"- **Gold Macro Bias:** {_safe(macro.get('gold_macro_bias'), fmt='{:.1f}')}/10",
        f"- **Inflation Regime:** {_safe(macro.get('inflation_regime'))} | "
        f"**Real Rates:** {_safe(macro.get('real_rate_regime'))}",
        f"- **USD Strength:** {_safe(macro.get('usd_strength_score'), fmt='{:.2f}')} | "
        f"**Geo Risk:** {_safe(macro.get('geopolitical_risk_index'), fmt='{:.1f}')}/10",
    ]
    for key, m in list((macro.get("metrics") or {}).items())[:4]:
        if isinstance(m, dict):
            lines.append(
                f"- **{m.get('name', key)}:** {_safe(m.get('current_value'), fmt='{:.2f}')} "
                f"(prev {_safe(m.get('previous_value'), fmt='{:.2f}')}) → "
                f"Gold ST: {_safe(m.get('short_term_gold_impact'))}"
            )
    return "\n".join(lines)


def _format_technical(tech: dict) -> str:
    if not tech or "error" in tech:
        return "Insufficient technical data."
    rows = []
    for tf in ["Monthly", "Weekly", "Daily", "H4", "H1", "M15", "M5", "M1"]:
        d = tech.get(tf, {})
        if not isinstance(d, dict) or "error" in d:
            continue
        bias = d.get("bias", "N/A")
        mom = d.get("momentum", {}) or {}
        trend = d.get("trend", {}) or {}
        struct = d.get("structure", {}) or {}
        rows.append(
            f"| {tf} | {_bias_emoji(bias)} {bias.upper()} | "
            f"{_safe(mom.get('rsi'), fmt='{:.1f}')} | "
            f"{_safe(trend.get('adx'), fmt='{:.1f}')} | "
            f"{_safe(struct.get('current_structure'))} | "
            f"{_safe(d.get('confidence'), fmt='{:.0%}')} |"
        )
    if not rows:
        return "No timeframe data."
    return ("| TF | Bias | RSI | ADX | Structure | Conf |\n"
            "|----|------|-----|-----|-----------|------|\n" + "\n".join(rows))


def _format_trade_opportunities(signals: list[dict], confidence: dict) -> str:
    rec = confidence.get("trading_recommendation", "wait") if isinstance(confidence, dict) else "wait"
    if rec == "avoid":
        pct = (confidence.get("overall_confidence", 0) * 100) if isinstance(confidence, dict) else 0
        return f"**No trade.** Confidence too low ({pct:.0f}%). Wait for clarity."
    if not signals:
        return "No high-probability setups at this time."
    lines = []
    for sig in signals:
        d = "🟢 LONG" if sig["direction"] == "long" else "🔴 SHORT"
        lines += [
            f"### {d} — {sig.get('setup_type', '').replace('_', ' ').title()}",
            f"- **Entry:** {sig['entry_zone'][0]:.2f}–{sig['entry_zone'][1]:.2f} | "
            f"**SL:** {sig['stop_loss']:.2f}",
            f"- **TP1:** {sig['tp1']:.2f} | **TP2:** {sig['tp2']:.2f} | **TP3:** {sig['tp3']:.2f}",
            f"- **R:R:** {sig['rr_ratio']:.1f}:1 | **P:** {sig['probability']:.0%} | "
            f"**Risk:** {sig['risk_percent']:.1f}%",
            f"- *{sig['reasoning']}*",
            "- **Invalidation:**",
        ]
        for inv in (sig.get("invalidation_conditions") or [])[:3]:
            lines.append(f"  - {inv}")
    return "\n".join(lines)


def _format_risk(risk: dict) -> str:
    if not risk or "error" in risk:
        return "Insufficient risk data."
    lines = [
        f"- **Recommended Risk:** {_safe(risk.get('recommended_risk_percent'), fmt='{:.1f}')}% | "
        f"**SL Distance:** {_safe(risk.get('adjusted_sl_points'), fmt='{:.2f}')} pts | "
        f"**Max Lots:** {_safe(risk.get('max_position_size_lots'), fmt='{:.2f}')}",
        f"- **Danger:** {_safe(risk.get('danger_level'))}/10 | "
        f"**Volatility:** {_safe(risk.get('volatility_regime'))}",
    ]
    for w in (risk.get("risk_warnings") or []):
        lines.append(f"- ⚠️ {w}")
    return "\n".join(lines)


def _format_final_outlook(confidence: dict, signals: list[dict], risk: dict) -> str:
    rec = (confidence.get("trading_recommendation") or "wait").upper() if isinstance(confidence, dict) else "WAIT"
    label = (confidence.get("confidence_label") or "low").upper() if isinstance(confidence, dict) else "LOW"
    pct = (confidence.get("overall_confidence", 0) * 100) if isinstance(confidence, dict) else 0
    danger = risk.get("danger_level", 5) if isinstance(risk, dict) else 5
    lines = [f"**{rec}** | Confidence: {label} ({pct:.0f}%) | Danger: {danger}/10", ""]
    if signals:
        s = signals[0]
        lines += [
            f"**Setup:** {'LONG' if s['direction'] == 'long' else 'SHORT'} — "
            f"{s['setup_type'].replace('_', ' ').title()}",
            f"- Entry {s['entry_zone'][0]:.2f}–{s['entry_zone'][1]:.2f} | "
            f"SL {s['stop_loss']:.2f} | TP2 {s['tp2']:.2f} | R:R {s['rr_ratio']:.1f}:1",
        ]
    else:
        lines.append("No actionable setup. Monitor key levels and await confirmation.")
    lines.append("\n*Manage risk. Never risk more than you can afford to lose.*")
    return "\n".join(lines)


def _structured_report(analysis: dict, signals: list[dict], confidence: dict) -> str:
    """Pure-Python fallback formatter (no LLM dependency)."""
    tech = analysis.get("technical_analysis", {})
    macro = analysis.get("macro_fundamentals", {})
    news = analysis.get("news_sentiment", {})
    session = analysis.get("session_analysis", {})
    patterns = analysis.get("historical_patterns", {})
    corr = analysis.get("correlation", {})
    smart = analysis.get("smart_money", {})
    risk = analysis.get("risk_assessment", {})
    snapshot = analysis.get("market_snapshot", {})

    def _news_fmt():
        if not news or "error" in news:
            return "Insufficient news data."
        sent = _safe(news.get("aggregated_sentiment"), fmt="{:.3f}")
        fng = news.get("fear_greed_score")
        retail = _safe(news.get("retail_sentiment_bias"))
        lines = [
            f"- **Sentiment:** {sent} | **Fear & Greed:** {f'{fng:.0f}/100' if fng else 'N/A'}",
            f"- **Retail Bias:** {retail.upper()}",
        ]
        for a in [x for x in (news.get("articles") or [])
                  if isinstance(x, dict) and x.get("relevance") == "direct"][:3]:
            lines.append(f"- {a.get('title', '')[:100]} [{a.get('gold_impact')}]")
        return "\n".join(lines)

    def _session_fmt():
        if not session or "error" in session:
            return "Insufficient session data."
        lines = [
            f"- **Session:** {_safe(session.get('active_session'))} | "
            f"**Kill Zone:** {_safe(session.get('active_kill_zone'), 'None')}",
            f"- **Range:** {_safe(session.get('current_range_pips'), fmt='{:.2f}')} pts | "
            f"**Regime:** {_safe(session.get('volatility_regime'))}",
            f"- {_safe(session.get('expected_session_outcome'))}",
        ]
        for t in (session.get("session_traps") or []):
            lines.append(f"- ⚠️ {t}")
        return "\n".join(lines)

    def _hist_fmt():
        if not patterns or "error" in patterns:
            return "Insufficient historical pattern data."
        lines = [
            f"- **P(Up Move):** {_safe(patterns.get('probability_of_up_move'), fmt='{:.1%}')} "
            f"[Conf: {_safe(patterns.get('pattern_confidence'), fmt='{:.1%}')}]",
            f"- **Expected 1D:** {_safe(patterns.get('expected_move_1d'), fmt='{:+.2f}%')} | "
            f"**1W:** {_safe(patterns.get('expected_move_1w'), fmt='{:+.2f}%')}",
        ]
        for a in (patterns.get("top_analogs") or [])[:3]:
            if isinstance(a, dict):
                lines.append(
                    f"- **{a.get('label')}** ({_safe(a.get('similarity_score'), fmt='{:.0%}')} sim) → "
                    f"1D: {_safe(a.get('outcome_1d'), fmt='{:+.1f}%')} | "
                    f"WR: {_safe(a.get('win_rate'), fmt='{:.0%}')}"
                )
        return "\n".join(lines)

    def _sent_fmt():
        sent = news.get("aggregated_sentiment", 0) if isinstance(news, dict) else 0
        fng = news.get("fear_greed_score") if isinstance(news, dict) else None
        retail = news.get("retail_sentiment_bias", "neutral") if isinstance(news, dict) else "neutral"
        ens = confidence.get("ensemble_agreement", 0.5) if isinstance(confidence, dict) else 0.5
        fng_l = ("Extreme Fear" if fng and fng < 25 else "Fear" if fng and fng < 45 else
                 "Greed" if fng and fng > 55 else "Neutral") if fng else "N/A"
        lines = [
            f"- **Sentiment:** {sent:.3f} | **F&G:** {f'{fng:.0f}/100 ({fng_l})' if fng else 'N/A'}",
            f"- **Retail:** {retail.upper()} | **Layer Agreement:** {ens:.0%}",
        ]
        for d in (confidence.get("layer_disagreements") or []):
            lines.append(f"- ⚠️ {d}")
        return "\n".join(lines)

    def _corr_fmt():
        if not corr or "error" in corr:
            return "Insufficient correlation data."
        lines = [
            f"- **Regime:** {_safe(corr.get('regime')).upper()} | "
            f"**Driver:** {_safe(corr.get('strongest_influence'))}",
        ]
        for sym, val in list((corr.get("correlation_matrix") or {}).items())[:6]:
            if val is not None:
                lines.append(f"  - XAUUSD/{sym}: {val:+.3f}")
        for w in (corr.get("divergence_warnings") or [])[:3]:
            lines.append(f"- ⚠️ {w}")
        return "\n".join(lines)

    def _sm_fmt():
        if not smart or "error" in smart:
            return "Insufficient smart money data."
        lines = [
            f"- **Phase:** {_safe(smart.get('sm_phase')).upper()} | "
            f"**Liquidity Score:** {_safe(smart.get('engineered_liquidity_score'), fmt='{:.0f}')}/100",
            f"- **VWAP:** {_safe(smart.get('vwap'), fmt='{:.2f}')} | "
            f"**vs VWAP:** {_safe(smart.get('price_vs_vwap'))}",
        ]
        for t in (smart.get("trap_warnings") or []):
            lines.append(f"- ⚠️ {t}")
        for z in [z for z in (smart.get("liquidity_zones") or [])[:3]
                  if isinstance(z, dict) and z.get("magnitude") == "high"]:
            lines.append(f"- {z['price']:.2f} [{z['zone_type']}] {'✓' if z.get('swept') else '⚡'}")
        return "\n".join(lines)

    def _fc_fmt():
        if not confidence:
            return "Insufficient data."
        stmt = confidence.get("uncertainty_statement", "No forecast available")
        lines = [f"**{stmt}**"]
        mc = confidence.get("monte_carlo")
        if isinstance(mc, dict):
            lines.append(
                f"- Monte Carlo: P(TP1)={mc.get('prob_hit_tp1', 0):.0%} | "
                f"P(TP2)={mc.get('prob_hit_tp2', 0):.0%} | P(SL)={mc.get('prob_hit_sl', 0):.0%}"
            )
        pe = confidence.get("price_estimate", {})
        if isinstance(pe, dict) and pe.get("confidence_interval_80"):
            ci = pe["confidence_interval_80"]
            lines.append(f"- **80% CI:** {ci[0]:.2f}–{ci[1]:.2f} USD")
        calib = confidence.get("calibration_warning")
        if calib:
            lines.append(f"- ⚠️ {calib}")
        return "\n".join(lines)

    sections = [
        ("1. Market Summary",               _format_market_summary(snapshot, tech, risk)),
        ("2. Macro Fundamentals",           _format_macro(macro)),
        ("3. News Analysis",               _news_fmt()),
        ("4. Technical Analysis",          _format_technical(tech)),
        ("5. Session Analysis",            _session_fmt()),
        ("6. Historical Trend Comparison", _hist_fmt()),
        ("7. Sentiment Analysis",          _sent_fmt()),
        ("8. Correlation Analysis",        _corr_fmt()),
        ("9. Trade Opportunities",         _format_trade_opportunities(signals, confidence)),
        ("10. Risk Management",            _format_risk(risk)),
        ("11. Smart Money Analysis",       _sm_fmt()),
        ("12. Forecast",                   _fc_fmt()),
        ("13. Final Trading Outlook",      _format_final_outlook(confidence, signals, risk)),
    ]
    out = [DISCLAIMER, ""]
    for title, content in sections:
        out += [f"## {title}", "", content, "", "---", ""]
    return "\n".join(out)


# ── Gemini client ─────────────────────────────────────────────────────────────

class LLMFormatter:
    def __init__(self, gemini_api_key: str = "", gemini_model: str = "gemini-2.0-flash"):
        self._api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self._model_name = gemini_model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self._model = None
        self._signal_agent = TradeSignalAgent()
        self._uq = UncertaintyQuantifier()

        if _GENAI_AVAILABLE and self._api_key:
            try:
                genai.configure(api_key=self._api_key)
                self._model = genai.GenerativeModel(
                    model_name=self._model_name,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=8192,
                    ),
                    system_instruction=(
                        "You are a professional institutional gold analyst. "
                        "Produce concise, accurate, non-emotional analysis. "
                        "Never guarantee outcomes. Always flag uncertainty. "
                        "Use markdown formatting with ## section headers."
                    ),
                )
                logger.info(f"Gemini model initialized: {self._model_name}")
            except Exception as e:
                logger.warning(f"Gemini init failed: {e} — will use structured fallback")
                self._model = None
        else:
            if not self._api_key:
                logger.warning("GEMINI_API_KEY not set — using structured fallback formatter")

    async def _gemini_generate(self, prompt: str) -> Optional[str]:
        if self._model is None:
            return None
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._model.generate_content(prompt)
            )
            return response.text
        except Exception as e:
            logger.warning(f"Gemini generation failed: {e}")
            return None

    async def format_analysis(self, analysis: dict) -> str:
        signals = self._signal_agent.generate_signals(analysis)
        confidence = self._uq.quantify(analysis, signals)
        meta = analysis.get("meta", {})
        ts = meta.get("analysis_timestamp", datetime.now(timezone.utc).isoformat())
        elapsed = meta.get("elapsed_seconds", 0)

        header = (
            f"{DISCLAIMER}\n\n"
            f"# XAUUSD Trading Intelligence Report\n"
            f"*{ts} | {elapsed}s analysis time*\n\n---\n"
        )

        # Build compact payload for Gemini (drop raw articles to save tokens)
        gemini_payload = {
            "meta": meta,
            "market_snapshot": analysis.get("market_snapshot", {}),
            "technical_analysis": {
                tf: {k: v for k, v in d.items() if k != "support_resistance"}
                for tf, d in (analysis.get("technical_analysis") or {}).items()
                if isinstance(d, dict)
            },
            "macro_fundamentals": {
                k: v for k, v in (analysis.get("macro_fundamentals") or {}).items()
                if k != "metrics"
            },
            "macro_metrics_summary": {
                key: {
                    "name": m.get("name"),
                    "current_value": m.get("current_value"),
                    "previous_value": m.get("previous_value"),
                    "short_term_gold_impact": m.get("short_term_gold_impact"),
                }
                for key, m in ((analysis.get("macro_fundamentals") or {}).get("metrics") or {}).items()
            },
            "news_sentiment": {
                k: v for k, v in (analysis.get("news_sentiment") or {}).items()
                if k != "articles"
            },
            "news_top_headlines": [
                {"title": a.get("title"), "category": a.get("category"), "gold_impact": a.get("gold_impact")}
                for a in ((analysis.get("news_sentiment") or {}).get("articles") or [])
                if isinstance(a, dict) and a.get("relevance") == "direct"
            ][:5],
            "session_analysis": analysis.get("session_analysis", {}),
            "historical_patterns": analysis.get("historical_patterns", {}),
            "correlation": {
                k: v for k, v in (analysis.get("correlation") or {}).items()
                if k != "correlations"
            },
            "correlation_matrix": (analysis.get("correlation") or {}).get("correlation_matrix", {}),
            "smart_money": {
                k: v for k, v in (analysis.get("smart_money") or {}).items()
                if k not in ("signals", "liquidity_zones")
            },
            "top_liquidity_zones": [
                z for z in ((analysis.get("smart_money") or {}).get("liquidity_zones") or [])[:5]
                if isinstance(z, dict) and z.get("magnitude") == "high"
            ],
            "risk_assessment": analysis.get("risk_assessment", {}),
            "trade_signals": signals,
            "confidence_summary": confidence,
        }

        prompt = REPORT_PROMPT.format(data=json.dumps(gemini_payload, indent=2, default=str))
        gemini_report = await self._gemini_generate(prompt)

        if gemini_report:
            logger.info("Gemini report generated successfully")
            return header + gemini_report
        else:
            logger.info("Using structured fallback formatter")
            return header + _structured_report(analysis, signals, confidence)


if __name__ == "__main__":
    import asyncio, sys
    sys.path.insert(0, ".")
    from orchestrator_agent import XAUUSDAnalysisOrchestrator

    async def _test():
        orch = XAUUSDAnalysisOrchestrator()
        analysis = await orch.generate_full_analysis()
        api_key = os.getenv("GEMINI_API_KEY", "")
        formatter = LLMFormatter(gemini_api_key=api_key)
        print(await formatter.format_analysis(analysis))

    asyncio.run(_test())

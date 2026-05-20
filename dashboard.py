"""XAUUSD Trading Intelligence Dashboard — Streamlit visual interface."""
import asyncio
import json
import os
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XAUUSD Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stMetric { background: #1a1a2e; border-radius: 8px; padding: 12px; }
    div[data-testid="stExpander"] { border: 1px solid #333; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ── Async runner ──────────────────────────────────────────────────────────────
def run_async(coro):
    try:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    except Exception:
        return asyncio.run(coro)


# ── Cached data fetchers ──────────────────────────────────────────────────────
@st.cache_data(ttl=0, show_spinner=False)
def fetch_analysis(_trigger: int) -> dict:
    from orchestrator_agent import XAUUSDAnalysisOrchestrator
    orch = XAUUSDAnalysisOrchestrator(
        fred_api_key=os.getenv("FRED_API_KEY", ""),
        news_api_key=os.getenv("NEWS_API_KEY", ""),
    )
    return run_async(orch.generate_full_analysis())


@st.cache_data(ttl=0, show_spinner=False)
def fetch_report(analysis_json: str, _trigger: int) -> str:
    from llm_interface import LLMFormatter
    analysis = json.loads(analysis_json)
    formatter = LLMFormatter(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
    )
    return run_async(formatter.format_analysis(analysis))


@st.cache_data(ttl=0, show_spinner=False)
def fetch_price_df(_trigger: int) -> pd.DataFrame:
    from market_data_agent import get_historical_data
    try:
        df = get_historical_data("XAUUSD", period="30d", interval="1h")
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        return pd.DataFrame({"_error": [str(e)]})


# ── Helpers ───────────────────────────────────────────────────────────────────
def _safe(val, default="N/A", fmt=None):
    if val is None or val == "":
        return default
    try:
        return fmt.format(val) if fmt else val
    except Exception:
        return default


def _danger_icon(level: int) -> str:
    return "🔴" if level >= 7 else "🟡" if level >= 4 else "🟢"


# ── Chart builders ─────────────────────────────────────────────────────────────
def price_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03,
        subplot_titles=("XAUUSD 1H — Price + EMAs", "RSI (14)", "MACD"),
    )
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price", increasing_line_color="#00c853", decreasing_line_color="#d50000",
    ), row=1, col=1)

    close = df["close"]
    for period, color, width in [(9, "#ff9800", 1.5), (20, "#29b6f6", 1.5),
                                  (50, "#ce93d8", 1), (200, "#ef5350", 1)]:
        fig.add_trace(go.Scatter(
            x=df.index, y=close.ewm(span=period, adjust=False).mean(),
            name=f"EMA{period}", line=dict(color=color, width=width),
        ), row=1, col=1)

    # RSI
    delta = close.diff()
    rsi = 100 - (100 / (1 + delta.clip(lower=0).rolling(14).mean() /
                        (-delta.clip(upper=0)).rolling(14).mean().replace(0, 1e-9)))
    fig.add_trace(go.Scatter(x=df.index, y=rsi, name="RSI",
                             line=dict(color="#ffd700", width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(213,0,0,0.5)", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="rgba(0,200,83,0.5)", row=2, col=1)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    fig.add_trace(go.Bar(x=df.index, y=hist, name="Hist",
                         marker_color=["#00c853" if v >= 0 else "#d50000" for v in hist]), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd, name="MACD",
                             line=dict(color="#29b6f6", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=signal, name="Signal",
                             line=dict(color="#ff9800", width=1.5)), row=3, col=1)

    fig.update_layout(
        template="plotly_dark", height=680, xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", y=1.02, x=0),
    )
    return fig


def tf_bias_chart(tech: dict) -> go.Figure:
    order = ["M1", "M5", "M15", "H1", "H4", "Daily", "Weekly", "Monthly"]
    labels, values, clrs = [], [], []
    for tf in order:
        d = tech.get(tf, {}) if isinstance(tech, dict) else {}
        if not isinstance(d, dict) or "error" in d:
            continue
        bias = d.get("bias", "neutral")
        conf = float(d.get("confidence", 0.5) or 0.5)
        val = conf if bias == "bullish" else -conf if bias == "bearish" else 0
        labels.append(tf)
        values.append(round(val, 2))
        clrs.append("#00c853" if val > 0 else "#d50000" if val < 0 else "#555")
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=clrs,
        text=[f"{'▲' if v > 0 else '▼' if v < 0 else '—'} {abs(v):.0%}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark", height=260,
        title="Multi-Timeframe Bias (▲ bullish / ▼ bearish)",
        yaxis=dict(range=[-1.1, 1.1], title="Strength"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)")
    return fig


def correlation_chart(corr: dict) -> go.Figure:
    matrix = corr.get("correlation_matrix", {}) if isinstance(corr, dict) else {}
    syms = list(matrix.keys())
    vals = [float(matrix[s] or 0) for s in syms]
    fig = go.Figure(go.Bar(
        x=syms, y=vals,
        marker_color=["#00c853" if v > 0 else "#d50000" for v in vals],
        text=[f"{v:+.2f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark", height=280, title="XAUUSD Rolling Correlations",
        yaxis=dict(range=[-1.1, 1.1], title="Correlation"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)")
    return fig


def macro_gauge(macro: dict) -> go.Figure:
    bias = float(macro.get("gold_macro_bias", 5) or 5)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=bias,
        delta={"reference": 5.0},
        title={"text": "Gold Macro Bias", "font": {"size": 15}},
        gauge={
            "axis": {"range": [1, 10]},
            "bar": {"color": "#ffd700"},
            "steps": [
                {"range": [1, 4], "color": "#3b0d0d"},
                {"range": [4, 6], "color": "#1a1a2e"},
                {"range": [6, 10], "color": "#0d3b1e"},
            ],
        },
    ))
    fig.update_layout(
        template="plotly_dark", height=230,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


def monte_carlo_chart(mc: dict) -> go.Figure:
    labels = ["Hit TP1", "Hit TP2", "Hit TP3", "Hit SL"]
    vals = [mc.get(k, 0) for k in ["prob_hit_tp1", "prob_hit_tp2", "prob_hit_tp3", "prob_hit_sl"]]
    clrs = ["#00c853", "#1de9b6", "#69f0ae", "#d50000"]
    fig = go.Figure(go.Bar(
        x=labels, y=[v * 100 for v in vals], marker_color=clrs,
        text=[f"{v:.0%}" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark", height=280,
        title=f"Monte Carlo — {mc.get('simulations', 0):,} simulations",
        yaxis=dict(range=[0, 110], title="Probability %"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def liquidity_chart(smart: dict, current_price: float) -> go.Figure:
    zones = [z for z in (smart.get("liquidity_zones") or []) if isinstance(z, dict)][:15]
    if not zones:
        return go.Figure()
    prices = [z.get("price", 0) for z in zones]
    all_prices = prices + [current_price]
    span = max(all_prices) - min(all_prices)
    fig = go.Figure()
    for z in zones:
        p = z.get("price", 0)
        is_swept = z.get("swept", False)
        mag = {"high": 2, "medium": 1.5, "low": 1}.get(z.get("magnitude", "low"), 1)
        color = "rgba(100,100,100,0.4)" if is_swept else (
            "rgba(0,200,83,0.7)" if p < current_price else "rgba(213,0,0,0.7)"
        )
        fig.add_hline(y=p, line_color=color, line_width=mag,
                      line_dash="dot" if is_swept else "solid",
                      annotation_text=f"{z.get('zone_type','')[:12]} {p:.1f}{'✓' if is_swept else '⚡'}",
                      annotation_position="right")
    fig.add_hline(y=current_price, line_color="#ffd700", line_width=2.5,
                  annotation_text=f"Current {current_price:.2f}", annotation_position="left")
    fig.update_layout(
        template="plotly_dark", height=340, title="Liquidity Zones",
        yaxis=dict(
            range=[min(all_prices) - span * 0.05, max(all_prices) + span * 0.05],
            title="Price (USD)",
        ),
        xaxis=dict(visible=False),
        margin=dict(l=0, r=140, t=40, b=0),
    )
    return fig


# ── Signal card ───────────────────────────────────────────────────────────────
def signal_card(sig: dict):
    is_long = sig.get("direction", "long") == "long"
    color = "#00c853" if is_long else "#d50000"
    bg = "#0a2e1a" if is_long else "#2e0a0a"
    arrow = "▲ LONG" if is_long else "▼ SHORT"
    setup = sig.get("setup_type", "").replace("_", " ").title()
    elo, ehi = sig.get("entry_zone", [0, 0])
    sl, tp1, tp2, tp3 = sig.get("stop_loss", 0), sig.get("tp1", 0), sig.get("tp2", 0), sig.get("tp3", 0)
    rr, prob, risk = sig.get("rr_ratio", 0), sig.get("probability", 0), sig.get("risk_percent", 1.0)
    invalidations = "  •  ".join((sig.get("invalidation_conditions") or [])[:2])
    st.markdown(f"""
<div style="border:2px solid {color};border-radius:10px;padding:16px;margin:8px 0;background:{bg};">
  <div style="font-size:1.25rem;font-weight:700;color:{color};">{arrow} &nbsp; {setup}</div>
  <hr style="border-color:{color}33;margin:8px 0;">
  <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin:8px 0;">
    <div><div style="color:#aaa;font-size:.75rem;">Entry</div><div style="font-weight:600;">{elo:.2f}–{ehi:.2f}</div></div>
    <div><div style="color:#aaa;font-size:.75rem;">Stop</div><div style="font-weight:600;color:#ef5350;">{sl:.2f}</div></div>
    <div><div style="color:#aaa;font-size:.75rem;">TP1</div><div style="font-weight:600;color:#69f0ae;">{tp1:.2f}</div></div>
    <div><div style="color:#aaa;font-size:.75rem;">TP2</div><div style="font-weight:600;color:#00e676;">{tp2:.2f}</div></div>
    <div><div style="color:#aaa;font-size:.75rem;">TP3</div><div style="font-weight:600;color:#00c853;">{tp3:.2f}</div></div>
    <div><div style="color:#aaa;font-size:.75rem;">Risk</div><div style="font-weight:600;">{risk:.1f}%</div></div>
  </div>
  <div style="display:flex;gap:16px;margin:8px 0;">
    <span style="background:#222;padding:3px 10px;border-radius:20px;font-size:.9rem;">R:R {rr:.1f}:1</span>
    <span style="background:#222;padding:3px 10px;border-radius:20px;font-size:.9rem;">P: {prob:.0%}</span>
  </div>
  <div style="color:#ccc;font-size:.85rem;margin-top:6px;"><em>{sig.get('reasoning','')}</em></div>
  <div style="color:#888;font-size:.8rem;margin-top:4px;">Invalidation: {invalidations}</div>
</div>
""", unsafe_allow_html=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
  <span style="font-size:2rem;">📊</span>
  <div>
    <div style="font-size:1.8rem;font-weight:700;color:#ffd700;">XAUUSD Trading Intelligence</div>
    <div style="color:#888;font-size:.9rem;">Powered by Gemini AI · 13-Layer Multi-Agent System</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Session state
    for key in ("trigger", "analysis", "report"):
        if key not in st.session_state:
            st.session_state[key] = 0 if key == "trigger" else None

    col_btn, col_ts = st.columns([1, 5])
    with col_btn:
        refresh = st.button("🔄 Run Analysis", type="primary", use_container_width=True)
    with col_ts:
        if st.session_state.analysis:
            m = st.session_state.analysis.get("meta", {})
            st.caption(f"Last run: {m.get('analysis_timestamp','')[:19].replace('T',' ')} UTC  ·  {m.get('elapsed_seconds',0)}s")

    if refresh:
        st.session_state.trigger += 1
        with st.spinner("Running 13-layer analysis… (30–90 s)"):
            st.session_state.analysis = fetch_analysis(st.session_state.trigger)
        with st.spinner("Generating Gemini report…"):
            st.session_state.report = fetch_report(
                json.dumps(st.session_state.analysis, default=str),
                st.session_state.trigger,
            )
        st.rerun()

    if not st.session_state.analysis:
        st.info("👆 Click **Run Analysis** to start. First run takes ~60 seconds.")
        st.stop()

    A = st.session_state.analysis
    tech = A.get("technical_analysis", {}) or {}
    macro = A.get("macro_fundamentals", {}) or {}
    news = A.get("news_sentiment", {}) or {}
    session = A.get("session_analysis", {}) or {}
    patterns = A.get("historical_patterns", {}) or {}
    corr = A.get("correlation", {}) or {}
    smart = A.get("smart_money", {}) or {}
    risk = A.get("risk_assessment", {}) or {}
    snap = A.get("market_snapshot", {}) or {}
    primary = snap.get("primary", {}) or {}
    cur_price = float(primary.get("close", 0) or 0)

    from trade_signal_agent import TradeSignalAgent
    from confidence_agent import UncertaintyQuantifier
    signals = TradeSignalAgent().generate_signals(A)
    conf = UncertaintyQuantifier().quantify(A, signals)

    h1_bias = (tech.get("H1", {}) or {}).get("bias", "neutral")
    d1_bias = (tech.get("Daily", {}) or {}).get("bias", "neutral")
    atr = float(risk.get("atr", 0) or 0)
    danger = int(risk.get("danger_level", 0) or 0)
    sm_phase = smart.get("sm_phase", "unknown")
    macro_bias = float(macro.get("gold_macro_bias", 5) or 5)
    sent_score = float(news.get("aggregated_sentiment", 0) or 0)
    fng = news.get("fear_greed_score")
    overall_conf = float(conf.get("overall_confidence", 0) or 0)
    rec = (conf.get("trading_recommendation") or "wait").upper()

    # ── 1. Key metrics ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Market Snapshot")
    xauusd_src = snap.get("meta", {}).get("xauusd_source", "historical-feed")
    c = st.columns(8)
    c[0].metric("🥇 XAUUSD", f"${cur_price:,.2f}" if cur_price else "N/A",
                help=f"Source: {xauusd_src}")
    c[1].metric("H1 Bias", h1_bias.upper())
    c[2].metric("Daily Bias", d1_bias.upper())
    c[3].metric("ATR", f"${atr:.2f}" if atr else "N/A")
    c[4].metric(f"{_danger_icon(danger)} Danger", f"{danger}/10")
    c[5].metric("SM Phase", sm_phase.title())
    c[6].metric("Confidence", f"{overall_conf:.0%}")
    c[7].metric("Signal", rec)

    c2 = st.columns(4)
    c2[0].metric("Macro Bias", f"{macro_bias:.1f}/10")
    c2[1].metric("Sentiment", f"{sent_score:+.3f}")
    c2[2].metric("Fear & Greed", f"{fng:.0f}/100" if fng else "N/A")
    c2[3].metric("Session", session.get("active_session", "N/A"))

    # ── 2. Price chart ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🕯️ Price Chart")
    price_df = fetch_price_df(st.session_state.trigger)
    chart_cols = {"open", "high", "low", "close"}
    if not price_df.empty and chart_cols.issubset(price_df.columns):
        data_source = snap.get("meta", {}).get("xauusd_source", "historical-feed")
        if "metals.live+" in data_source:
            history_source = data_source.split("+", 1)[1]
            st.info(f"Live price from metals.live · historical bars from {history_source}")
        st.plotly_chart(price_chart(price_df), use_container_width=True)
    elif "_error" in price_df.columns:
        st.warning(f"Chart unavailable — historical provider fetch failed. Live price shown above when available. Error: {price_df['_error'].iloc[0]}")
    else:
        st.warning("Price chart data not available.")

    # ── 3. Multi-TF ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📐 Multi-Timeframe Analysis")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.plotly_chart(tf_bias_chart(tech), use_container_width=True)
    with col_b:
        rows = []
        for tf in ["Monthly", "Weekly", "Daily", "H4", "H1", "M15", "M5", "M1"]:
            d = tech.get(tf, {})
            if not isinstance(d, dict) or "error" in d:
                continue
            mom = d.get("momentum", {}) or {}
            tr = d.get("trend", {}) or {}
            st_ = d.get("structure", {}) or {}
            rows.append({
                "TF": tf, "Bias": d.get("bias", "N/A").upper(),
                "RSI": round(float(mom.get("rsi", 0) or 0), 1),
                "ADX": round(float(tr.get("adx", 0) or 0), 1),
                "Structure": st_.get("current_structure", "N/A"),
                "BOS": "✓" if st_.get("bos_detected") else "—",
                "CHOCH": "✓" if st_.get("choch_detected") else "—",
                "Conf": f"{float(d.get('confidence', 0) or 0):.0%}",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── 4. Correlation ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔗 Correlation Analysis")
    col_c1, col_c2 = st.columns([2, 1])
    with col_c1:
        st.plotly_chart(correlation_chart(corr), use_container_width=True)
    with col_c2:
        st.markdown(f"**Regime:** `{(corr.get('regime') or 'N/A').upper()}`")
        st.markdown(f"**Primary Driver:** `{corr.get('strongest_influence', 'N/A')}`")
        for w in (corr.get("divergence_warnings") or [])[:3]:
            st.caption(f"⚠️ {w}")

    # ── 5. Macro ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🌍 Macro Fundamentals")
    col_m1, col_m2 = st.columns([1, 2])
    with col_m1:
        st.plotly_chart(macro_gauge(macro), use_container_width=True)
        st.markdown(f"**Inflation:** `{macro.get('inflation_regime','N/A')}`")
        st.markdown(f"**Real Rates:** `{macro.get('real_rate_regime','N/A')}`")
        usd = macro.get("usd_strength_score")
        if usd is not None:
            st.markdown(f"**USD Strength:** `{float(usd):+.2f}`")
    with col_m2:
        macro_rows = [
            {
                "Indicator": m.get("name", k),
                "Current": m.get("current_value"),
                "Previous": m.get("previous_value"),
                "ST Gold": m.get("short_term_gold_impact", "N/A"),
                "LT Gold": m.get("long_term_gold_impact", "N/A"),
            }
            for k, m in list((macro.get("metrics") or {}).items())[:8]
            if isinstance(m, dict)
        ]
        if macro_rows:
            st.dataframe(pd.DataFrame(macro_rows), use_container_width=True, hide_index=True)

    # ── 6. News ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📰 News & Sentiment")
    col_n1, col_n2 = st.columns([1, 2])
    with col_n1:
        st.metric("Sentiment", f"{sent_score:+.3f}")
        vol = float(news.get("news_volume_normalized", 0) or 0)
        st.progress(min(1.0, vol), text=f"Volume {vol:.0%}")
        st.markdown(f"**Retail Bias:** `{(news.get('retail_sentiment_bias') or 'neutral').upper()}`")
        alerts = news.get("high_impact_alerts") or []
        if alerts:
            st.error(f"⚠️ {len(alerts)} high-impact event(s)")
            for a in alerts[:2]:
                if isinstance(a, dict):
                    st.caption(f"• {a.get('event_name')} [{a.get('impact')}]")
    with col_n2:
        direct = [a for a in (news.get("articles") or [])
                  if isinstance(a, dict) and a.get("relevance") == "direct"][:6]
        for a in direct:
            icon = {"bullish": "🟢", "bearish": "🔴"}.get(a.get("gold_impact", ""), "⚪")
            st.markdown(f"{icon} **{a.get('source','')}** — {a.get('title','')[:100]}")

    # ── 7. Session ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⏰ Session Analysis")
    sc = st.columns(3)
    sc[0].metric("Session", session.get("active_session", "N/A"))
    sc[1].metric("Kill Zone", session.get("active_kill_zone") or "None")
    sc[2].metric("Regime", (session.get("volatility_regime") or "N/A").title())
    r_pips = float(session.get("current_range_pips", 0) or 0)
    r_pct = float(session.get("current_range_percentile", 50) or 50)
    st.caption(f"Range: **{r_pips:.2f} pts** ({r_pct:.0f}th pct vs 90-day avg)  ·  {session.get('expected_session_outcome','')}")
    for t in (session.get("session_traps") or []):
        st.warning(f"🪤 {t}")

    # ── 8. Smart money ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏦 Smart Money & Liquidity")
    col_sm1, col_sm2 = st.columns([1, 2])
    with col_sm1:
        st.metric("Phase", sm_phase.title())
        st.metric("Liquidity Score", f"{float(smart.get('engineered_liquidity_score', 0) or 0):.0f}/100")
        vwap = smart.get("vwap")
        if vwap:
            st.metric("VWAP", f"${float(vwap):.2f}", delta=smart.get("price_vs_vwap"))
        for t in (smart.get("trap_warnings") or []):
            st.warning(f"⚠️ {t}")
    with col_sm2:
        if cur_price > 0:
            st.plotly_chart(liquidity_chart(smart, cur_price), use_container_width=True)

    # ── 9. Historical patterns ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📅 Historical Pattern Matching")
    col_p1, col_p2 = st.columns([1, 2])
    with col_p1:
        prob_up = float(patterns.get("probability_of_up_move", 0.5) or 0.5)
        st.metric("P(Up Move)", f"{prob_up:.0%}")
        st.metric("Expected 1D", f"{float(patterns.get('expected_move_1d', 0) or 0):+.2f}%")
        st.metric("Expected 1W", f"{float(patterns.get('expected_move_1w', 0) or 0):+.2f}%")
        st.caption(f"Confidence: {float(patterns.get('pattern_confidence', 0) or 0):.0%}")
    with col_p2:
        analog_rows = [
            {
                "Pattern": a.get("label", "Unknown"),
                "Similarity": f"{float(a.get('similarity_score', 0)):.0%}",
                "1D": f"{float(a.get('outcome_1d', 0)):+.1f}%",
                "1W": f"{float(a.get('outcome_1w', 0)):+.1f}%",
                "Win Rate": f"{float(a.get('win_rate', 0)):.0%}",
            }
            for a in (patterns.get("top_analogs") or []) if isinstance(a, dict)
        ]
        if analog_rows:
            st.dataframe(pd.DataFrame(analog_rows), use_container_width=True, hide_index=True)

    # ── 10. Risk ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚖️ Risk Management")
    rc = st.columns(4)
    rc[0].metric("Risk %", f"{float(risk.get('recommended_risk_percent', 0) or 0):.1f}%")
    rc[1].metric("SL Distance", f"{float(risk.get('adjusted_sl_points', 0) or 0):.2f} pts")
    rc[2].metric("Max Lots", f"{float(risk.get('max_position_size_lots', 0) or 0):.2f}")
    rc[3].metric("Volatility", (risk.get("volatility_regime") or "N/A").title())
    for w in (risk.get("risk_warnings") or []):
        st.warning(f"⚠️ {w}")

    # ── 11. Trade signals ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Trade Signals")
    label = (conf.get("confidence_label") or "low").upper()
    oa = float(conf.get("ensemble_agreement", 0) or 0)
    fn = {"TRADE": st.success, "WAIT": st.warning, "AVOID": st.error}.get(rec, st.info)
    fn(f"**{rec}**  ·  Confidence: {label} ({overall_conf:.0%})  ·  Layer agreement: {oa:.0%}")
    for d in (conf.get("layer_disagreements") or []):
        st.caption(f"⚠️ {d}")
    if signals:
        for sig in signals:
            signal_card(sig)
    else:
        st.info("No high-probability setup detected (requires P > 65%). Monitor key levels.")

    # ── 12. Monte Carlo ───────────────────────────────────────────────────────
    mc = conf.get("monte_carlo")
    if mc and isinstance(mc, dict):
        st.markdown("---")
        st.markdown("### 🎲 Monte Carlo Simulation")
        col_mc1, col_mc2 = st.columns([2, 1])
        with col_mc1:
            st.plotly_chart(monte_carlo_chart(mc), use_container_width=True)
        with col_mc2:
            st.metric("Expected Return", f"{float(mc.get('expected_return_pct', 0)):+.2f}%")
            st.metric("Avg Max Adverse", f"{float(mc.get('max_adverse_excursion_avg', 0)):.2f}%")
            pe = conf.get("price_estimate", {}) or {}
            ci = pe.get("confidence_interval_80")
            if ci and len(ci) == 2:
                st.metric("80% Price Range", f"${ci[0]:.0f} – ${ci[1]:.0f}")
            calib = conf.get("calibration_warning")
            if calib:
                st.warning(f"⚠️ {calib}")

    # ── 13. Gemini Report ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🤖 Gemini AI Full Report")
    if st.session_state.report:
        with st.expander("📄 View Full Institutional Report", expanded=True):
            st.markdown(st.session_state.report)
    else:
        st.info("Report will appear here after analysis.")

    st.markdown("---")
    st.caption("⚠️ Research purposes only. Not financial advice. Past performance does not guarantee future results.")


if __name__ == "__main__":
    main()

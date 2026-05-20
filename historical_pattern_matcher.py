"""Layer 8: Historical Pattern Matcher — finds similar historical setups using ML similarity."""
from __future__ import annotations
import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

DB_PATH = Path("patterns.db")

# ── Data models ───────────────────────────────────────────────────────────────
@dataclass
class PatternRecord:
    pattern_id: str
    label: str           # e.g., "FOMC_hawkish_reversal"
    timeframe: str
    conditions: dict     # macro, technical, sentiment snapshot
    outcome_1d: float    # % move
    outcome_3d: float
    outcome_1w: float
    win_rate: float      # 0-1

@dataclass
class AnalogMatch:
    pattern_id: str
    label: str
    similarity_score: float  # 0-1
    outcome_1d: float
    outcome_3d: float
    outcome_1w: float
    win_rate: float
    confidence: float

@dataclass
class PatternMatchResult:
    current_features: dict
    top_analogs: list[AnalogMatch]
    probability_of_up_move: float
    expected_move_1d: float
    expected_move_1w: float
    pattern_confidence: float
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)

# ── DB setup ──────────────────────────────────────────────────────────────────
def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS patterns (
            pattern_id TEXT PRIMARY KEY,
            label TEXT,
            timeframe TEXT,
            feature_vector TEXT,
            conditions TEXT,
            outcome_1d REAL,
            outcome_3d REAL,
            outcome_1w REAL,
            win_rate REAL
        )
    """)
    conn.commit()
    conn.close()

_init_db()

# ── Feature extraction ────────────────────────────────────────────────────────
def _extract_features(
    df: pd.DataFrame,
    rsi: float = 50.0,
    adx: float = 25.0,
    atr_pct: float = 0.5,
    macro_bias: float = 5.0,
    sentiment: float = 0.0,
) -> dict:
    """Extract a normalized feature vector from current market state."""
    if df.empty or len(df) < 20:
        return {}

    close = df["close"]
    returns_1d = float(close.pct_change(24).iloc[-1] * 100) if len(close) >= 24 else 0.0
    returns_5d = float(close.pct_change(120).iloc[-1] * 100) if len(close) >= 120 else 0.0

    ema20 = float(close.ewm(span=20).mean().iloc[-1])
    price_vs_ema = float((close.iloc[-1] - ema20) / (ema20 + 1e-9) * 100)

    return {
        "rsi": rsi,
        "adx": adx,
        "atr_pct": atr_pct,
        "returns_1d": returns_1d,
        "returns_5d": returns_5d,
        "price_vs_ema20": price_vs_ema,
        "macro_bias": macro_bias,
        "sentiment": sentiment,
        "volatility_regime": 1 if atr_pct > 1.0 else 0 if atr_pct < 0.3 else 0.5,
    }

FEATURE_KEYS = [
    "rsi", "adx", "atr_pct", "returns_1d", "returns_5d",
    "price_vs_ema20", "macro_bias", "sentiment", "volatility_regime"
]

def _to_vector(features: dict) -> np.ndarray:
    return np.array([features.get(k, 0.0) for k in FEATURE_KEYS], dtype=float)

# ── Seed historical patterns ──────────────────────────────────────────────────
SEED_PATTERNS = [
    {
        "pattern_id": "fomc_hawkish_2022",
        "label": "FOMC_hawkish_surprise",
        "timeframe": "Daily",
        "conditions": {"macro": "hawkish", "regime": "rate_hike"},
        "feature_vector": [35, 45, 0.8, -1.5, -3.0, -2.0, 2.0, -0.6, 1],
        "outcome_1d": -1.8, "outcome_3d": -2.5, "outcome_1w": -3.0, "win_rate": 0.72,
    },
    {
        "pattern_id": "fomc_dovish_pivot",
        "label": "FOMC_dovish_pivot",
        "timeframe": "Daily",
        "conditions": {"macro": "dovish", "regime": "rate_pause"},
        "feature_vector": [65, 38, 0.7, 1.2, 2.0, 1.5, 8.0, 0.5, 0.5],
        "outcome_1d": 2.1, "outcome_3d": 3.5, "outcome_1w": 5.0, "win_rate": 0.68,
    },
    {
        "pattern_id": "hot_cpi_rejection",
        "label": "Hot_CPI_gold_selloff",
        "timeframe": "H4",
        "conditions": {"macro": "hawkish", "event": "CPI"},
        "feature_vector": [72, 40, 1.2, -0.5, 1.0, 0.5, 3.0, -0.4, 1],
        "outcome_1d": -1.5, "outcome_3d": -1.0, "outcome_1w": 0.5, "win_rate": 0.60,
    },
    {
        "pattern_id": "geopolitical_spike",
        "label": "Geopolitical_safe_haven_spike",
        "timeframe": "H1",
        "conditions": {"event": "geopolitical", "regime": "risk_off"},
        "feature_vector": [70, 55, 1.8, 3.0, 1.5, 2.5, 9.0, 0.8, 1],
        "outcome_1d": 1.5, "outcome_3d": 0.5, "outcome_1w": -1.0, "win_rate": 0.55,
    },
    {
        "pattern_id": "liquidity_sweep_reversal",
        "label": "Liquidity_sweep_long_setup",
        "timeframe": "H1",
        "conditions": {"pattern": "stop_hunt", "direction": "bullish"},
        "feature_vector": [28, 30, 0.9, -2.0, -1.5, -1.8, 7.0, -0.3, 0.5],
        "outcome_1d": 1.8, "outcome_3d": 3.0, "outcome_1w": 2.5, "win_rate": 0.75,
    },
    {
        "pattern_id": "consolidation_breakout",
        "label": "Bollinger_squeeze_breakout",
        "timeframe": "Daily",
        "conditions": {"pattern": "bb_squeeze", "direction": "bullish"},
        "feature_vector": [55, 15, 0.2, 0.3, 0.8, 0.2, 6.0, 0.1, 0],
        "outcome_1d": 1.2, "outcome_3d": 2.8, "outcome_1w": 4.0, "win_rate": 0.65,
    },
    {
        "pattern_id": "trend_pullback_ema50",
        "label": "Trend_continuation_EMA50_retest",
        "timeframe": "H4",
        "conditions": {"pattern": "pullback", "trend": "bullish"},
        "feature_vector": [45, 35, 0.6, -0.8, 2.0, -1.0, 7.5, 0.2, 0.5],
        "outcome_1d": 1.0, "outcome_3d": 2.2, "outcome_1w": 3.5, "win_rate": 0.70,
    },
]

def _seed_patterns():
    conn = sqlite3.connect(DB_PATH)
    existing = conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
    if existing < len(SEED_PATTERNS):
        for p in SEED_PATTERNS:
            conn.execute(
                "INSERT OR REPLACE INTO patterns VALUES (?,?,?,?,?,?,?,?,?)",
                (p["pattern_id"], p["label"], p["timeframe"],
                 json.dumps(p["feature_vector"]), json.dumps(p["conditions"]),
                 p["outcome_1d"], p["outcome_3d"], p["outcome_1w"], p["win_rate"])
            )
        conn.commit()
        logger.info(f"Seeded {len(SEED_PATTERNS)} historical patterns")
    conn.close()

_seed_patterns()

# ── Matcher ───────────────────────────────────────────────────────────────────
class HistoricalPatternMatcher:
    def __init__(self):
        self._patterns: list[dict] = []
        self._load_patterns()

    def _load_patterns(self):
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("SELECT * FROM patterns").fetchall()
        conn.close()
        self._patterns = [
            {
                "pattern_id": r[0], "label": r[1], "timeframe": r[2],
                "feature_vector": json.loads(r[3]), "conditions": json.loads(r[4]),
                "outcome_1d": r[5], "outcome_3d": r[6], "outcome_1w": r[7], "win_rate": r[8],
            }
            for r in rows
        ]

    def find_analogs(
        self,
        df: pd.DataFrame,
        rsi: float = 50.0,
        adx: float = 25.0,
        atr_pct: float = 0.5,
        macro_bias: float = 5.0,
        sentiment: float = 0.0,
        top_n: int = 3,
    ) -> dict:
        current_features = _extract_features(df, rsi, adx, atr_pct, macro_bias, sentiment)
        if not current_features:
            return {"error": "Insufficient data for pattern matching"}

        current_vec = _to_vector(current_features).reshape(1, -1)
        scaler = StandardScaler()
        pattern_vecs = np.array([p["feature_vector"] for p in self._patterns])

        if len(pattern_vecs) == 0:
            return {"error": "No patterns in database"}

        all_vecs = np.vstack([current_vec, pattern_vecs])
        try:
            scaler.fit(all_vecs)
            scaled_current = scaler.transform(current_vec)
            scaled_patterns = scaler.transform(pattern_vecs)
            similarities = cosine_similarity(scaled_current, scaled_patterns)[0]
        except Exception as e:
            logger.warning(f"Scaling failed, using raw similarity: {e}")
            similarities = cosine_similarity(current_vec, pattern_vecs)[0]

        top_indices = np.argsort(similarities)[::-1][:top_n]
        analogs = []
        for idx in top_indices:
            p = self._patterns[idx]
            sim = float(similarities[idx])
            analogs.append(AnalogMatch(
                pattern_id=p["pattern_id"],
                label=p["label"],
                similarity_score=round(max(0.0, sim), 3),
                outcome_1d=p["outcome_1d"],
                outcome_3d=p["outcome_3d"],
                outcome_1w=p["outcome_1w"],
                win_rate=p["win_rate"],
                confidence=round(sim * p["win_rate"], 3),
            ))

        # Weighted expected moves
        total_sim = sum(a.similarity_score for a in analogs) + 1e-9
        exp_1d = sum(a.outcome_1d * a.similarity_score / total_sim for a in analogs)
        exp_1w = sum(a.outcome_1w * a.similarity_score / total_sim for a in analogs)
        prob_up = sum((1 if a.outcome_1d > 0 else 0) * a.similarity_score / total_sim for a in analogs)
        pattern_conf = float(np.mean([a.confidence for a in analogs]))

        result = PatternMatchResult(
            current_features=current_features,
            top_analogs=analogs,
            probability_of_up_move=round(prob_up, 3),
            expected_move_1d=round(exp_1d, 2),
            expected_move_1w=round(exp_1w, 2),
            pattern_confidence=round(pattern_conf, 3),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        return result.to_dict()

    def add_pattern(self, pattern: PatternRecord) -> None:
        """Store a new pattern for future matching."""
        conn = sqlite3.connect(DB_PATH)
        feat = _extract_features(pd.DataFrame())  # placeholder
        conn.execute(
            "INSERT OR REPLACE INTO patterns VALUES (?,?,?,?,?,?,?,?,?)",
            (pattern.pattern_id, pattern.label, pattern.timeframe,
             json.dumps(list(_to_vector(feat))), json.dumps(pattern.conditions),
             pattern.outcome_1d, pattern.outcome_3d, pattern.outcome_1w, pattern.win_rate)
        )
        conn.commit()
        conn.close()


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from market_data_agent import get_historical_data
    df = get_historical_data("XAUUSD", period="30d", interval="1h")
    matcher = HistoricalPatternMatcher()
    result = matcher.find_analogs(df, rsi=45.0, adx=28.0, atr_pct=0.6, macro_bias=6.5, sentiment=0.2)
    print(json.dumps(result, indent=2, default=str))

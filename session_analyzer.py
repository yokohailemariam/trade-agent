"""Layer 5: Session Behavior Analyzer — characterizes trading session behavior for XAUUSD."""
from __future__ import annotations
import sqlite3
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from loguru import logger

DB_PATH = Path("session_data.db")

SESSIONS = {
    "Asia":               (0, 9),
    "London":             (8, 17),
    "NewYork":            (13, 22),
    "LondonNY_Overlap":   (13, 17),
    "AsiaLondon_Overlap": (8, 9),
}

KILL_ZONES = {
    "Asia_Kill":   (0, 3),
    "London_Kill": (8, 10),
    "NY_Kill":     (13, 15),
    "NY_Close":    (20, 22),
}


@dataclass
class SessionStats:
    session_name: str
    avg_range_pips: float
    typical_direction_bias: str
    manipulation_window_minutes: int
    reversal_hour_utc: int
    kill_zone_times: list[str]
    liquidity_grab_pattern: str


@dataclass
class CurrentSessionReport:
    active_session: str
    active_kill_zone: Optional[str]
    current_range_pips: float
    current_range_percentile: float
    volatility_regime: str
    expected_session_outcome: str
    session_traps: list[str]
    confidence: float
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_ranges (
            date TEXT, session TEXT, range_pips REAL, direction TEXT
        )
    """)
    conn.commit()
    conn.close()


_init_db()


def _current_session(dt: datetime) -> str:
    h = dt.hour
    if 13 <= h < 17:
        return "LondonNY_Overlap"
    if 8 <= h < 9:
        return "AsiaLondon_Overlap"
    if 0 <= h < 9:
        return "Asia"
    if 8 <= h < 17:
        return "London"
    if 13 <= h < 22:
        return "NewYork"
    return "OffHours"


def _current_kill_zone(dt: datetime) -> Optional[str]:
    h = dt.hour
    for name, (start, end) in KILL_ZONES.items():
        if start <= h < end:
            return name
    return None


def _range_pips(df_session: pd.DataFrame) -> float:
    if df_session.empty:
        return 0.0
    return round(float(df_session["high"].max() - df_session["low"].min()), 2)


def _compute_direction(df_session: pd.DataFrame) -> str:
    if df_session.empty:
        return "mixed"
    open_p = float(df_session["open"].iloc[0])
    close_p = float(df_session["close"].iloc[-1])
    pct = (close_p - open_p) / (open_p + 1e-9) * 100
    if pct > 0.1:
        return "bullish"
    if pct < -0.1:
        return "bearish"
    return "mixed"


def _store_session_range(date: str, session: str, range_pips: float, direction: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO session_ranges VALUES (?,?,?,?)",
        (date, session, range_pips, direction)
    )
    conn.commit()
    conn.close()


def _get_historical_ranges(session: str, days: int = 90) -> list[float]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT range_pips FROM session_ranges WHERE session=? ORDER BY date DESC LIMIT ?",
        (session, days)
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def _percentile(value: float, history: list[float]) -> float:
    if not history:
        return 50.0
    sorted_h = sorted(history)
    pos = sum(1 for h in sorted_h if h <= value)
    return round(pos / len(sorted_h) * 100, 1)


def _volatility_regime(current_range: float, avg_range: float) -> str:
    ratio = current_range / (avg_range + 1e-9)
    if ratio > 1.3:
        return "expanding"
    if ratio < 0.7:
        return "squeeze"
    return "contracting"


def _detect_traps(session: str, df_session: pd.DataFrame) -> list[str]:
    traps = []
    if session == "Asia" and not df_session.empty:
        traps.append("Asian_range_break_fakeout risk during first 30 min")
    if session in ("London", "AsiaLondon_Overlap"):
        traps.append("London_open_reversal — first push often reversed")
    if session == "LondonNY_Overlap":
        traps.append("NY_London_correction — NY may correct London direction")
    if not df_session.empty:
        r = _range_pips(df_session)
        if r > 0:
            wicks = (df_session["high"] - df_session[["close", "open"]].max(axis=1)).mean()
            if wicks > 0.3 * r:
                traps.append("Stop_hunt detected — large wicks suggest liquidity sweeps")
    return traps


class SessionAnalyzer:
    def analyze(self, df: pd.DataFrame) -> dict:
        now = datetime.now(timezone.utc)
        active = _current_session(now)
        kill_zone = _current_kill_zone(now)

        today = now.date()
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            today_df = df[df.index.date == today]
        else:
            today_df = df.tail(24)

        start_h, end_h = SESSIONS.get(active, (0, 24))
        if not today_df.empty:
            session_df = today_df[
                (today_df.index.hour >= start_h) & (today_df.index.hour < end_h)
            ]
        else:
            session_df = pd.DataFrame()

        current_range = _range_pips(session_df)
        direction = _compute_direction(session_df)
        _store_session_range(str(today), active, current_range, direction)

        history = _get_historical_ranges(active, days=90)
        avg_range = float(np.mean(history)) if history else current_range
        pct = _percentile(current_range, history)
        vol_regime = _volatility_regime(current_range, avg_range)

        outcome_map = {
            "expanding":   "High volatility — breakout likely. Wait for pullback entry.",
            "contracting": "Normal session — trend continuation likely.",
            "squeeze":     "Low volatility — consolidation. Expect breakout soon.",
        }

        traps = _detect_traps(active, session_df)
        confidence = 0.7 if len(history) > 30 else 0.4

        return CurrentSessionReport(
            active_session=active,
            active_kill_zone=kill_zone,
            current_range_pips=current_range,
            current_range_percentile=pct,
            volatility_regime=vol_regime,
            expected_session_outcome=outcome_map[vol_regime],
            session_traps=traps,
            confidence=confidence,
            timestamp=now.isoformat(),
        ).to_dict()

    def get_session_stats(self) -> dict[str, dict]:
        stats = {}
        for session in SESSIONS:
            history = _get_historical_ranges(session, 90)
            avg_range = round(float(np.mean(history)), 2) if history else 0.0
            kz_times = [f"{s}:00-{e}:00 UTC" for name, (s, e) in KILL_ZONES.items()]
            stats[session] = asdict(SessionStats(
                session_name=session,
                avg_range_pips=avg_range,
                typical_direction_bias="mixed",
                manipulation_window_minutes=15 if session in ("London", "NewYork") else 30,
                reversal_hour_utc={"Asia": 6, "London": 13, "NewYork": 18}.get(session, 12),
                kill_zone_times=kz_times,
                liquidity_grab_pattern="price spike to sweep stops before reversal",
            ))
        return stats


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from market_data_agent import get_historical_data
    df = get_historical_data("XAUUSD", period="7d", interval="1h")
    analyzer = SessionAnalyzer()
    print(json.dumps(analyzer.analyze(df), indent=2))

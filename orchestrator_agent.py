"""Layer 10: Orchestrator Agent — coordinates all layers and generates full analysis."""
from __future__ import annotations
import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from market_data_agent import get_current_market_snapshot, get_historical_data, get_multi_timeframe_data
from technical_analysis_agent import analyze_all_timeframes
from fundamental_agent import FundamentalAgent
from news_sentiment_agent import NewsAnalyzer
from session_analyzer import SessionAnalyzer
from correlation_agent import CorrelationAnalyzer
from smart_money_agent import SmartMoneyAnalyzer
from historical_pattern_matcher import HistoricalPatternMatcher
from risk_manager import RiskManager


class XAUUSDAnalysisOrchestrator:
    def __init__(self, account_balance: float = 10000.0, fred_api_key: str = "", news_api_key: str = ""):
        self.fundamental_agent = FundamentalAgent(fred_api_key=fred_api_key)
        self.news_analyzer = NewsAnalyzer(news_api_key=news_api_key)
        self.session_analyzer = SessionAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.smart_money = SmartMoneyAnalyzer()
        self.pattern_matcher = HistoricalPatternMatcher()
        self.risk_manager = RiskManager(account_balance=account_balance)

    async def generate_full_analysis(self) -> dict[str, Any]:
        logger.info("Starting full XAUUSD analysis...")
        start_time = datetime.now(timezone.utc)
        loop = asyncio.get_event_loop()

        # ── Layer 1: Market snapshot ──────────────────────────────────────
        try:
            snapshot = await get_current_market_snapshot()
            logger.info("Layer 1 complete: market snapshot fetched")
        except Exception as e:
            logger.error(f"Layer 1 failed: {e}")
            snapshot = {"error": str(e)}

        # ── Historical data for multiple layers ───────────────────────────
        try:
            gold_df = await loop.run_in_executor(
                None, get_historical_data, "XAUUSD", "60d", "1h"
            )
            tf_data = await loop.run_in_executor(
                None, get_multi_timeframe_data, "XAUUSD"
            )
            corr_syms = ["DXY", "US10Y", "EURUSD", "GBPUSD", "USDJPY",
                         "XAGUSD", "WTI", "SPX", "VIX", "BTCUSD"]
            correlated_tasks = {
                sym: loop.run_in_executor(None, get_historical_data, sym, "30d", "1h")
                for sym in corr_syms
            }
            correlated_dfs = {}
            for sym, task in correlated_tasks.items():
                try:
                    correlated_dfs[sym] = await task
                except Exception as e:
                    logger.warning(f"Correlated data fetch failed for {sym}: {e}")
        except Exception as e:
            logger.error(f"Historical data fetch failed: {e}")
            gold_df = None
            tf_data = {}
            correlated_dfs = {}

        import pandas as pd

        def _df_ok(df) -> bool:
            return df is not None and isinstance(df, pd.DataFrame) and not df.empty

        # ── Layer 2: Technical analysis ───────────────────────────────────
        async def _tech():
            if not tf_data:
                return {"error": "no timeframe data"}
            return await loop.run_in_executor(None, analyze_all_timeframes, tf_data)

        # ── Layer 3: Macro fundamentals ───────────────────────────────────
        async def _macro():
            return await loop.run_in_executor(None, self.fundamental_agent.get_impact_summary)

        # ── Layer 4: News sentiment ───────────────────────────────────────
        async def _news():
            return await loop.run_in_executor(None, self.news_analyzer.get_impact_summary)

        # ── Layer 5: Session analysis ─────────────────────────────────────
        async def _session():
            if not _df_ok(gold_df):
                return {"error": "no data"}
            return await loop.run_in_executor(None, self.session_analyzer.analyze, gold_df)

        # ── Layer 6: Correlation ──────────────────────────────────────────
        async def _correlation():
            def _run():
                if _df_ok(gold_df):
                    self.correlation_analyzer.update_data(gold_df, correlated_dfs)
                return self.correlation_analyzer.analyze()
            return await loop.run_in_executor(None, _run)

        # ── Layer 7: Smart money ──────────────────────────────────────────
        async def _smart_money():
            if not _df_ok(gold_df):
                return {"error": "no data"}
            return await loop.run_in_executor(None, self.smart_money.analyze, gold_df)

        # Run layers 2-7 in parallel
        (
            technical_data,
            macro_data,
            news_data,
            session_data,
            correlation_data,
            smart_money_data,
        ) = await asyncio.gather(
            _tech(), _macro(), _news(), _session(), _correlation(), _smart_money(),
        )
        logger.info("Layers 2-7 complete")

        # ── Layer 8: Pattern matching ─────────────────────────────────────
        try:
            h1_tech = technical_data.get("H1", {}) if isinstance(technical_data, dict) else {}
            rsi_val = h1_tech.get("momentum", {}).get("rsi", 50.0) if isinstance(h1_tech, dict) else 50.0
            adx_val = h1_tech.get("trend", {}).get("adx", 25.0) if isinstance(h1_tech, dict) else 25.0
            atr_pct = h1_tech.get("volatility", {}).get("atr_pct", 0.5) if isinstance(h1_tech, dict) else 0.5
            macro_bias = macro_data.get("gold_macro_bias", 5.0) if isinstance(macro_data, dict) else 5.0
            news_sent = news_data.get("aggregated_sentiment", 0.0) if isinstance(news_data, dict) else 0.0

            pattern_data = await loop.run_in_executor(
                None,
                lambda: self.pattern_matcher.find_analogs(
                    gold_df, rsi=rsi_val, adx=adx_val, atr_pct=atr_pct,
                    macro_bias=macro_bias, sentiment=news_sent,
                ) if _df_ok(gold_df) else {"error": "no data"}
            )
            logger.info("Layer 8 complete (Pattern Matching)")
        except Exception as e:
            logger.error(f"Layer 8 failed: {e}")
            pattern_data = {"error": str(e)}

        # ── Layer 9: Risk assessment ──────────────────────────────────────
        try:
            has_news_risk = bool(
                isinstance(news_data, dict) and len(news_data.get("high_impact_alerts", [])) > 0
            )
            risk_data = await loop.run_in_executor(
                None,
                lambda: self.risk_manager.assess_trade_risk(
                    gold_df, high_impact_news_in_2h=has_news_risk,
                ) if _df_ok(gold_df) else {"error": "no data"}
            )
            logger.info("Layer 9 complete (Risk)")
        except Exception as e:
            logger.error(f"Layer 9 failed: {e}")
            risk_data = {"error": str(e)}

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info(f"Full analysis complete in {elapsed:.1f}s")

        return {
            "meta": {
                "analysis_timestamp": start_time.isoformat(),
                "elapsed_seconds": round(elapsed, 1),
                "symbol": "XAUUSD",
            },
            "market_snapshot":     snapshot,
            "technical_analysis":  technical_data,
            "macro_fundamentals":  macro_data,
            "news_sentiment":      news_data,
            "session_analysis":    session_data,
            "correlation":         correlation_data,
            "smart_money":         smart_money_data,
            "historical_patterns": pattern_data,
            "risk_assessment":     risk_data,
        }


if __name__ == "__main__":
    async def _test():
        orchestrator = XAUUSDAnalysisOrchestrator()
        result = await orchestrator.generate_full_analysis()
        print(json.dumps(result, indent=2, default=str))

    asyncio.run(_test())

#!/usr/bin/env python3
"""Main entry point for the XAUUSD Trading Intelligence System."""
import asyncio
import json
from loguru import logger
from orchestrator_agent import XAUUSDAnalysisOrchestrator
from llm_interface import LLMFormatter


async def main():
    logger.info("Starting XAUUSD Trading Intelligence System...")
    orchestrator = XAUUSDAnalysisOrchestrator()

    try:
        analysis_data = await orchestrator.generate_full_analysis()
        formatter = LLMFormatter()
        report = await formatter.format_analysis(analysis_data)
        print(report)

        with open("latest_analysis.json", "w") as f:
            json.dump(analysis_data, f, indent=2, default=str)
        logger.info("Analysis complete. Saved to latest_analysis.json")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())

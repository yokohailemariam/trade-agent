import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class Config(BaseModel):
    alpha_vantage_key: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    currencyfreaks_api_key: str = os.getenv("CURRENCYFREAKS_API_KEY", "")
    fred_api_key: str = os.getenv("FRED_API_KEY", "")
    news_api_key: str = os.getenv("NEWS_API_KEY", "")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    base_risk_percent: float = 1.0
    max_risk_percent: float = 2.0
    min_signal_probability: float = 0.65

    primary_symbol: str = "XAUUSD"
    correlated_symbols: list = [
        "DX-Y.NYB",
        "^TNX",
        "EURUSD=X",
        "GBPUSD=X",
        "USDJPY=X",
        "SI=F",
        "CL=F",
        "^GSPC",
        "^NDX",
        "^VIX",
        "BTC-USD",
    ]
    timeframes: list = ["1mo", "1wk", "1d", "4h", "1h", "15m", "5m", "1m"]


config = Config()

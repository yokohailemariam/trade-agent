"""Layer 4: News & Sentiment Analyzer — news processing, classification, market impact."""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

import feedparser
import requests
from loguru import logger
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

RSS_FEEDS = [
    ("Reuters", "https://feeds.reuters.com/reuters/businessNews"),
    ("ForexLive", "https://www.forexlive.com/feed/news"),
    ("MarketWatch", "https://feeds.marketwatch.com/marketwatch/marketpulse/"),
]

GOLD_KEYWORDS = [
    "gold", "xauusd", "xau", "precious metal", "bullion",
    "fed", "federal reserve", "fomc", "powell", "inflation", "cpi", "ppi",
    "nfp", "jobs", "unemployment", "gdp", "rate hike", "rate cut",
    "dollar", "usd", "treasury", "yield", "geopolit", "war", "crisis",
    "central bank", "sanctions", "risk",
]

CATEGORY_MAP = {
    "Fed":         ["fed", "fomc", "powell", "federal reserve", "rate hike", "rate cut", "monetary policy"],
    "Inflation":   ["cpi", "ppi", "inflation", "deflation", "price", "consumer price"],
    "Jobs":        ["nfp", "jobs", "unemployment", "payroll", "employment", "labor"],
    "Geopolitics": ["war", "conflict", "sanctions", "geopolit", "military", "attack", "crisis", "tension"],
    "Dollar":      ["dollar", "usd", "dxy", "greenback", "currency"],
    "Risk":        ["risk", "recession", "vix", "safe haven", "panic", "crash", "bank"],
    "Gold":        ["gold", "xau", "bullion", "precious metal", "silver", "platinum"],
}

SCHEDULED_EVENTS = [
    {"name": "FOMC Meeting",   "pattern": "fomc",         "impact": "high"},
    {"name": "CPI Release",    "pattern": "cpi",          "impact": "high"},
    {"name": "NFP Release",    "pattern": "nfp",          "impact": "high"},
    {"name": "Powell Speech",  "pattern": "powell",       "impact": "high"},
    {"name": "Interest Rate",  "pattern": "rate decision","impact": "high"},
    {"name": "GDP Release",    "pattern": "gdp",          "impact": "medium"},
    {"name": "PPI Release",    "pattern": "ppi",          "impact": "medium"},
]

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)


def _cache_get(key: str, ttl: int) -> Optional[dict]:
    p = CACHE_DIR / f"news_{key}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d["v"] if time.time() - d.get("t", 0) < ttl else None


def _cache_set(key: str, v: dict) -> None:
    (CACHE_DIR / f"news_{key}.json").write_text(json.dumps({"t": time.time(), "v": v}))


@dataclass
class NewsArticle:
    title: str
    source: str
    published: str
    url: str
    relevance: str
    category: str
    sentiment_score: float
    gold_impact: str
    urgency: str
    confidence: float


@dataclass
class NewsAlert:
    event_name: str
    impact: str
    detected_at: str
    is_upcoming: bool


@dataclass
class NewsSummary:
    articles: list[NewsArticle]
    aggregated_sentiment: float
    news_volume_normalized: float
    high_impact_alerts: list[NewsAlert]
    fakeout_warnings: list[str]
    fear_greed_score: Optional[float]
    retail_sentiment_bias: str
    analysis_timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


_vader = SentimentIntensityAnalyzer()


def _sentiment(text: str) -> float:
    return round(_vader.polarity_scores(text)["compound"], 3)


def _is_relevant(title: str, summary: str) -> str:
    text = (title + " " + summary).lower()
    matches = sum(1 for kw in GOLD_KEYWORDS if kw in text)
    if matches >= 3:
        return "direct"
    if matches >= 1:
        return "indirect"
    return "ignore"


def _categorize(text: str) -> str:
    text_lower = text.lower()
    for category, keywords in CATEGORY_MAP.items():
        if any(kw in text_lower for kw in keywords):
            return category
    return "Other"


def _gold_impact_from_sentiment(sentiment: float, category: str) -> str:
    if category in ("Fed", "Dollar"):
        if sentiment < -0.2:
            return "bullish"
        if sentiment > 0.2:
            return "bearish"
        return "neutral"
    elif category == "Geopolitics":
        return "bullish" if abs(sentiment) > 0.1 else "neutral"
    elif category == "Gold":
        return "bullish" if sentiment > 0.1 else "bearish" if sentiment < -0.1 else "neutral"
    elif category == "Inflation":
        return "bullish" if sentiment > 0.1 else "neutral"
    return "neutral"


def _urgency(category: str, sentiment: float) -> str:
    if category in ("Fed", "Geopolitics") and abs(sentiment) > 0.5:
        return "high"
    if category in ("Inflation", "Jobs", "Gold") and abs(sentiment) > 0.3:
        return "medium"
    return "low"


def _detect_scheduled(title: str) -> Optional[NewsAlert]:
    title_lower = title.lower()
    for ev in SCHEDULED_EVENTS:
        if ev["pattern"] in title_lower:
            return NewsAlert(
                event_name=ev["name"],
                impact=ev["impact"],
                detected_at=datetime.now(timezone.utc).isoformat(),
                is_upcoming="upcoming" in title_lower or "preview" in title_lower,
            )
    return None


def _fetch_fear_greed() -> Optional[float]:
    cached = _cache_get("fng", ttl=3600)
    if cached:
        return cached.get("value")
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
        val = float(r.json()["data"][0]["value"])
        _cache_set("fng", {"value": val})
        return val
    except Exception as e:
        logger.warning(f"Fear & Greed fetch failed: {e}")
        return None


class NewsAnalyzer:
    def __init__(self, news_api_key: str = ""):
        self.news_api_key = news_api_key

    def _fetch_rss(self) -> list[dict]:
        cached = _cache_get("rss_articles", ttl=600)
        if cached:
            return cached
        articles = []
        for source, url in RSS_FEEDS:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:10]:
                    articles.append({
                        "title": entry.get("title", ""),
                        "summary": entry.get("summary", ""),
                        "source": source,
                        "published": entry.get("published", datetime.now(timezone.utc).isoformat()),
                        "url": entry.get("link", ""),
                    })
            except Exception as e:
                logger.warning(f"RSS fetch failed ({source}): {e}")
        _cache_set("rss_articles", articles)
        return articles

    def _fetch_newsapi(self, query: str = "gold XAUUSD Federal Reserve") -> list[dict]:
        if not self.news_api_key:
            return []
        cached = _cache_get("newsapi", ttl=600)
        if cached:
            return cached
        try:
            r = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query, "language": "en", "sortBy": "publishedAt",
                    "pageSize": 20, "apiKey": self.news_api_key,
                },
                timeout=10,
            )
            articles = [
                {
                    "title": a.get("title", ""),
                    "summary": a.get("description", ""),
                    "source": a.get("source", {}).get("name", "NewsAPI"),
                    "published": a.get("publishedAt", ""),
                    "url": a.get("url", ""),
                }
                for a in r.json().get("articles", [])
            ]
            _cache_set("newsapi", articles)
            return articles
        except Exception as e:
            logger.warning(f"NewsAPI fetch failed: {e}")
            return []

    def get_impact_summary(self) -> dict:
        raw = self._fetch_rss() + self._fetch_newsapi()
        parsed: list[NewsArticle] = []
        alerts: list[NewsAlert] = []
        sentiments: list[float] = []
        fakeout_warnings: list[str] = []

        for raw_a in raw:
            title = raw_a.get("title", "")
            summary = raw_a.get("summary", "")
            relevance = _is_relevant(title, summary)
            if relevance == "ignore":
                continue

            sentiment = _sentiment(title + " " + summary)
            category = _categorize(title + " " + summary)
            gold_impact = _gold_impact_from_sentiment(sentiment, category)
            urgency = _urgency(category, sentiment)
            conf = min(10.0, max(1.0, abs(sentiment) * 10 + (3 if relevance == "direct" else 1)))

            parsed.append(NewsArticle(
                title=title[:200], source=raw_a.get("source", ""),
                published=raw_a.get("published", ""), url=raw_a.get("url", ""),
                relevance=relevance, category=category, sentiment_score=sentiment,
                gold_impact=gold_impact, urgency=urgency, confidence=round(conf, 1),
            ))
            sentiments.append(sentiment)

            alert = _detect_scheduled(title)
            if alert and urgency == "high":
                alerts.append(alert)

            if "gold rises" in title.lower() and sentiment < -0.3:
                fakeout_warnings.append(f"Possible fakeout: '{title[:80]}' — sentiment contradicts price action")

        agg_sentiment = round(sum(sentiments) / len(sentiments), 3) if sentiments else 0.0
        volume_norm = min(1.0, len(parsed) / 30.0)
        fng = _fetch_fear_greed()
        retail_bias = "bearish" if (fng or 50) < 35 else "bullish" if (fng or 50) > 65 else "neutral"

        return NewsSummary(
            articles=parsed[:20],
            aggregated_sentiment=agg_sentiment,
            news_volume_normalized=round(volume_norm, 2),
            high_impact_alerts=alerts,
            fakeout_warnings=fakeout_warnings[:5],
            fear_greed_score=fng,
            retail_sentiment_bias=retail_bias,
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
        ).to_dict()


if __name__ == "__main__":
    analyzer = NewsAnalyzer()
    print(json.dumps(analyzer.get_impact_summary(), indent=2, default=str))

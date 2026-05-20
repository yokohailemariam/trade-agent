"""Layer 1: Market Data Agent — fetches and normalizes XAUUSD and correlated market data."""
from __future__ import annotations
import asyncio
import io as _io
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import pandas as pd
import requests
from loguru import logger

from config import config


class TradingSession(str, Enum):
    ASIA = "Asia"
    LONDON = "London"
    NY = "NewYork"
    LONDON_NY_OVERLAP = "London-NY Overlap"
    ASIA_LONDON_OVERLAP = "Asia-London Overlap"
    OFF = "Off-Hours"


@dataclass
class OHLCV:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    session_label: TradingSession = TradingSession.OFF


@dataclass
class MarketSnapshot:
    timestamp: datetime
    primary: OHLCV
    correlated: dict[str, OHLCV] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        def _fix(d: dict) -> dict:
            return {
                **d,
                "timestamp": d["timestamp"].isoformat() if isinstance(d.get("timestamp"), datetime) else str(d.get("timestamp", "")),
                "session_label": d["session_label"].value if isinstance(d.get("session_label"), TradingSession) else str(d.get("session_label", "")),
            }
        return {
            "timestamp": self.timestamp.isoformat(),
            "primary": _fix(asdict(self.primary)),
            "correlated": {k: _fix(asdict(v)) for k, v in self.correlated.items()},
            "meta": self.meta,
        }


def get_session_label(dt: datetime) -> TradingSession:
    hour = dt.hour
    if 8 <= hour < 9:
        return TradingSession.ASIA_LONDON_OVERLAP
    if 13 <= hour < 17:
        return TradingSession.LONDON_NY_OVERLAP
    if 0 <= hour < 9:
        return TradingSession.ASIA
    if 8 <= hour < 17:
        return TradingSession.LONDON
    if 13 <= hour < 22:
        return TradingSession.NY
    return TradingSession.OFF


CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def cache_get(key: str, ttl_seconds: int = 60) -> Optional[dict]:
    p = _cache_path(key)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if time.time() - data.get("_ts", 0) > ttl_seconds:
        return None
    return data.get("payload")


def cache_set(key: str, payload: dict) -> None:
    _cache_path(key).write_text(json.dumps({"_ts": time.time(), "payload": payload}))


SYMBOL_MAP = {
    "XAUUSD": "GC=F",
    "DXY":    "DX-Y.NYB",
    "US10Y":  "^TNX",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "XAGUSD": "SI=F",
    "WTI":    "CL=F",
    "SPX":    "^GSPC",
    "NDX":    "^NDX",
    "VIX":    "^VIX",
    "BTCUSD": "BTC-USD",
}

# Stooq free CSV historical data — daily/weekly/monthly
STOOQ_SYMBOL_MAP = {
    "XAUUSD": "xauusd",
    "DXY":    "dx.f",
    "US10Y":  "10yt.b",
    "EURUSD": "eurusd",
    "GBPUSD": "gbpusd",
    "USDJPY": "usdjpy",
    "XAGUSD": "xagusd",
    "WTI":    "cl.f",
    "SPX":    "^spx",
    "NDX":    "^ndq",
    "VIX":    "vix.cboe",
    "BTCUSD": "btcusd",
}

# Binance spot symbols for intraday crypto/metals
BINANCE_SYMBOL_MAP = {
    "XAUUSD": "PAXGUSDT",
    "XAGUSD": "PAXGUSDT",
    "BTCUSD": "BTCUSDT",
}

BINANCE_KLINE_INTERVALS = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1wk": "1w",
    "1mo": "1M",
}

INTERVAL_TO_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "1wk": 10080,
    "1mo": 43200,
}


def _period_to_bars(period: str, interval: str, max_bars: int = 1000) -> int:
    unit = period[-1].lower()
    amount = int(period[:-1])
    interval_minutes = INTERVAL_TO_MINUTES.get(interval, 60)
    period_minutes = {
        "d": amount * 1440,
        "w": amount * 10080,
        "m": amount * 43200,
        "y": amount * 525600,
    }.get(unit, 1440)
    bars = max(1, period_minutes // interval_minutes)
    return min(max_bars, bars + 5)


def _normalize_history_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy().sort_index()
    out.index = pd.to_datetime(out.index, utc=True)
    return out[["open", "high", "low", "close", "volume"]].dropna()




def _fetch_alpha_vantage_treasury(period: str = "5d", interval: str = "1h") -> pd.DataFrame:
    api_key = config.alpha_vantage_key
    if not api_key or api_key == "your_key_here":
        raise ValueError("ALPHA_VANTAGE_API_KEY is not configured")

    av_interval = "daily" if interval in {"1m", "5m", "15m", "1h", "4h", "1d"} else "weekly" if interval == "1wk" else "monthly"
    resp = requests.get(
        f"https://www.alphavantage.co/query?{urlencode({'function': 'TREASURY_YIELD', 'interval': av_interval, 'maturity': '10year', 'apikey': api_key})}",
        timeout=10,
        headers={"User-Agent": "trade-agent/1.0"},
    )
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data", [])
    if not data:
        note = payload.get("Note") or payload.get("Information") or payload.get("Error Message")
        raise ValueError(note or "Alpha Vantage returned no treasury yield data")

    bars = _period_to_bars(period, interval, max_bars=5000)
    rows = []
    for item in data:
        raw_value = item.get("value")
        if raw_value in {None, "."}:
            continue
        value = float(raw_value)
        ts = pd.Timestamp(item["date"], tz="UTC")
        rows.append({
            "timestamp": ts,
            "open": value,
            "high": value,
            "low": value,
            "close": value,
            "volume": 0.0,
        })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("timestamp")
    df = df.tail(min(len(df), max(60, bars)))
    df = df.set_index("timestamp")

    if interval in {"1m", "5m", "15m", "1h", "4h"}:
        rule = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h"}[interval]
        end_ts = pd.Timestamp.now(tz="UTC").floor(rule)
        df = df.resample(rule).ffill()
        df = df[df.index <= end_ts]
        df = df.tail(min(len(df), max(60, bars)))

    return _normalize_history_frame(df)


def _period_to_date_range(period: str) -> tuple[str, str]:
    unit = period[-1].lower()
    amount = int(period[:-1])
    to_dt = datetime.now(timezone.utc)
    delta = {
        "d": timedelta(days=amount),
        "w": timedelta(weeks=amount),
        "m": timedelta(days=amount * 30),
        "y": timedelta(days=amount * 365),
    }.get(unit, timedelta(days=amount))
    return (to_dt - delta).strftime("%Y%m%d"), to_dt.strftime("%Y%m%d")


def _fetch_stooq_history(symbol: str, period: str = "60d", interval: str = "1h") -> pd.DataFrame:
    stooq_sym = STOOQ_SYMBOL_MAP.get(symbol)
    if not stooq_sym:
        raise ValueError(f"No Stooq mapping for {symbol}")

    stooq_interval = {"1wk": "w", "1mo": "m"}.get(interval, "d")
    from_date, to_date = _period_to_date_range(period)
    url = (
        f"https://stooq.com/q/d/l/?s={stooq_sym}"
        f"&d1={from_date}&d2={to_date}&i={stooq_interval}"
    )
    resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    df = pd.read_csv(_io.StringIO(resp.text))
    if df.empty or "Date" not in df.columns:
        raise ValueError(f"Stooq returned no data for {stooq_sym}")

    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df["date"], utc=True)
    if "vol" in df.columns:
        df = df.rename(columns={"vol": "volume"})
    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = _normalize_history_frame(df)

    # For sub-daily intervals, forward-fill daily bars into the requested frequency
    if interval not in ("1d", "1wk", "1mo") and not df.empty:
        rule = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h"}.get(interval)
        if rule:
            end_ts = pd.Timestamp.now(tz="UTC").floor(rule)
            df = df.resample(rule).ffill().dropna()
            df = df[df.index <= end_ts]

    bars_needed = _period_to_bars(period, interval)
    return df.tail(bars_needed)


def _fetch_binance_asset(binance_symbol: str, period: str = "5d", interval: str = "1h") -> pd.DataFrame:
    binance_interval = BINANCE_KLINE_INTERVALS.get(interval)
    if not binance_interval:
        raise ValueError(f"Unsupported Binance interval: {interval}")

    resp = requests.get(
        "https://api.binance.com/api/v3/klines",
        params={
            "symbol": binance_symbol,
            "interval": binance_interval,
            "limit": _period_to_bars(period, interval),
        },
        timeout=10,
        headers={"User-Agent": "trade-agent/1.0"},
    )
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return _normalize_history_frame(df)


def _fetch_market_history(symbol: str, ticker_symbol: str, period: str = "5d", interval: str = "1h") -> tuple[pd.DataFrame, str]:
    providers: list[tuple[str, object]] = []

    # Binance for crypto and gold/silver (intraday preferred)
    binance_sym = BINANCE_SYMBOL_MAP.get(symbol)
    if binance_sym:
        providers.append((f"binance-{binance_sym.lower()}", lambda s=binance_sym: _fetch_binance_asset(s, period=period, interval=interval)))

    # Alpha Vantage for US10Y
    if symbol == "US10Y":
        providers.append(("alpha-vantage", lambda: _fetch_alpha_vantage_treasury(period=period, interval=interval)))

    # Stooq for everything (daily data, resampled for intraday)
    if symbol in STOOQ_SYMBOL_MAP:
        providers.append(("stooq", lambda: _fetch_stooq_history(symbol, period=period, interval=interval)))

    last_error: Exception | None = None
    for provider_name, provider in providers:
        try:
            df = provider()
            if df is not None and not df.empty:
                return df, provider_name
        except Exception as exc:
            last_error = exc
            logger.warning(f"{provider_name} fetch failed for {symbol} ({type(exc).__name__}): {exc}")

    if last_error is not None:
        raise last_error
    return pd.DataFrame(), "unavailable"


def _df_to_ohlcv(symbol: str, df: pd.DataFrame) -> OHLCV:
    row = df.iloc[-1]
    ts = df.index[-1].to_pydatetime()
    return OHLCV(
        symbol=symbol,
        timestamp=ts,
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row.get("volume", 0)),
        session_label=get_session_label(ts),
    )


def _load_cached_df(cache_key: str, ttl: int) -> Optional[pd.DataFrame]:
    cached = cache_get(cache_key, ttl_seconds=ttl)
    if cached:
        df = pd.DataFrame(cached)
        df.index = pd.to_datetime(df["timestamp"], utc=True)
        df.attrs["source"] = "cache"
        return df
    return None


def _load_stale_df(cache_key: str) -> Optional[pd.DataFrame]:
    """Return cached data regardless of age — used as a rate-limit fallback."""
    p = _cache_path(cache_key)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text()).get("payload")
        if data:
            df = pd.DataFrame(data)
            df.index = pd.to_datetime(df["timestamp"], utc=True)
            df.attrs["source"] = "stale-cache"
            return df
    except Exception:
        pass
    return None


def get_historical_data(
    symbol: str,
    period: str = "60d",
    interval: str = "1h",
    use_cache: bool = True,
) -> pd.DataFrame:
    cache_key = f"hist_{symbol}_{period}_{interval}"
    ttl = 600  # 10-minute fresh TTL

    if use_cache:
        df = _load_cached_df(cache_key, ttl)
        if df is not None:
            return df

    ticker = SYMBOL_MAP.get(symbol, symbol)
    try:
        df, source = _fetch_market_history(symbol, ticker, period=period, interval=interval)
    except Exception as e:
        logger.warning(f"historical fetch failed for {symbol} ({type(e).__name__}): {e}")
        stale = _load_stale_df(cache_key) if use_cache else None
        if stale is not None:
            logger.info(f"Returning stale cache for {symbol}")
            return stale
        return pd.DataFrame()

    logger.debug(f"Historical data source for {symbol}: {source}")
    df.attrs["source"] = source

    df["symbol"] = symbol
    df["timestamp"] = df.index.astype(str)
    df["session_label"] = df.index.map(lambda ts: get_session_label(ts.to_pydatetime()).value)

    if use_cache:
        cache_set(cache_key, df.reset_index(drop=True).to_dict(orient="list"))

    return df


async def _fetch_all_async(symbols: list[str], period: str = "5d", interval: str = "1h",
                           batch_size: int = 3, batch_delay: float = 2.0) -> dict[str, pd.DataFrame]:
    """Fetch symbols in small batches with a delay to avoid rate limits."""
    loop = asyncio.get_event_loop()
    results = {}
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        tasks = {sym: loop.run_in_executor(None, get_historical_data, sym, period, interval)
                 for sym in batch}
        for sym, coro in tasks.items():
            try:
                results[sym] = await coro
            except Exception as e:
                logger.warning(f"Failed to fetch {sym}: {e}")
        if i + batch_size < len(symbols):
            await asyncio.sleep(batch_delay)
    return results


# CurrencyFreaks: fiat FX only (free tier excludes XAU/metals)
_CF_SYMBOLS = ["EUR", "GBP", "JPY", "CHF", "AUD", "CAD"]
_CF_TO_SNAPSHOT = {
    "EUR": "EURUSD",
    "GBP": "GBPUSD",
    "JPY": "USDJPY",
    "CHF": "USDCHF",
    "AUD": "AUDUSD",
    "CAD": "USDCAD",
}


def get_live_gold_price() -> Optional[float]:
    """Fetch live XAUUSD spot price.
    Sources tried in order:
    1. Binance PAXGUSDT  (realtime, no key)
    2. Stooq CSV         (free, no key)
    3. Stale file-cache  (any prior successful fetch)
    """
    cache_key = "live_xauusd_price"
    cached = cache_get(cache_key, ttl_seconds=60)
    if cached and cached.get("price"):
        return cached["price"]

    # ── 1. Binance PAXGUSDT ──────────────────────────────────────────────────
    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": "PAXGUSDT"},
            timeout=6,
            headers={"User-Agent": "trade-agent/1.0"},
        )
        resp.raise_for_status()
        price = float(resp.json()["price"])
        if price > 100:
            cache_set(cache_key, {"price": price})
            logger.info(f"Binance PAXGUSDT: {price:.2f}")
            return price
    except Exception as e:
        logger.debug(f"Binance gold fetch failed: {e}")

    # ── 2. Stooq ─────────────────────────────────────────────────────────────
    try:
        resp = requests.get(
            "https://stooq.com/q/l/?s=xauusd&f=sd2t2ohlcv&h&e=csv",
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        df_stooq = pd.read_csv(_io.StringIO(resp.text))
        close_col = next((c for c in df_stooq.columns if c.strip().lower() == "close"), None)
        if close_col and not df_stooq.empty:
            price = float(df_stooq[close_col].iloc[-1])
            if price > 100:
                cache_set(cache_key, {"price": price})
                logger.info(f"Stooq XAUUSD: {price:.2f}")
                return price
    except Exception as e:
        logger.debug(f"Stooq gold fetch failed: {e}")

    # ── 3. Stale file-cache ───────────────────────────────────────────────────
    for suffix in ("5d_1h", "30d_1h", "60d_1h", "1y_1d"):
        stale = _load_stale_df(f"hist_XAUUSD_{suffix}")
        if stale is not None and not stale.empty and "close" in stale.columns:
            price = float(stale["close"].iloc[-1])
            if price > 100:
                logger.info(f"Stale cache XAUUSD: {price:.2f}")
                return price

    logger.warning("All XAUUSD price sources failed")
    return None


def get_live_fx_rates() -> dict[str, float]:
    """Fetch live rates: XAUUSD from Alpha Vantage, FX pairs from CurrencyFreaks."""
    rates: dict[str, float] = {}

    gold_price = get_live_gold_price()
    if gold_price:
        rates["XAUUSD"] = gold_price

    cf_key = config.currencyfreaks_api_key
    if cf_key and cf_key != "your_key_here":
        cache_key = "cf_live_rates"
        cached = cache_get(cache_key, ttl_seconds=60)
        if cached:
            rates.update(cached)
        else:
            url = "https://api.currencyfreaks.com/v2.0/rates/latest"
            params = {"apikey": cf_key, "symbols": ",".join(_CF_SYMBOLS)}
            try:
                resp = requests.get(url, params=params, timeout=5)
                resp.raise_for_status()
                data = resp.json().get("rates", {})
                cf_rates: dict[str, float] = {}
                for cf_code, snap_key in _CF_TO_SNAPSHOT.items():
                    raw = data.get(cf_code)
                    if raw is None:
                        continue
                    val = float(raw)
                    if val == 0:
                        continue
                    cf_rates[snap_key] = val if snap_key.startswith("USD") else 1.0 / val
                cache_set(cache_key, cf_rates)
                rates.update(cf_rates)
                logger.debug(f"CurrencyFreaks FX rates: {cf_rates}")
            except Exception as e:
                logger.warning(f"CurrencyFreaks fetch failed: {e}")

    return rates


async def get_current_market_snapshot() -> dict:
    now = datetime.now(timezone.utc)
    cache_key = "snapshot_latest"
    cached = cache_get(cache_key, ttl_seconds=60)
    if cached:
        logger.debug("Returning cached market snapshot")
        return cached

    # Fetch live FX rates early so we can use them as fallback
    live_rates = get_live_fx_rates()

    all_syms = list(SYMBOL_MAP.keys())
    data = await _fetch_all_async(all_syms)

    primary_sym = "XAUUSD"
    primary_df = data.get(primary_sym)

    live_gold = live_rates.get("XAUUSD")

    if primary_df is not None and not primary_df.empty:
        primary_ohlcv = _df_to_ohlcv(primary_sym, primary_df)
        history_source = primary_df.attrs.get("source", "historical-feed")
        if live_gold:
            primary_ohlcv.close = live_gold
        xauusd_source = (f"metals.live+{history_source}" if live_gold else history_source)
    elif live_gold:
        logger.warning(f"Historical provider unavailable for XAUUSD; using live price {live_gold:.2f}")
        primary_ohlcv = OHLCV(
            symbol=primary_sym, timestamp=now,
            open=live_gold, high=live_gold, low=live_gold, close=live_gold,
            volume=0.0, session_label=get_session_label(now),
        )
        xauusd_source = "metals.live"
    else:
        logger.error("All XAUUSD price sources failed — snapshot will have 0 price")
        primary_ohlcv = OHLCV(
            symbol=primary_sym, timestamp=now,
            open=0.0, high=0.0, low=0.0, close=0.0,
            volume=0.0, session_label=get_session_label(now),
        )
        xauusd_source = "unavailable"

    correlated: dict[str, dict] = {}
    for sym, df in data.items():
        if sym == primary_sym or df is None or df.empty:
            continue
        try:
            correlated[sym] = asdict(_df_to_ohlcv(sym, df))
        except Exception as e:
            logger.warning(f"Could not parse {sym}: {e}")

    snapshot = MarketSnapshot(
        timestamp=now,
        primary=primary_ohlcv,
        correlated={k: OHLCV(**{**v, "session_label": TradingSession(v["session_label"])})
                    for k, v in correlated.items()},
    )
    result = snapshot.to_dict()

    if live_rates:
        result["live_fx_rates"] = live_rates
    result["meta"]["xauusd_source"] = xauusd_source

    cache_set(cache_key, result)
    return result


def get_multi_timeframe_data(symbol: str = "XAUUSD") -> dict[str, pd.DataFrame]:
    tf_map = {
        "Monthly": ("2y",  "1mo"),
        "Weekly":  ("2y",  "1wk"),
        "Daily":   ("1y",  "1d"),
        "H4":      ("60d", "1h"),
        "H1":      ("30d", "1h"),
        "M15":     ("7d",  "15m"),
        "M5":      ("5d",  "5m"),
        "M1":      ("2d",  "1m"),
    }
    result: dict[str, pd.DataFrame] = {}
    for label, (period, interval) in tf_map.items():
        try:
            df = get_historical_data(symbol, period=period, interval=interval)
            if label == "H4":
                df = df.resample("4h").agg(
                    {"open": "first", "high": "max", "low": "min",
                     "close": "last", "volume": "sum"}
                ).dropna()
            result[label] = df
        except Exception as e:
            logger.warning(f"Skipping {label} for {symbol}: {e}")
    return result


if __name__ == "__main__":
    async def _test():
        snap = await get_current_market_snapshot()
        print(json.dumps({k: str(v) for k, v in snap.items()}, indent=2))

    asyncio.run(_test())

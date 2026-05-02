import requests
import pandas as pd
import logging
import time

from tradax.bot.stock_fetcher import (
    calculate_ma, calculate_rsi_advanced, wyckoff_phase, generate_trading_signal_advanced
)

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

_STABLE_IDS = {
    "tether", "usd-coin", "dai", "binance-usd", "true-usd", "pax-dollar",
    "usdd", "frax", "gemini-dollar", "terrausd", "neutrino", "fei-usd",
    "liquity-usd", "origin-dollar", "reserve", "usde", "first-digital-usd",
}
_STABLE_SYMBOLS = {
    "usdt", "usdc", "dai", "busd", "tusd", "usdp", "usdd", "frax",
    "gusd", "lusd", "ousd", "rsv", "usde", "fdusd",
}

# --- cache: key -> (expires_at, data) ---
_cache: dict = {}
_CACHE_TTL = 300        # 5 minutes
_MIN_REQUEST_GAP = 6.0  # 10 calls/min max
_last_request_at = 0.0


def _api_get(url: str, params: dict = None) -> dict:
    """Single entry point for all CoinGecko requests.

    Applies:
    - In-memory cache (5 min TTL)
    - Rate limiting (min 6s between calls)
    - Retry with exponential backoff on 429 (5s → 10s → 20s → 40s)
    """
    global _last_request_at

    cache_key = (url, str(sorted((params or {}).items())))
    now = time.time()

    # Return cached data if still fresh
    if cache_key in _cache:
        expires_at, data = _cache[cache_key]
        if now < expires_at:
            logging.debug(f"Cache hit: {url}")
            return data

    # Enforce minimum gap between requests
    gap = time.time() - _last_request_at
    if gap < _MIN_REQUEST_GAP:
        time.sleep(_MIN_REQUEST_GAP - gap)

    # Request with retry on 429
    wait = 5.0
    max_retries = 4
    for attempt in range(max_retries):
        _last_request_at = time.time()
        try:
            resp = requests.get(url, params=params, timeout=30)

            if resp.status_code == 429:
                if attempt < max_retries - 1:
                    logging.warning(f"429 from CoinGecko, waiting {wait}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait)
                    wait *= 2
                    continue
                else:
                    logging.error("429 persists after all retries, returning cached or empty")
                    return _cache.get(cache_key, (None, None))[1] or {}

            resp.raise_for_status()
            data = resp.json()
            _cache[cache_key] = (time.time() + _CACHE_TTL, data)
            return data

        except requests.exceptions.HTTPError as e:
            if attempt < max_retries - 1:
                logging.warning(f"HTTP error {e}, retrying in {wait}s")
                time.sleep(wait)
                wait *= 2
            else:
                logging.error(f"Request failed after {max_retries} attempts: {e}")
                raise

    return {}


def _get_top_coins(limit=150):
    data = _api_get(f"{COINGECKO_BASE}/coins/markets", params={
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": limit,
        "page": 1,
        "price_change_percentage": "24h,7d,30d",
        "sparkline": False,
    })
    if not data:
        return []
    return [
        c for c in data
        if c["id"] not in _STABLE_IDS
        and c.get("symbol", "").lower() not in _STABLE_SYMBOLS
    ]


def _get_coin_market_chart(coin_id: str, days=31) -> pd.DataFrame:
    data = _api_get(f"{COINGECKO_BASE}/coins/{coin_id}/market_chart", params={
        "vs_currency": "usd",
        "days": days,
        "interval": "daily",
    })
    if not data:
        return pd.DataFrame()

    prices = pd.DataFrame(data["prices"], columns=["ts", "Close"])
    volumes = pd.DataFrame(data["total_volumes"], columns=["ts", "Volume"])
    df = prices.merge(volumes, on="ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("ts").sort_index()
    return df


def get_top_cryptos(top_n=10, intersect_n=20, max_signal_coins=5):
    coins = _get_top_coins(limit=150)
    if not coins:
        logging.error("No coin data returned from CoinGecko")
        return {"daily": [], "weekly": [], "monthly": [], "intersection": [], "intersection_with_signals": []}

    rows = []
    for c in coins:
        try:
            daily_change = c.get("price_change_percentage_24h_in_currency") or c.get("price_change_percentage_24h")
            weekly_change = c.get("price_change_percentage_7d_in_currency")
            monthly_change = c.get("price_change_percentage_30d_in_currency")
            price = c.get("current_price")
            vol = c.get("total_volume") or 0

            if None in (daily_change, weekly_change, monthly_change, price):
                continue

            rows.append({
                "id": c["id"],
                "symbol": c["symbol"].upper(),
                "price": round(price, 6),
                "daily_change": round(daily_change, 2),
                "weekly_change": round(weekly_change, 2),
                "monthly_change": round(monthly_change, 2),
                "daily_vol": int(vol),
                "weekly_vol": int(vol),
                "monthly_vol": int(vol),
                "Signal": "Hold",
            })
        except Exception as e:
            logging.warning(f"Error parsing coin {c.get('id')}: {e}")

    if not rows:
        return {"daily": [], "weekly": [], "monthly": [], "intersection": [], "intersection_with_signals": []}

    daily_sorted = sorted(rows, key=lambda x: abs(x["daily_change"]), reverse=True)[:intersect_n]
    weekly_sorted = sorted(rows, key=lambda x: abs(x["weekly_change"]), reverse=True)[:intersect_n]
    monthly_sorted = sorted(rows, key=lambda x: abs(x["monthly_change"]), reverse=True)[:intersect_n]

    intersection_ids = (
        {x["id"] for x in daily_sorted}
        & {x["id"] for x in weekly_sorted}
        & {x["id"] for x in monthly_sorted}
    )

    # Cap signal fetches to avoid hammering the API
    signal_ids = list(intersection_ids)[:max_signal_coins]
    signals: dict[str, str] = {}

    for coin_id in signal_ids:
        try:
            df = _get_coin_market_chart(coin_id, days=31)
            if df.empty or len(df) < 15:
                continue
            df = calculate_ma(df, short_window=5, long_window=15)
            df["RSI"] = calculate_rsi_advanced(df["Close"])
            df = wyckoff_phase(df, window=5)
            df = generate_trading_signal_advanced(df)
            signals[coin_id] = df["Signal"].iloc[-1]
        except Exception as e:
            logging.warning(f"Signal compute failed for {coin_id}: {e}")

    for lst in (daily_sorted, weekly_sorted, monthly_sorted):
        for item in lst:
            if item["id"] in signals:
                item["Signal"] = signals[item["id"]]

    intersection_symbols = [x["symbol"] for x in daily_sorted if x["id"] in intersection_ids]

    return {
        "daily": daily_sorted[:top_n],
        "weekly": weekly_sorted[:top_n],
        "monthly": monthly_sorted[:top_n],
        "intersection": intersection_symbols,
        "intersection_with_signals": [
            {"symbol": x["symbol"], "Signal": x["Signal"]}
            for x in daily_sorted if x["id"] in intersection_ids
        ],
    }

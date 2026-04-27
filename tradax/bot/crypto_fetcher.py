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


def _get_top_coins(limit=150):
    url = f"{COINGECKO_BASE}/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": limit,
        "page": 1,
        "price_change_percentage": "24h,7d,30d",
        "sparkline": False,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    coins = resp.json()
    return [
        c for c in coins
        if c["id"] not in _STABLE_IDS
        and c.get("symbol", "").lower() not in _STABLE_SYMBOLS
    ]


def _get_coin_market_chart(coin_id, days=31):
    """Returns DataFrame with daily Close and Volume columns."""
    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    prices = pd.DataFrame(data["prices"], columns=["ts", "Close"])
    volumes = pd.DataFrame(data["total_volumes"], columns=["ts", "Volume"])
    df = prices.merge(volumes, on="ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("ts").sort_index()
    return df


def get_top_cryptos(top_n=10, intersect_n=20):
    coins = _get_top_coins(limit=150)

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

    daily_sorted = sorted(rows, key=lambda x: abs(x["daily_change"]), reverse=True)[:intersect_n]
    weekly_sorted = sorted(rows, key=lambda x: abs(x["weekly_change"]), reverse=True)[:intersect_n]
    monthly_sorted = sorted(rows, key=lambda x: abs(x["monthly_change"]), reverse=True)[:intersect_n]

    daily_ids = {x["id"] for x in daily_sorted}
    weekly_ids = {x["id"] for x in weekly_sorted}
    monthly_ids = {x["id"] for x in monthly_sorted}
    intersection_ids = daily_ids & weekly_ids & monthly_ids

    # Fetch OHLCV and compute signals only for intersection coins
    signals: dict[str, str] = {}
    for coin_id in intersection_ids:
        try:
            time.sleep(1.5)  # CoinGecko free tier rate limit
            df = _get_coin_market_chart(coin_id, days=31)
            if len(df) < 15:
                continue
            df = calculate_ma(df, short_window=5, long_window=15)
            df["RSI"] = calculate_rsi_advanced(df["Close"])
            df = wyckoff_phase(df, window=5)
            df = generate_trading_signal_advanced(df)
            signals[coin_id] = df["Signal"].iloc[-1]
        except Exception as e:
            logging.warning(f"Could not compute signal for {coin_id}: {e}")

    for lst in (daily_sorted, weekly_sorted, monthly_sorted):
        for item in lst:
            if item["id"] in signals:
                item["Signal"] = signals[item["id"]]

    intersection_symbols = [
        x["symbol"] for x in daily_sorted if x["id"] in intersection_ids
    ]

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

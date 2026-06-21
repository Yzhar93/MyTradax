from __future__ import annotations

import yfinance as yf
import pandas as pd
import logging

from tradax.bot.stock_fetcher import (
    calculate_ma, calculate_rsi_advanced, wyckoff_phase, generate_trading_signal_advanced
)

INDICES = {
    "US Major": {
        "^GSPC":  "S&P 500",
        "^IXIC":  "NASDAQ Composite",
        "^DJI":   "Dow Jones",
        "^NDX":   "NASDAQ 100",
        "^RUT":   "Russell 2000",
    },
    "US Sectors": {
        "XLK": "Technology",
        "XLF": "Financials",
        "XLE": "Energy",
        "XLV": "Health Care",
        "XLI": "Industrials",
    },
    "Volatility & Rates": {
        "^VIX": "VIX Fear Index",
        "^TNX": "10Y Treasury Yield",
    },
    "International": {
        "^FTSE":     "FTSE 100 (UK)",
        "^GDAXI":    "DAX (Germany)",
        "^N225":     "Nikkei 225 (Japan)",
        "^HSI":      "Hang Seng (HK)",
        "^STOXX50E": "Euro Stoxx 50",
    },
    "Commodities": {
        "GC=F": "Gold",
        "CL=F": "Crude Oil",
    },
    "AI & Robotics": {
        "BOTZ": "Robotics & AI ETF",
        "AIQ":  "AI & Technology ETF",
        "ROBO": "Robotics & Automation ETF",
    },
    "Clean Energy": {
        "ICLN": "Global Clean Energy ETF",
        "TAN":  "Solar ETF",
        "LIT":  "Lithium & Battery ETF",
        "QCLN": "Clean Edge Green Energy ETF",
    },
    "Space": {
        "UFO":  "Space ETF",
        "ROKT": "Kensho Final Frontiers ETF",
    },
}

_ALL_TICKERS = [t for cat in INDICES.values() for t in cat]

_CATEGORY_EMOJI = {
    "US Major":           "🇺🇸",
    "US Sectors":         "💼",
    "Volatility & Rates": "🌡",
    "International":      "🌍",
    "Commodities":        "⛏",
    "AI & Robotics":      "🤖",
    "Clean Energy":       "🌱",
    "Space":              "🚀",
}


def get_indices_data() -> dict:
    """
    Fetch all index/ETF data and compute daily/weekly/monthly changes + signals.
    Returns dict keyed by category name, each value a sorted list of index dicts.
    """
    data = yf.download(
        _ALL_TICKERS,
        period="1mo",
        interval="1d",
        group_by="ticker",
        progress=False,
        timeout=30,
    )

    result = {}

    for category, tickers in INDICES.items():
        rows = []
        for symbol, name in tickers.items():
            try:
                df = _extract_df(data, symbol)
                if df is None or len(df) < 2:
                    logging.warning(f"Insufficient data for {symbol}")
                    continue

                today      = float(df["Close"].iloc[-1])
                yesterday  = float(df["Close"].iloc[-2])
                week_ago   = float(df["Close"].iloc[-6])  if len(df) > 6  else float(df["Close"].iloc[0])
                month_ago  = float(df["Close"].iloc[-22]) if len(df) > 22 else float(df["Close"].iloc[0])

                daily_change   = ((today - yesterday) / yesterday) * 100
                weekly_change  = ((today - week_ago)  / week_ago)  * 100
                monthly_change = ((today - month_ago) / month_ago) * 100

                rows.append({
                    "symbol":         symbol,
                    "name":           name,
                    "price":          round(today, 2),
                    "daily_change":   round(daily_change, 2),
                    "weekly_change":  round(weekly_change, 2),
                    "monthly_change": round(monthly_change, 2),
                    "Signal":         _compute_signal(df),
                })
            except Exception as e:
                logging.warning(f"Error processing {symbol}: {e}")

        rows.sort(key=lambda x: abs(x["daily_change"]), reverse=True)
        result[category] = rows

    return result


def _extract_df(data: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
    """Extract single-symbol DataFrame from a multi-ticker yfinance download."""
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if symbol not in data.columns.get_level_values(0):
                return None
            df = data[symbol].copy()
        else:
            df = data.copy()

        df = df.dropna(subset=["Close"])
        return df if not df.empty else None
    except Exception:
        return None


def _compute_signal(df: pd.DataFrame) -> str:
    """Run MA + RSI + Wyckoff signal pipeline. Returns 'Buy', 'Sell', or 'Hold'."""
    try:
        if len(df) < 15:
            return "Hold"

        df = df.copy()
        # VIX, TNX and some ETFs have no volume — fill with 0 so Wyckoff doesn't crash
        if "Volume" not in df.columns or df["Volume"].isna().all():
            df["Volume"] = 0
        else:
            df["Volume"] = df["Volume"].fillna(0)

        df = calculate_ma(df, short_window=5, long_window=15)
        df["RSI"] = calculate_rsi_advanced(df["Close"])
        df = wyckoff_phase(df, window=5)
        df = generate_trading_signal_advanced(df)
        return str(df["Signal"].iloc[-1])
    except Exception as e:
        logging.warning(f"Signal compute failed: {e}")
        return "Hold"

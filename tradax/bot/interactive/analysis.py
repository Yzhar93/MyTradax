from __future__ import annotations

import logging
import pandas as pd
import yfinance as yf

from tradax.bot.stock_fetcher import (
    calculate_ma, calculate_rsi_advanced, wyckoff_phase, generate_trading_signal_advanced
)
from tradax.bot.crypto_fetcher import _get_coin_market_chart

CRYPTO_ID_MAP = {
    "BTC":   "bitcoin",
    "ETH":   "ethereum",
    "BNB":   "binancecoin",
    "SOL":   "solana",
    "XRP":   "ripple",
    "ADA":   "cardano",
    "AVAX":  "avalanche-2",
    "DOT":   "polkadot",
    "LINK":  "chainlink",
    "MATIC": "matic-network",
}


def run_analysis(market: str, symbol: str) -> str:
    try:
        df = _fetch_df(market, symbol, days=30)
        if df is None or len(df) < 10:
            return f"Not enough data for {symbol}."

        df = calculate_ma(df, short_window=5, long_window=15)
        df["RSI"] = calculate_rsi_advanced(df["Close"])
        df = wyckoff_phase(df, window=5)
        df = generate_trading_signal_advanced(df)

        row    = df.iloc[-1]
        signal = df["Signal"].iloc[-1]
        reason = df["SignalReason"].iloc[-1] if "SignalReason" in df.columns else ""
        rsi    = round(row["RSI"], 1) if not pd.isna(row["RSI"]) else "N/A"
        phase  = row.get("WyckoffPhase") or "N/A"
        price  = round(float(row["Close"]), 4)

        emoji = "🔼" if signal == "Buy" else "🔽" if signal == "Sell" else "⏺"
        reason_short = reason[:120] if reason else "N/A"

        return (
            f"{emoji} {symbol} — Analysis (30d)\n\n"
            f"Price:   ${price}\n"
            f"RSI:     {rsi}\n"
            f"Phase:   {phase}\n"
            f"Signal:  {signal}\n"
            f"Reason:  {reason_short}"
        )
    except Exception as e:
        logging.error(f"Analysis failed for {symbol}: {e}")
        return f"Analysis failed for {symbol}."


def _fetch_df(market: str, symbol: str, days: int) -> pd.DataFrame | None:
    try:
        if market == "stocks":
            df = yf.download(symbol, period="1mo", interval="1d", progress=False)
            if df.empty:
                return None
            df = df[["Close", "Volume"]].dropna()
            df.columns = ["Close", "Volume"]
            return df
        elif market == "indices":
            df = yf.download(symbol, period="1mo", interval="1d", progress=False)
            if df.empty:
                return None
            df = df[["Close", "Volume"]].copy()
            df["Volume"] = df["Volume"].fillna(0)
            return df.dropna(subset=["Close"])
        else:
            coin_id = CRYPTO_ID_MAP.get(symbol.upper(), symbol.lower())
            return _get_coin_market_chart(coin_id, days=days)
    except Exception as e:
        logging.error(f"_fetch_df failed for {market}/{symbol}: {e}")
        return None

import logging
import pandas as pd

from tradax.bot.stock_fetcher import (
    calculate_ma, calculate_rsi_advanced, wyckoff_phase, generate_trading_signal_advanced
)
from tradax.bot.interactive.analysis import _fetch_df

PERIOD_DAYS = {"7d": 7, "30d": 30, "90d": 90, "1y": 365}


def run_backtest(market: str, symbol: str, period: str) -> str:
    days = PERIOD_DAYS.get(period, 30)
    try:
        df = _fetch_df(market, symbol, days=days)
        if df is None or len(df) < 10:
            return f"Not enough data for {symbol} ({period})."

        df = calculate_ma(df, short_window=5, long_window=15)
        df["RSI"] = calculate_rsi_advanced(df["Close"])
        df = wyckoff_phase(df, window=5)
        df = generate_trading_signal_advanced(df)

        roi, win_rate, drawdown, n_trades = _simulate_trades(df)
        current_signal = df["Signal"].iloc[-1]
        recommendation = _recommend(roi, win_rate, drawdown)

        return (
            f"🔁 Backtest: {symbol} — {period}\n\n"
            f"ROI:        {roi:+.2f}%\n"
            f"Win Rate:   {win_rate:.0%}  ({n_trades} trades)\n"
            f"Max DD:     -{drawdown:.2f}%\n"
            f"Signal now: {current_signal}\n\n"
            f"Verdict:    {recommendation}"
        )
    except Exception as e:
        logging.error(f"Backtest failed for {symbol}/{period}: {e}")
        return f"Backtest failed for {symbol}."


def _simulate_trades(df: pd.DataFrame):
    signals = df["Signal"].tolist()
    prices  = df["Close"].tolist()

    equity   = 1.0
    peak     = 1.0
    trades   = []
    position = None

    for sig, price in zip(signals, prices):
        if sig == "Buy" and position is None:
            position = price
        elif sig == "Sell" and position is not None:
            ret = (price - position) / position
            trades.append(ret)
            equity *= (1 + ret)
            peak    = max(peak, equity)
            position = None

    # Close open position at last price
    if position is not None:
        ret = (prices[-1] - position) / position
        trades.append(ret)
        equity *= (1 + ret)
        peak = max(peak, equity)

    roi       = (equity - 1.0) * 100
    win_rate  = sum(1 for t in trades if t > 0) / len(trades) if trades else 0.0
    drawdown  = (peak - equity) / peak * 100
    return roi, win_rate, drawdown, len(trades)


def _recommend(roi: float, win_rate: float, drawdown: float) -> str:
    if roi > 10 and win_rate >= 0.55:
        return "✅ BUY"
    if roi < -5 or (drawdown > 15 and win_rate < 0.45):
        return "❌ AVOID"
    return "⏸ HOLD"

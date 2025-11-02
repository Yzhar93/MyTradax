import yfinance as yf
import pandas as pd
import numpy as np
import logging


def get_sp500_tickers():
    url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'
    df = pd.read_csv(url)
    sp500_symbols = df['Symbol'].tolist()
    return sp500_symbols

def get_top_stocks(top_n=10):
    tickers = get_sp500_tickers()
    data = yf.download(tickers, period="2d", interval="1d", group_by="ticker", progress=False)

    movers = []
    for t in tickers:
        try:
            closes = data[t]["Close"]
            today, yesterday = closes.iloc[-1], closes.iloc[-2]
            change_pct = ((today - yesterday) / yesterday) * 100
            movers.append({
                "symbol": t,
                "price": round(today, 2),
                "change_pct": round(change_pct, 2)
            })
        except Exception as e:
            logging.warning(f'issue with ticket {t}: {e}')

    movers_sorted = sorted(movers, key=lambda x: abs(x["change_pct"]), reverse=True)
    return movers_sorted[:top_n]



# def get_top_stocks_advance(top_n=10, intersect_n=20):
#     tickers = get_sp500_tickers()
#     data = yf.download(tickers, period="1mo", interval="1d", group_by="ticker", progress=False)
#
#     daily_changes, weekly_changes, monthly_changes = [], [], []
#
#     for t in tickers:
#         try:
#             df = data[t]
#             if len(df) < 2:
#                 continue
#
#             today = df["Close"].iloc[-1]
#             yesterday = df["Close"].iloc[-2]
#             week_ago = df["Close"].iloc[-6] if len(df) > 6 else df["Close"].iloc[0]
#             month_ago = df["Close"].iloc[-22] if len(df) > 22 else df["Close"].iloc[0]
#
#             # --- % changes ---
#             daily_change = ((today - yesterday) / yesterday) * 100
#             weekly_change = ((today - week_ago) / week_ago) * 100
#             monthly_change = ((today - month_ago) / month_ago) * 100
#
#             # --- volumes ---
#             daily_vol = df["Volume"].iloc[-1]
#             weekly_vol = df["Volume"].iloc[-5:].mean()
#             monthly_vol = df["Volume"].iloc[-21:].mean()
#
#             stock_data = {
#                 "symbol": t,
#                 "price": round(today, 2),
#                 "daily_change": round(daily_change, 2),
#                 "weekly_change": round(weekly_change, 2),
#                 "monthly_change": round(monthly_change, 2),
#                 "daily_vol": int(daily_vol),
#                 "weekly_vol": int(weekly_vol),
#                 "monthly_vol": int(monthly_vol)
#             }
#
#             daily_changes.append(stock_data)
#             weekly_changes.append(stock_data)
#             monthly_changes.append(stock_data)
#
#         except Exception:
#             continue
#
#     # --- Sort by absolute change ---
#     daily_sorted = sorted(daily_changes, key=lambda x: abs(x["daily_change"]), reverse=True)[:intersect_n]
#     weekly_sorted = sorted(weekly_changes, key=lambda x: abs(x["weekly_change"]), reverse=True)[:intersect_n]
#     monthly_sorted = sorted(monthly_changes, key=lambda x: abs(x["monthly_change"]), reverse=True)[:intersect_n]
#
#     # --- Find intersection (stocks appearing in all three) ---
#     daily_set = {x["symbol"] for x in daily_sorted}
#     weekly_set = {x["symbol"] for x in weekly_sorted}
#     monthly_set = {x["symbol"] for x in monthly_sorted}
#     intersection = daily_set & weekly_set & monthly_set
#
#     # --- Output ---
#     print("\n📈 Top 10 Daily Movers:")
#     for s in daily_sorted[:top_n]:
#         print(f"{s['symbol']}: {s['daily_change']}% | Vol: {s['daily_vol']}")
#
#     print("\n📊 Top 10 Weekly Movers:")
#     for s in weekly_sorted[:top_n]:
#         print(f"{s['symbol']}: {s['weekly_change']}% | Avg Vol: {s['weekly_vol']}")
#
#     print("\n📅 Top 10 Monthly Movers:")
#     for s in monthly_sorted[:top_n]:
#         print(f"{s['symbol']}: {s['monthly_change']}% | Avg Vol: {s['monthly_vol']}")
#
#     print("\n🔁 Intersection of All Periods:")
#     print(intersection if intersection else "No overlapping top movers.")
#
#     return {
#         "daily": daily_sorted[:top_n],
#         "weekly": weekly_sorted[:top_n],
#         "monthly": monthly_sorted[:top_n],
#         "intersection": list(intersection)
#     }
# -------------------------------------------------------------------------------------------
def calculate_ma(data, short_window=50, long_window=100):
    """Add short and long moving averages."""
    data = data.copy()
    data.loc[:, 'MA_short'] = data['Close'].rolling(window=short_window).mean()
    data.loc[:, 'MA_long'] = data['Close'].rolling(window=long_window).mean()
    return data

def calculate_rsi(series, period=14):
    series = series.astype(float).squeeze()  # Ensure 1D
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=series.index).rolling(window=period, min_periods=period).mean()
    avg_loss = pd.Series(loss, index=series.index).rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_rsi_advanced(series, period=14):
    prices = pd.Series(series).astype(float)
    delta = prices.diff()

    # Gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder’s initial SMA for first period
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    # Now apply Wilder’s smoothing recursively
    for i in range(period, len(prices)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period

    # RS and RSI calculation
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Stability: if avg_loss is 0 → RSI should be 100 (no losses)
    rsi = rsi.fillna(0).clip(lower=0, upper=100)

    return rsi

def wyckoff_phase(df, window=20, volume_multiplier=1.5):
    """
    מזהה שלבי Wyckoff לפי מחיר ונפח
    df - DataFrame עם עמודות ['Close', 'Volume']
    window - מספר ימים לחישוב טווח ממוצע
    volume_multiplier - כמה גבוה הנפח לעומת ממוצע כדי לזהות פעילות חריגה
    """
    df = df.copy()
    df['High_roll'] = df['Close'].rolling(window).max()
    df['Low_roll'] = df['Close'].rolling(window).min()
    df['Volume_avg'] = df['Volume'].rolling(window).mean()

    phases = []
    for i in range(len(df)):
        price = df['Close'].iloc[i]
        vol = df['Volume'].iloc[i]
        high = df['High_roll'].iloc[i]
        low = df['Low_roll'].iloc[i]
        vol_avg = df['Volume_avg'].iloc[i]

        if np.isnan(high) or np.isnan(low) or np.isnan(vol_avg):
            phases.append(None)
            continue

        # Accumulation:
        if low <= price <= high and vol < vol_avg * volume_multiplier:
            phases.append('Accumulation')
        # Markup:
        elif price > high and vol > vol_avg:
            phases.append('Markup')
        # Distribution:
        elif low <= price <= high and vol > vol_avg * volume_multiplier:
            phases.append('Distribution')
        # Markdown:
        elif price < low and vol > vol_avg * 0.5:
            phases.append('Markdown')
        else:
            phases.append(None)

    df['WyckoffPhase'] = phases
    return df


def generate_trading_signal_advanced(df, volume_multiplier=1.5):
    """
    Generate Buy / Sell / Hold signals using:
    ✅ Trend (MA crossover)
    ✅ Strength (RSI)
    ✅ Market cycle context (Wyckoff)
    ✅ Volume confirmation (optional)

    Output:
      - df['Signal'] = final trading signal
      - df['SignalReason'] = explanation for the decision
    """

    df = df.copy()
    df['VolumeSpike'] = df['Volume'] / df['Volume'].rolling(window=20).mean()

    signals = []
    reasons = []

    for i in range(len(df)):
        row = df.iloc[i]
        signal = "Hold"
        reason_log = []

        # ==== 1️⃣ TREND: Moving Average Crossover ====
        if row['MA_short'] > row['MA_long']:
            signal = "Buy"
            reason_log.append("Trend Up (MA short > MA long)")
        elif row['MA_short'] < row['MA_long']:
            signal = "Sell"
            reason_log.append("Trend Down (MA short < MA long)")
        else:
            reason_log.append("No clear trend")

        # ==== 2️⃣ STRENGTH: RSI Filter ====
        if row['RSI'] >= 70:
            signal = "Sell"
            reason_log.append("Overbought (RSI ≥ 70)")
        elif row['RSI'] <= 30:
            signal = "Buy"
            reason_log.append("Oversold (RSI ≤ 30)")

        # ==== 3️⃣ MARKET CYCLE: Wyckoff Adjustment ====
        phase = row.get('WyckoffPhase', None)

        if phase == "Accumulation":
            if signal == "Hold":
                signal = "Buy"
            reason_log.append("Accumulation Phase ✅ Buy support")

        elif phase == "Markup":
            if signal == "Sell":
                signal = "Hold"  # don't panic sell in growth
            reason_log.append("Markup Phase 🚀 Trend support")

        elif phase == "Distribution":
            if signal == "Hold":
                signal = "Sell"
            reason_log.append("Distribution Phase ⚠️ Sell risk")

        elif phase == "Markdown":
            if signal == "Buy":
                signal = "Hold"
            reason_log.append("Markdown Phase ⬇️ Weak market")

        # ==== 4️⃣ VOLUME Confirmation ====
        if row['VolumeSpike'] > volume_multiplier:
            reason_log.append("High Volume 🔥 Institutional interest")

        signals.append(signal)
        reasons.append("; ".join(reason_log))

    df['Signal'] = signals
    df['SignalReason'] = reasons
    return df

def generate_trading_signal(df):
    """
    Generate buy/sell/hold signals based on MA crossovers, RSI, and Wyckoff phases.
    Assumes df has ['Close', 'MA_short', 'MA_long', 'RSI', 'WyckoffPhase'].
    """
    signals = []

    for i in range(len(df)):
        row = df.iloc[i]
        signal = 'Hold'

        # MA crossover strategy
        if row['MA_short'] > row['MA_long'] and row['RSI'] < 70:
            signal = 'Buy'
        elif row['MA_short'] < row['MA_long'] and row['RSI'] > 30:
            signal = 'Sell'

        # Wyckoff phase filter
        if row['WyckoffPhase'] == 'Accumulation':
            signal = 'Buy'
        elif row['WyckoffPhase'] == 'Distribution':
            signal = 'Sell'
        elif row['WyckoffPhase'] == 'Markup' and signal != 'Sell':
            signal = 'Hold'
        elif row['WyckoffPhase'] == 'Markdown' and signal != 'Buy':
            signal = 'Hold'

        signals.append(signal)

    df['Signal'] = signals
    return df

def get_top_stocks_advance(top_n=10, intersect_n=20):
    tickers = get_sp500_tickers()
    data = yf.download(tickers, period="1mo", interval="1d", group_by="ticker", progress=False)

    daily_changes, weekly_changes, monthly_changes = [], [], []

    for t in tickers:
        try:
            df = data[t]
            if len(df) < 2:
                continue

            today = df["Close"].iloc[-1]
            yesterday = df["Close"].iloc[-2]
            week_ago = df["Close"].iloc[-6] if len(df) > 6 else df["Close"].iloc[0]
            month_ago = df["Close"].iloc[-22] if len(df) > 22 else df["Close"].iloc[0]

            # --- % changes ---
            daily_change = ((today - yesterday) / yesterday) * 100
            weekly_change = ((today - week_ago) / week_ago) * 100
            monthly_change = ((today - month_ago) / month_ago) * 100

            # --- volumes ---
            daily_vol = df["Volume"].iloc[-1]
            weekly_vol = df["Volume"].iloc[-5:].mean()
            monthly_vol = df["Volume"].iloc[-21:].mean()

            stock_data = {
                "symbol": t,
                "price": round(today, 2),
                "daily_change": round(daily_change, 2),
                "weekly_change": round(weekly_change, 2),
                "monthly_change": round(monthly_change, 2),
                "daily_vol": int(daily_vol),
                "weekly_vol": int(weekly_vol),
                "monthly_vol": int(monthly_vol)
            }

            daily_changes.append(stock_data)
            weekly_changes.append(stock_data)
            monthly_changes.append(stock_data)

        except Exception as e:
            logging.warning(f'error in {t}: {e}')
            continue

    # --- Sort by absolute change ---
    daily_sorted = sorted(daily_changes, key=lambda x: abs(x["daily_change"]), reverse=True)[:intersect_n]
    weekly_sorted = sorted(weekly_changes, key=lambda x: abs(x["weekly_change"]), reverse=True)[:intersect_n]
    monthly_sorted = sorted(monthly_changes, key=lambda x: abs(x["monthly_change"]), reverse=True)[:intersect_n]

    # --- Find intersection (stocks appearing in all three) ---
    daily_set = {x["symbol"] for x in daily_sorted}
    weekly_set = {x["symbol"] for x in weekly_sorted}
    monthly_set = {x["symbol"] for x in monthly_sorted}
    intersection = daily_set & weekly_set & monthly_set

    # --- Output ---
    print("\n📈 Top 10 Daily Movers:")
    for s in daily_sorted[:top_n]:
        print(f"{s['symbol']}: {s['daily_change']}% | Vol: {s['daily_vol']}")

    print("\n📊 Top 10 Weekly Movers:")
    for s in weekly_sorted[:top_n]:
        print(f"{s['symbol']}: {s['weekly_change']}% | Avg Vol: {s['weekly_vol']}")

    print("\n📅 Top 10 Monthly Movers:")
    for s in monthly_sorted[:top_n]:
        print(f"{s['symbol']}: {s['monthly_change']}% | Avg Vol: {s['monthly_vol']}")

    print("\n🔁 Intersection of All Periods:")
    print(intersection if intersection else "No overlapping top movers.")

    return {
        "daily": daily_sorted[:top_n],
        "weekly": weekly_sorted[:top_n],
        "monthly": monthly_sorted[:top_n],
        "intersection": list(intersection)
    }



def get_top_stocks_extra(top_n=10, intersect_n=20):
    tickers = get_sp500_tickers()
    data = yf.download(tickers, period="1mo", interval="1d", group_by="ticker", progress=False)

    daily_changes, weekly_changes, monthly_changes = [], [], []

    for t in tickers:
        try:
            df = data[t].copy()
            if len(df) < 2:
                continue

            # Price references
            today = df["Close"].iloc[-1]
            yesterday = df["Close"].iloc[-2]
            week_ago = df["Close"].iloc[-6] if len(df) > 6 else df["Close"].iloc[0]
            month_ago = df["Close"].iloc[-22] if len(df) > 22 else df["Close"].iloc[0]

            # --- % changes ---
            daily_change = ((today - yesterday) / yesterday) * 100
            weekly_change = ((today - week_ago) / week_ago) * 100
            monthly_change = ((today - month_ago) / month_ago) * 100

            # --- volumes ---
            daily_vol = df["Volume"].iloc[-1]
            weekly_vol = df["Volume"].iloc[-5:].mean()
            monthly_vol = df["Volume"].iloc[-21:].mean()

            # --- Technical indicators ---
            df = calculate_ma(df, short_window=5, long_window=15)  # shorter windows for 1mo data
            df['RSI'] = calculate_rsi_advanced(df['Close'])
            df = wyckoff_phase(df, window=5)

            df = generate_trading_signal_advanced(df)  # add Signal column
            latest_signal = df['Signal'].iloc[-1]

            stock_data = {
                "symbol": t,
                "price": round(today, 2),
                "daily_change": round(daily_change, 2),
                "weekly_change": round(weekly_change, 2),
                "monthly_change": round(monthly_change, 2),
                "daily_vol": int(daily_vol),
                "weekly_vol": int(weekly_vol),
                "monthly_vol": int(monthly_vol),
                "Signal": latest_signal
            }

            daily_changes.append(stock_data)
            weekly_changes.append(stock_data)
            monthly_changes.append(stock_data)

        except Exception as e:
            logging.warning(f"⛔ Error for {t}: {e}")
            continue

    # --- Sort by absolute change ---
    daily_sorted = sorted(daily_changes, key=lambda x: abs(x["daily_change"]), reverse=True)[:intersect_n]
    weekly_sorted = sorted(weekly_changes, key=lambda x: abs(x["weekly_change"]), reverse=True)[:intersect_n]
    monthly_sorted = sorted(monthly_changes, key=lambda x: abs(x["monthly_change"]), reverse=True)[:intersect_n]

    # --- Find intersection (stocks appearing in all three) ---
    daily_set = {x["symbol"] for x in daily_sorted}
    weekly_set = {x["symbol"] for x in weekly_sorted}
    monthly_set = {x["symbol"] for x in monthly_sorted}
    intersection = daily_set & weekly_set & monthly_set

    # --- Output ---
    print("\n📈 Top 10 Daily Movers:")
    for s in daily_sorted[:top_n]:
        print(f"{s['symbol']}: {s['daily_change']}% | Vol: {s['daily_vol']} | Signal: {s['Signal']}")

    print("\n📊 Top 10 Weekly Movers:")
    for s in weekly_sorted[:top_n]:
        print(f"{s['symbol']}: {s['weekly_change']}% | Avg Vol: {s['weekly_vol']} | Signal: {s['Signal']}")

    print("\n📅 Top 10 Monthly Movers:")
    for s in monthly_sorted[:top_n]:
        print(f"{s['symbol']}: {s['monthly_change']}% | Avg Vol: {s['monthly_vol']} | Signal: {s['Signal']}")

    print("\n🔁 Intersection of All Periods:")
    if intersection:
        for s in intersection:
            stock_signal = next((x["Signal"] for x in daily_sorted if x["symbol"] == s), "Hold")
            print(f"{s} | Signal: {stock_signal}")
    else:
        print("No overlapping top movers.")

    return {
        "daily": daily_sorted[:top_n],
        "weekly": weekly_sorted[:top_n],
        "monthly": monthly_sorted[:top_n],
        "intersection": list(intersection)
    }

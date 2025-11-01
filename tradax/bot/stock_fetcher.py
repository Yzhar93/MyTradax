import yfinance as yf
import pandas as pd
import numpy as np


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
        except Exception:
            pass

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
    data['MA_short'] = data['Close'].rolling(window=short_window).mean()
    data['MA_long'] = data['Close'].rolling(window=long_window).mean()
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

        # Accumulation: בתוך טווח, נפח נמוך
        if low <= price <= high and vol < vol_avg * volume_multiplier:
            phases.append('Accumulation')
        # Markup: פריצה למעלה
        elif price > high and vol > vol_avg:
            phases.append('Markup')
        # Distribution: בתוך טווח, נפח גבוה
        elif low <= price <= high and vol > vol_avg * volume_multiplier:
            phases.append('Distribution')
        # Markdown: פריצה למטה
        elif price < low and vol > vol_avg * 0.5:
            phases.append('Markdown')
        else:
            phases.append(None)

    df['WyckoffPhase'] = phases
    return df


# דוגמה לשימוש
# df - DataFrame עם עמודות ['Close', 'Volume'] יומי
# df_analyzed = wyckoff_phase(df)
# print(df_analyzed[['Close', 'Volume', 'WyckoffPhase']])

# def get_top_stocks_advanced(top_n=10, ticker_count=20):
#     tickers = get_sp500_tickers()[:ticker_count]
#     results = []
#
#     for ticker in tickers:
#         try:
#             df = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=True)
#
#             if df.empty or len(df) < 22:
#                 print(f"⛔ Not enough data for {ticker}")
#                 continue
#
#             # Flatten MultiIndex if present
#             if isinstance(df.columns, pd.MultiIndex):
#                 df.columns = df.columns.get_level_values(0)
#
#             # Use Adj Close if Close missing
#             if 'Close' in df.columns:
#                 close_series = df['Close']
#             elif 'Adj Close' in df.columns:
#                 close_series = df['Adj Close']
#             else:
#                 print(f"⛔ No Close/Adj Close data for {ticker}")
#                 continue
#
#             close_series = close_series.astype(float).squeeze()  # Ensure 1D
#             df['RSI'] = calculate_rsi(close_series)
#
#             daily_change = (close_series.iloc[-1] / close_series.iloc[-2] - 1) * 100
#             weekly_change = (close_series.iloc[-1] / close_series.iloc[-6] - 1) * 100
#             monthly_change = (close_series.iloc[-1] / close_series.iloc[-21] - 1) * 100
#
#             volume_series = df['Volume'].astype(float).squeeze()
#             volume_mean = volume_series.tail(20).mean()
#             volume_spike = volume_series.iloc[-1] / volume_mean if volume_mean > 0 else 0
#
#             results.append({
#                 "Ticker": ticker,
#                 "RSI": round(df['RSI'].iloc[-1], 2) if not np.isnan(df['RSI'].iloc[-1]) else None,
#                 "DailyChange(%)": round(daily_change, 2),
#                 "WeeklyChange(%)": round(weekly_change, 2),
#                 "MonthlyChange(%)": round(monthly_change, 2),
#                 "VolumeSpike": round(volume_spike, 2)
#             })
#
#         except Exception as e:
#             print(f"⛔ Error for {ticker}: {e}")
#             continue
#
#     df_results = pd.DataFrame(results)
#     if not df_results.empty:
#         df_results['RSI_rank'] = df_results['RSI'].rank(ascending=False, method='min')
#         df_results['Daily_rank'] = df_results['DailyChange(%)'].rank(ascending=False, method='min')
#         df_results['Weekly_rank'] = df_results['WeeklyChange(%)'].rank(ascending=False, method='min')
#         df_results['Monthly_rank'] = df_results['MonthlyChange(%)'].rank(ascending=False, method='min')
#         df_results['Volume_rank'] = df_results['VolumeSpike'].rank(ascending=False, method='min')
#         df_results['TotalScore'] = df_results[['RSI_rank','Daily_rank','Weekly_rank','Monthly_rank','Volume_rank']].sum(axis=1)
#         df_results = df_results.sort_values('TotalScore')
#         return df_results.head(top_n)
#     else:
#         print("❌ No valid results found.")
#         return pd.DataFrame()


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

        except Exception:
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
TOP_STOCKS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "JPM", "V", "NFLX"]
TOP_CRYPTO = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "DOT", "LINK", "MATIC"]


def _keyboard(rows: list) -> dict:
    return {"inline_keyboard": rows}


def main_menu() -> dict:
    return _keyboard([[
        {"text": "📈 Stocks", "callback_data": "menu:stocks"},
        {"text": "🪙 Crypto",  "callback_data": "menu:crypto"},
    ]])


def asset_list_menu(market: str) -> dict:
    assets = TOP_STOCKS if market == "stocks" else TOP_CRYPTO
    rows = [
        [{"text": s, "callback_data": f"pick:{market}:{s}"} for s in assets[i:i+3]]
        for i in range(0, len(assets), 3)
    ]
    rows.append([{"text": "⬅ Back", "callback_data": "menu:main"}])
    return _keyboard(rows)


def action_menu(market: str, symbol: str) -> dict:
    return _keyboard([
        [
            {"text": "📊 Analysis", "callback_data": f"action:analysis:{market}:{symbol}"},
            {"text": "🔁 Backtest", "callback_data": f"action:backtest:{market}:{symbol}"},
        ],
        [{"text": "⬅ Back", "callback_data": f"menu:{market}"}],
    ])


def timerange_menu(market: str, symbol: str) -> dict:
    return _keyboard([
        [
            {"text": t, "callback_data": f"action:backtest:{market}:{symbol}:{t}"}
            for t in ("7d", "30d", "90d", "1y")
        ],
        [{"text": "⬅ Back", "callback_data": f"pick:{market}:{symbol}"}],
    ])

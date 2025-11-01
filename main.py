from tradax.bot.stock_fetcher import get_top_stocks, get_top_stocks_advance
from tradax.bot.message_builder import build_message, build_message_advance
from tradax.bot.telegram_client import send_telegram_message
from tradax.bot.llm_integration import enhance_message, enhance_message_advance  # optional
import logging
def stock_summary(event, context):
    try:
        stocks = get_top_stocks()
        msg = build_message(stocks)
        msg = enhance_message(msg)  # optional Gemini integration
        send_telegram_message(msg)
    except Exception as e:
        logging.error(f"Function failed: {e}")





# def stock_summary(event, context):
#     try:
#         stocks = get_top_stocks_advance()
#         msg = build_message_advance(stocks)
#         msg = enhance_message_advance(msg)  # optional Gemini integration
#         send_telegram_message(msg)
#     except Exception as e:
#         logging.error(f"Function failed: {e}")
import logging
import os
from dotenv import load_dotenv

load_dotenv()


def stock_summary(event, context):
    from tradax.bot.stock_fetcher import get_top_stocks_extra
    from tradax.bot.message_builder import build_message_extra
    from tradax.bot.telegram_client import send_telegram_message
    from tradax.bot.llm_integration import enhance_message_extra
    try:
        stocks = get_top_stocks_extra()
        msg = build_message_extra(stocks)
        msg = enhance_message_extra(msg)
        send_telegram_message(msg)
        return "OK", 200
    except Exception as e:
        logging.error(f"stock_summary failed: {e}")
        return f"Error: {e}", 500


def crypto_summary(event, context):
    from tradax.bot.crypto_fetcher import get_top_cryptos
    from tradax.bot.message_builder import build_message_crypto
    from tradax.bot.telegram_client import send_telegram_message
    from tradax.bot.llm_integration import enhance_message_crypto
    try:
        cryptos = get_top_cryptos()
        msg = build_message_crypto(cryptos)
        msg = enhance_message_crypto(msg)
        crypto_chat_id = os.environ.get("TELEGRAM_CRYPTO_CHAT_ID")
        send_telegram_message(msg, chat_id=crypto_chat_id)
        return "OK", 200
    except Exception as e:
        logging.error(f"crypto_summary failed: {e}")
        return f"Error: {e}", 500


def telegram_webhook(request):
    from tradax.bot.interactive.handlers import handle_message, handle_callback
    try:
        data = request.get_json(silent=True)
        if not data:
            return "OK", 200
        if "callback_query" in data:
            handle_callback(data["callback_query"])
        elif "message" in data:
            handle_message(data["message"])
        return "OK", 200
    except Exception as e:
        logging.error(f"telegram_webhook failed: {e}")
        return "OK", 200  # always 200 so Telegram doesn't retry

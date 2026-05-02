from tradax.bot.stock_fetcher import get_top_stocks, get_top_stocks_advance, get_top_stocks_extra
from tradax.bot.crypto_fetcher import get_top_cryptos
from tradax.bot.message_builder import build_message, build_message_advance, build_message_extra, build_message_crypto
from tradax.bot.telegram_client import send_telegram_message
from tradax.bot.llm_integration import enhance_message, enhance_message_advance, enhance_message_extra, enhance_message_crypto
import logging
import os
from dotenv import load_dotenv
load_dotenv()


def stock_summary(event, context):
    try:
        stocks = get_top_stocks_extra()
        msg = build_message_extra(stocks)
        msg = enhance_message_extra(msg)  # optional Gemini integration
        send_telegram_message(msg)
        return "OK", 200
    except Exception as e:
        logging.error(f"Function failed: {e}")
        return f"Error: {e}", 500


def crypto_summary(event, context):
    try:
        cryptos = get_top_cryptos()
        msg = build_message_crypto(cryptos)
        msg = enhance_message_crypto(msg)
        crypto_chat_id = os.environ.get("TELEGRAM_CRYPTO_CHAT_ID")
        send_telegram_message(msg, chat_id=crypto_chat_id)
        return "OK", 200
    except Exception as e:
        logging.error(f"Crypto function failed: {e}")
        return f"Error: {e}", 500


def telegram_webhook(request):
    try:
        data = request.get_json(silent=True)
        if not data:
            return "OK", 200
        from tradax.bot.interactive.handlers import handle_message, handle_callback
        if "callback_query" in data:
            handle_callback(data["callback_query"])
        elif "message" in data:
            handle_message(data["message"])
        return "OK", 200
    except Exception as e:
        logging.error(f"Webhook failed: {e}")
        return "OK", 200  # always 200 so Telegram doesn't retry
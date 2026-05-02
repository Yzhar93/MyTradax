import os
import requests
from dotenv import load_dotenv
load_dotenv()
import re

def escape_markdown(text):
    # Escape Telegram MarkdownV2 special characters
    escape_chars = r"_*[]()~`>#+-=|{}.! "
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)

def send_telegram_message(message, chat_id=None):
    """
    Send a message to a Telegram chat using a bot.

    Requires the environment variables:
    - TELEGRAM_BOT_TOKEN
    - TELEGRAM_CHAT_ID (used when chat_id is not passed explicitly)

    :param message: string, the message to send
    :param chat_id: optional Telegram chat ID; defaults to TELEGRAM_CHAT_ID env var
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if chat_id is None:
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        raise ValueError("Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": escape_markdown(message),
        "parse_mode": "MarkdownV2"
    }

    response = requests.post(url, json=payload)

    if response.status_code != 200:
        raise Exception(f"Failed to send message: {response.text}")

    return response.json()
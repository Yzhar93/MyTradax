import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()


def escape_markdown(text):
    escape_chars = r"_*[]()~`>#+-=|{}.! "
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)


def _token() -> str:
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN_INTERACTIVE")
    if not token:
        raise ValueError("Neither TELEGRAM_BOT_TOKEN nor TELEGRAM_BOT_TOKEN_INTERACTIVE is set.")
    return token


def send_telegram_message(message, chat_id=None, reply_markup=None, parse_mode="MarkdownV2"):
    """
    Send a new message to a Telegram chat.

    - chat_id: defaults to TELEGRAM_CHAT_ID env var
    - reply_markup: optional inline keyboard dict
    - parse_mode: "MarkdownV2" (default, escapes text) or None (plain text)
    """
    if chat_id is None:
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not chat_id:
        raise ValueError("chat_id is not set.")

    text = escape_markdown(message) if parse_mode == "MarkdownV2" else message

    payload = {"chat_id": chat_id, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    if reply_markup:
        payload["reply_markup"] = reply_markup

    resp = requests.post(f"https://api.telegram.org/bot{_token()}/sendMessage", json=payload)
    if resp.status_code != 200:
        raise Exception(f"sendMessage failed: {resp.text}")
    return resp.json()


def edit_telegram_message(chat_id, message_id, text, reply_markup=None):
    """Edit an existing bot message (plain text, no markdown escaping)."""
    payload = {
        "chat_id":    chat_id,
        "message_id": message_id,
        "text":       text,
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup

    resp = requests.post(
        f"https://api.telegram.org/bot{_token()}/editMessageText",
        json=payload,
    )
    # 400 "message is not modified" is harmless
    if resp.status_code != 200 and "message is not modified" not in resp.text:
        raise Exception(f"editMessageText failed: {resp.text}")
    return resp.json()


def answer_callback_query(callback_query_id: str):
    """Dismiss the loading spinner on an inline button tap."""
    requests.post(
        f"https://api.telegram.org/bot{_token()}/answerCallbackQuery",
        json={"callback_query_id": callback_query_id},
    )

import logging

from tradax.bot.interactive.menus import (
    main_menu, asset_list_menu, action_menu, timerange_menu
)
from tradax.bot.interactive.analysis import run_analysis
from tradax.bot.interactive.backtest import run_backtest
from tradax.bot.telegram_client import (
    send_telegram_message, edit_telegram_message, answer_callback_query
)


def handle_message(message: dict) -> None:
    text    = message.get("text", "")
    chat_id = message["chat"]["id"]

    if text in ("/start", "/menu"):
        send_telegram_message(
            "Choose a market:",
            chat_id=chat_id,
            reply_markup=main_menu(),
            parse_mode=None,
        )


def handle_callback(callback: dict) -> None:
    chat_id  = callback["message"]["chat"]["id"]
    msg_id   = callback["message"]["message_id"]
    cb_id    = callback["id"]
    data     = callback.get("data", "")
    parts    = data.split(":")

    answer_callback_query(cb_id)

    try:
        tag = parts[0]

        if tag == "menu":
            target = parts[1]
            if target == "main":
                edit_telegram_message(chat_id, msg_id, "Choose a market:", main_menu())
            elif target in ("stocks", "crypto"):
                edit_telegram_message(chat_id, msg_id, "Pick an asset:", asset_list_menu(target))

        elif tag == "pick":
            _, market, symbol = parts
            edit_telegram_message(
                chat_id, msg_id,
                f"{symbol} — choose action:",
                action_menu(market, symbol),
            )

        elif tag == "action":
            action = parts[1]
            market = parts[2]
            symbol = parts[3]

            if action == "analysis":
                edit_telegram_message(chat_id, msg_id, "Running analysis...")
                result = run_analysis(market, symbol)
                edit_telegram_message(chat_id, msg_id, result, action_menu(market, symbol))

            elif action == "backtest":
                if len(parts) == 4:
                    edit_telegram_message(
                        chat_id, msg_id,
                        f"{symbol} — choose time range:",
                        timerange_menu(market, symbol),
                    )
                else:
                    period = parts[4]
                    edit_telegram_message(chat_id, msg_id, f"Running backtest ({period})...")
                    result = run_backtest(market, symbol, period)
                    edit_telegram_message(chat_id, msg_id, result, timerange_menu(market, symbol))

    except Exception as e:
        logging.error(f"handle_callback error: {e}")
        edit_telegram_message(chat_id, msg_id, "Something went wrong. Try /menu again.")

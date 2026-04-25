import os
import requests
from dotenv import load_dotenv
import logging

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_API_URL = os.environ.get("GEMINI_API_URL")

import os
# import google.generativeai as genai
import google.genai as genai
from tradax.helpers.utils import retry_with_backoff
# genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
#
# # Initialize the client (API key should already be set via environment)
# client = genai.Client()

def enhance_message(results):
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    prompt = f"""
You are a professional financial assistant. Here are the top S&P 500 stocks with their recent percentage changes:

{results}

Please provide output in two parts:
1. **Stock summary:** List all the stocks exactly as given, with their percentage changes.
2. **Actionable advice:** Give a short, clear summary and investment advice **relevant to these stocks as a group**. Focus only on trends, dominant performers, or patterns in the provided list. Keep the advice concise (3–5 sentences).

Do not add generic or unrelated information.
    """

    try:
        return _generate_with_retry(client, prompt)
    except Exception as e:
        logging.error(f"❌ Error generating message with Gemini: {e}")
        return results



client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

@retry_with_backoff(retries=3, backoff_in_seconds=2)
def _generate_with_retry(client_instance, prompt):
    resp = client_instance.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return resp.text

@retry_with_backoff(retries=3, backoff_in_seconds=2)
def _generate_with_retry_extra(client_instance, prompt):
    resp = client_instance.models.generate_content(
        model="gemini-2.5-flash",
        contents=[{"text": prompt}]
    )
    return resp.text

def enhance_message_advance(results):
    prompt = f"""
    You are a professional financial assistant creating a concise, Telegram-friendly message.

    Here is the recent S&P 500 stock data:
    {results}

    Please format the output in **four separate sections**:

    1. 📅 Daily Movers:
       - List only the stocks relevant for daily changes.
       - Each stock on a separate line.
       - Show Ticker, RSI, DailyChange, VolumeSpike.
       - Add an **emoji for up (🔼), down (🔽), or neutral (⏺️)** based on the daily change.
       - Keep it clean and readable.

    2. 📈 Weekly Movers:
       - Same as above, but focus on weekly change.

    3. 📆 Monthly Movers:
       - Same as above, but focus on monthly change.

    4. 🔁 Intersection Movers:
       - List any stocks that appear in multiple timeframes.
       - Keep formatting consistent.

    Finally, in 💡 Insight & Advice:
       - Give a short summary (3–5 sentences).
       - Focus on trends, dominant performers, and patterns in these lists only.
       - Make advice actionable and **relevant only to these stocks**.
    """
    try:
        return _generate_with_retry(client, prompt)
    except Exception as e:
        logging.error(f"❌ Error in Gemini API call: {e}")
        return results



def enhance_message_extra(results):
    prompt = f"""
You are a professional financial assistant creating a concise, Telegram-friendly message.

Here is the recent S&P 500 stock data:
{results}

Please format the output in **four separate sections**:

1. 📅 Daily Movers:
   - List only the stocks relevant for daily changes.
   - Each stock on a separate line.
   - Show Ticker, RSI, DailyChange, VolumeSpike, and Signal.
   - Add an **emoji for up (🔼), down (🔽), or neutral (⏺️)** based on the daily change.
   - Keep it clean and readable.

2. 📈 Weekly Movers:
   - Same as above, but focus on weekly change.

3. 📆 Monthly Movers:
   - Same as above, but focus on monthly change.

4. 🔁 Intersection Movers:
   - List any stocks that appear in multiple timeframes.
   - Include the Signal for each stock.
   - Keep formatting consistent.

Finally, in 💡 Insight & Advice:
   - Give a short summary (3–5 sentences).
   - Focus on trends, dominant performers, and patterns in these lists only.
   - Include the Signal to justify Buy/Sell/Hold suggestions.
   - Keep advice actionable and **relevant only to these stocks**.
"""
    try:
        return _generate_with_retry_extra(client, prompt)
    except Exception as e:
        logging.error(f"❌ Error in Gemini API call: {e}")
        return results
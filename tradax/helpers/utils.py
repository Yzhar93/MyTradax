import time
import logging
from functools import wraps

def retry_with_backoff(retries=3, backoff_in_seconds=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    if "429" in error_str or "quota" in error_str:
                        logging.error(f"🚫 Quota Limit Reached (429). Aborting retries for {func.__name__}.")
                        raise e
                    
                    if attempt >= retries:
                        raise e
                    sleep_time = backoff_in_seconds * (2 ** attempt)
                    logging.warning(f"⚠️ Retrying {func.__name__} in {sleep_time}s due to: {str(e)}")
                    time.sleep(sleep_time)
                    attempt += 1
        return wrapper
    return decorator

import os
import time
import requests
import logging
from requests.exceptions import RequestException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SarvamAPIClient:
    def __init__(self, api_key: str = None, max_retries: int = 5, backoff_factor: float = 2.0):
        self.api_key = api_key or os.environ.get("SARVAM_API_KEY")
        if not self.api_key:
            raise ValueError("SARVAM_API_KEY must be provided or set as environment variable.")
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.headers = {
            "api-subscription-key": f"{self.api_key}"
        }

    def with_retry(self, func, *args, **kwargs):
        retries = 0
        while retries <= self.max_retries:
            try:
                response = func(*args, **kwargs)
                if response.status_code == 429:
                    logger.warning("Rate limit exceeded (429). Backing off...")
                elif response.status_code == 200:
                    return response
                else:
                    logger.error(f"API Error: {response.status_code} - {response.text}")
                    # Usually we want to retry on 500s too
                    if response.status_code < 500 and response.status_code not in [429]:
                         return response # Return non-retriable error for parsing
            except RequestException as e:
                logger.error(f"Request failed: {e}")
            
            wait_time = self.backoff_factor * (2 ** retries)
            logger.info(f"Retrying in {wait_time} seconds (Attempt {retries + 1}/{self.max_retries})")
            time.sleep(wait_time)
            retries += 1
            
        raise Exception(f"Failed after {self.max_retries} retries.")

    def post(self, url, **kwargs):
        # Merge headers
        headers = kwargs.get('headers', {})
        local_headers = self.headers.copy()
        local_headers.update(headers)
        kwargs['headers'] = local_headers
        
        def _make_req():
            return requests.post(url, **kwargs)
            
        return self.with_retry(_make_req)

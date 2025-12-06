import time
import requests
from typing import List
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from ..utils.logger_setup import setup_logger

logger = setup_logger(
    logger_name="web_scraper.py", 
    filename="web_scraper.log"
)


class WebScraper:
    """
    A simple web scraping utility class to fetch and extract webpage content.
    """

    def __init__(self, request_timeout: int = 10) -> None:
        """Initializes the WebScraper instance."""
        self.request_timeout = request_timeout

    def is_valid_url(self, url: str) -> bool:
        """
        Validates whether the given input is a valid URL.
        
        Args:
            url: The string to validate as a URL
            
        Returns:
            bool: True if the input is a valid URL, False otherwise
        """
        try:
            result = urlparse(url)
            # CHECK FOR SCHEME AND NETLOC
            return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
        except Exception:
            return False

    def process(self, url: str) -> List[str]:
        """Fetches and returns the main textual content of a given URL."""
        # VALIDATE URL
        if not self.is_valid_url(url):
            error_msg = f"Invalid URL provided: '{url}'"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        processing_start_time = time.time()
        logger.info(f"Scraping URL: '{url}' ...")
        try:
            response = requests.get(url, timeout=self.request_timeout)
            response.raise_for_status()    # RAISES HTTPError FOR BAD RESPONSES
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch URL '{url}'! {e}")

        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        logger.info(
            f"URL: '{url}' processed in "
            f"{time.time() - processing_start_time:.2f} seconds"
        )
        return [text]


# if __name__ == "__main__":
#     scraper = WebScraper()
#     url = ""
#     result = scraper.process(url)
#     print(result)
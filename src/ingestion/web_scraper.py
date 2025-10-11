import time
import requests
from bs4 import BeautifulSoup
from ..utils.logger_setup import setup_logger

logger = setup_logger("web_scraper.py")


class WebScraper:
    """
    A simple web scraping utility class to fetch and extract webpage content.
    """

    def __init__(self, request_timeout: int = 10) -> None:
        """Initializes the WebScraper instance."""
        self.request_timeout = request_timeout

    def __call__(self, url: str) -> str:
        """Fetches and returns the main textual content of a given URL."""
        processing_start_time = time.time()
        logger.info(f"Scraping URL: '{url}' ...")
        try:
            response = requests.get(url, timeout=self.request_timeout)
            response.raise_for_status()    # RAISES HTTPError FOR BAD RESPONSES
        except requests.RequestException as e:
            return f"Error fetching URL: {e}"

        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        logger.info(
            f"URL: '{url}' processed in "
            f"{time.time() - processing_start_time:.2f} seconds"
        )
        return text


# if __name__ == "__main__":
#     scraper = WebScraper()
#     url = ""
#     result = scraper(url)
#     print(result)
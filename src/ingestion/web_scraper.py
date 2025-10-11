import requests
from bs4 import BeautifulSoup


class WebScraper:
    """
    A simple web scraping utility class to fetch and extract webpage content.
    """

    def __init__(self, request_timeout: int = 10) -> None:
        """Initializes the WebScraper instance."""
        self.request_timeout = request_timeout

    def __call__(self, url: str) -> str:
        """Fetches and returns the main textual content of a given URL."""
        try:
            response = requests.get(url, timeout=self.request_timeout)
            response.raise_for_status()    # RAISES HTTPError FOR BAD RESPONSES
        except requests.RequestException as e:
            return f"Error fetching URL: {e}"

        soup = BeautifulSoup(response.text, "html.parser")

        print(len(soup.get_text(separator="\n", strip=True)))

        text = soup.get_text(separator="\n", strip=True)
        return text


# if __name__ == "__main__":
#     scraper = WebScraper()
#     url = ""
#     result = scraper(url)
#     print(result)
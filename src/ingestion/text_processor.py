import time
from typing import List
from ..utils.logger_setup import setup_logger
from ..utils.common import validate_file_location, validate_file_type

logger = setup_logger("text_processor.py")


class TextProcessor():
    """Class for Text files processing"""

    def __init__(self) -> None:
        """Initializes the necessary parameters"""
        self.valid_extensions = [".txt", ".md", ".log"]

    def validate_txt_file(self, filepath: str) -> None:
        """Validates if the given file exists and is a text file"""
        validate_file_location(filepath=filepath)
        validate_file_type(
            filepath=filepath,
            valid_extensions=self.valid_extensions
        )

    def process(self, filepath: str) -> List[str]:
        """Processes the text file and returns text found"""
        try:
            self.validate_txt_file(filepath=filepath)
            processing_start_time = time.time()
            logger.info(f"Processing file: '{filepath}' ...")
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
            logger.info(
                f"File: '{filepath}' processed in "
                f"{time.time() - processing_start_time:.2f} seconds"
            )
            return [text]
        except Exception as e:
            raise RuntimeError(f"Failed to process '{filepath}' text file! {e}")


# EXAMPLE USAGE
# if __name__ == "__main__":
#     processor = TextProcessor()
#     results = processor.process(
#         filepath="",
#     )

#     for page in results:
#         print("******************************")
#         print(page)
#         print("******************************")
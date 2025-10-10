import time
from typing import List
from abc import ABC, abstractmethod
from langchain_community.document_loaders import Docx2txtLoader
from ..utils.logger_setup import setup_logger
from ..utils.common import validate_file_location, validate_file_type

logger = setup_logger("doc_processor.py")


class BaseDocProcessor(ABC):
    """Abstract base class for Word Document processing"""

    def __init__(self, filepath: str) -> None:
        """Initializes the file path"""
        self.valid_extensions = {".doc", ".docx"}
        self.filepath = filepath
        self.validate_doc_file()

    def validate_doc_file(self) -> None:
        """Validates if the given file exists and is a Document"""
        validate_file_location(filepath=self.filepath)
        validate_file_type(
            filepath=self.filepath,
            valid_extensions=self.valid_extensions
        )

    @abstractmethod
    def process(self) -> List[str]:
        """Processes the Document and returns text found for each page"""
        pass


class Doc2TxtProcessor(BaseDocProcessor):
    """Class for Document processing using Docx2txt library"""

    def __init__(self, filepath: str) -> None:
        """
        Initializes the Docx2txt Loader

        Args:
            filepath (str): Path to the Document file
        """
        try:
            super().__init__(filepath)
            self.loader = Docx2txtLoader(file_path=self.filepath)
        except Exception as e:
            logger.error(f"Error initializing Docx2txtLoader: {e}")
            raise

    def process(self) -> List[str]:
        """Processes the Document and returns text found"""
        try:
            processing_start_time = time.time()
            logger.info(f"Processing file: '{self.filepath}' using PyMuPDF4LLM...")
            documents = self.loader.load()
            if not documents or not documents[0].page_content.strip():
                logger.warning("No text found in the document.")
                return []
            logger.info(
                f"File: '{self.filepath}' processed in "
                f"{time.time() - processing_start_time:.2f} seconds"
            )
            return [documents[0].page_content.strip()]
        except Exception as e:
            raise RuntimeError(f"Failed to process '{self.filepath}' document using Doc2Txt! {e}")


# EXAMPLE USAGE
# if __name__ == "__main__":
#     processor = Doc2TxtProcessor(
#         filepath="",
#     )
#     results = processor.process()

#     for page in results:
#         print("******************************")
#         print(page)
#         print("******************************")
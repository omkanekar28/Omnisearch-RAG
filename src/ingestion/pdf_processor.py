import os
import time
from typing import List, Optional
from abc import ABC, abstractmethod
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.document_loaders.base import BaseBlobParser
from ..utils.logger_setup import setup_logger

logger = setup_logger("pdf_processor.py")


class BasePDFProcessor(ABC):
    """Abstract base class for PDF processing"""

    def __init__(self, filepath: str) -> None:
        """Initializes the file path"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Provided file '{filepath}' does not exist!")
        self.filepath = filepath

    @abstractmethod
    def process(self) -> List[str]:
        """Processes the PDF and returns text found for each page"""
        pass


class PyMuPDF4LLMPDFProcessor(BasePDFProcessor):
    """Class for PDF processing using PyMuPDF4LLM library"""

    def __init__(
        self,
        filepath: str,
        extract_images: bool = False,
        images_parser: Optional[BaseBlobParser] = None,
    ) -> None:
        """
        Initializes the PyMuPDF4LLM Loader

        Args:
            filepath (str): Path to the PDF file
            extract_images (bool): Whether to extract images from the PDF
            images_parser (Optional[BaseBlobParser]): Parser to use for image extraction
        """
        try:
            super().__init__(filepath)

            self.loader = PyMuPDF4LLMLoader(
                file_path=self.filepath,
                mode="page",
                extract_images=extract_images,
                images_parser=images_parser
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PyMuPDF4LLM Loader! {e}")
    
    def process(self) -> List[str]:
        """Processes the PDF and returns extracted text for each page"""
        try:
            processing_start_time = time.time()
            logger.info(f"Processing file: '{self.filepath}' using PyMuPDF4LLM...")
            pages = self.loader.load()
            logger.info(
                f"File: '{self.filepath}' having {len(pages)} pages processed "
                f"in {time.time() - processing_start_time:.2f} seconds"
            )
            return [page.page_content or "" for page in pages]
        except Exception as e:
            raise RuntimeError(f"Failed to process PDF using PyMuPDF4LLM! {e}")

# EXAMPLE USAGE
# if __name__ == "__main__":
#     from langchain_community.document_loaders.parsers import TesseractBlobParser
#     processor = PyMuPDF4LLMPDFProcessor(
#         filepath="",
#         extract_images=True,
#         images_parser=TesseractBlobParser()
#     )
#     results = processor.process()

#     for page in results:
#         print("******************************")
#         print(page)
#         print("******************************")
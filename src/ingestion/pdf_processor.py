import time
from typing import List, Literal, Optional
from abc import ABC, abstractmethod
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.document_loaders.base import BaseBlobParser
from ..utils.logger_setup import setup_logger
from ..utils.common import validate_file_location, validate_file_type

logger = setup_logger(
    logger_name="pdf_preprocessor.py", 
    filename="pdf_preprocessor.log"
)


class BasePDFProcessor(ABC):
    """Abstract base class for PDF processing"""

    def __init__(self) -> None:
        """Initializes the file path"""
        self.valid_extensions = {".pdf"}

    def validate_pdf_file(self, filepath: str) -> None:
        """Validates if the given file exists and is a PDF"""
        validate_file_location(filepath=filepath)
        validate_file_type(
            filepath=filepath,
            valid_extensions=self.valid_extensions
        )

    @abstractmethod
    def process(self, filepath: str) -> List[str]:
        """Processes the PDF and returns text found for each page"""
        pass


class PyMuPDF4LLMPDFProcessor(BasePDFProcessor):
    """Class for PDF processing using PyMuPDF4LLM library"""

    def __init__(
        self,
        mode: Literal["single", "page"] = "page",
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
            super().__init__()
            self.mode = mode
            self.extract_images = extract_images
            self.images_parser = images_parser
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PyMuPDF4LLM Loader! {e}")
    
    def process(self, filepath: str) -> List[str]:
        """Processes the PDF and returns extracted text for each page"""
        try:
            processing_start_time = time.time()
            self.validate_pdf_file(filepath=filepath)
            loader = PyMuPDF4LLMLoader(
                file_path=filepath,
                mode=self.mode,
                extract_images=self.extract_images,
                images_parser=self.images_parser
            )
            logger.info(f"Processing file: '{filepath}' using PyMuPDF4LLM...")
            pages = loader.load()
            results = [page.page_content or "" for page in pages]

            # ADDING PAGE NUMBER HEADERS
            results_with_headers = []
            for idx, content in enumerate(results):
                results_with_headers.append(f"--- Page {idx + 1} ---")
                results_with_headers.append(content.strip())

            logger.info(
                f"File: '{filepath}' having {len(pages)} pages processed "
                f"in {time.time() - processing_start_time:.2f} seconds"
            )
            return results_with_headers
        except Exception as e:
            raise RuntimeError(f"Failed to process '{filepath}' PDF using PyMuPDF4LLM! {e}")

# EXAMPLE USAGE
# if __name__ == "__main__":
#     from langchain_community.document_loaders.parsers import TesseractBlobParser
#     processor = PyMuPDF4LLMPDFProcessor(
#         mode="page",
#         extract_images=True,
#         images_parser=TesseractBlobParser()
#     )
#     results = processor.process(filepath="")

#     for page in results:
#         print(page)
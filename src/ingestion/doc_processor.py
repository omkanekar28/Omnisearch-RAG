import time
from typing import List, Literal
from abc import ABC, abstractmethod
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredWordDocumentLoader
from ..utils.logger_setup import setup_logger
from ..utils.common import validate_file_location, validate_file_type

logger = setup_logger("doc_processor.py")


class BaseDocProcessor(ABC):
    """Abstract base class for Word Document processing"""

    def __init__(self) -> None:
        """Initializes the file path"""
        self.valid_extensions = {".doc", ".docx"}

    def validate_doc_file(self, filepath: str) -> None:
        """Validates if the given file exists and is a Document"""
        validate_file_location(filepath=filepath)
        validate_file_type(
            filepath=filepath,
            valid_extensions=self.valid_extensions
        )

    @abstractmethod
    def process(self, filepath: str) -> List[str]:
        """Processes the Document and returns text found"""
        pass


class Doc2TxtDocProcessor(BaseDocProcessor):
    """Class for Document processing using Docx2txt library (.doc not supported)"""

    def __init__(self) -> None:
        """
        Initializes the Docx2txt Loader

        Args:
            filepath (str): Path to the Document file
        """
        try:            
            super().__init__()
        except Exception as e:
            logger.error(f"Error initializing Docx2txtLoader: {e}")
            raise

    def process(self, filepath: str) -> List[str]:
        """Processes the Document using Doc2Txt and returns text found"""
        try:
            processing_start_time = time.time()
            # THROW ERROR IF .doc FILE IS PROVIDED
            if filepath.lower().endswith(".doc"):
                raise ValueError("Docx2txtLoader does not support .doc files! "
                                 "Please use UnstructuredDocProcessor for .doc files.")
            self.validate_doc_file(filepath=filepath)

            logger.info(f"Processing file: '{filepath}' using Doc2Txt...")
            loader = Docx2txtLoader(file_path=filepath)
            documents = loader.load()
            if not documents or not documents[0].page_content.strip():
                logger.warning("No text found in the document.")
                return []
            logger.info(
                f"File: '{filepath}' processed in "
                f"{time.time() - processing_start_time:.2f} seconds"
            )
            return [documents[0].page_content.strip()]
        except Exception as e:
            raise RuntimeError(f"Failed to process '{filepath}' document using Doc2Txt! {e}")


class UnstructuredDocProcessor(BaseDocProcessor):
    """Class for Document processing using Docx2txt library"""

    def __init__(
        self, 
        mode: Literal["single", "elements", "paged"] = "single"
    ) -> None:
        """
        Initializes the Unstructured Loader

        Args:
            filepath (str): Path to the Document file
        """
        try:
            super().__init__()
            self.mode = mode
        except Exception as e:
            logger.error(f"Error initializing UnstructuredWordDocumentLoader: {e}")
            raise

    def process(self, filepath: str) -> List[str]:
        """Processes the Document using UnstructuredWordDocumentLoader and returns text found"""
        try:
            processing_start_time = time.time()
            loader = UnstructuredWordDocumentLoader(
                file_path=filepath,
                mode=self.mode
            )
            self.validate_doc_file(filepath=filepath)

            logger.info(f"Processing file: '{filepath}' using UnstructuredDocProcessor...")
            pages = loader.load()
            results = [page.page_content or "" for page in pages]

            logger.info(
                f"File: '{filepath}' processed in "
                f"{time.time() - processing_start_time:.2f} seconds"
            )
            return results
        except Exception as e:
            raise RuntimeError(f"Failed to process '{filepath}' document "
                               f"using UnstructuredWordDocumentLoader! {e}")


# EXAMPLE USAGE
# if __name__ == "__main__":
#     processor = UnstructuredDocProcessor(
#         filepath="",
#         mode="single"
#     )
#     results = processor.process()

#     for page in results:
#         print(page)
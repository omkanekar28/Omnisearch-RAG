import re
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from abc import ABC, abstractmethod
from ..utils.logger_setup import setup_logger

logger = setup_logger("text_chunker.py")


class BaseChunker(ABC):
    """Base class for chunkers"""
    
    def __init__(
            self, 
            chunk_size,
            chunk_overlap,
    ) -> None:
        """"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.validate_init_params()

    def validate_init_params(self) -> None:
        """Validates the initialization parameters"""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap '{self.chunk_overlap}' must be smaller "
                             f"than chunk_size '{self.chunk_size}'!")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap '{self.chunk_overlap}' must be non-negative!")
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size '{self.chunk_size}' must be positive!")
        if self.chunk_overlap == 0:
            logger.warning("chunk_overlap is set to 0. Chunks will be non-overlapping.")
    
    @abstractmethod
    def __call__(self, text: str) -> List[str]:
        """Chunks the input text and returns a list of chunks"""
        pass


class CustomTextChunker(BaseChunker):
    """Class to chunk text based on max character threshold and overlap"""

    def __init__(
            self, 
            chunk_size: int = 200,
            chunk_overlap: int = 20,
    ) -> None:
        """Initializes the chunker with threshold and overlap"""
        super().__init__(chunk_size, chunk_overlap)
        self.word_splitting_pattern = r'\S+(?:\s+|$)'
        logger.info(f"CustomTextChunker initialized with chunk_size: "
                    f"{self.chunk_size}, chunk_overlap: {self.chunk_overlap}")

    def __call__(self, text: str) -> List[str]:
        """Chunks the given text into smaller parts based on the chunk_size and chunk_overlap"""
        logger.info(f"Chunking text having length: {len(text)} ...")
        words = re.findall(self.word_splitting_pattern, text)

        if not len(words) > self.chunk_size:
            logger.info("Text length is within the threshold. No chunking needed.")
            return [text]
        
        chunks = []
        current_chunk = []

        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = current_chunk[-self.chunk_overlap:]    # RETAIN OVERLAP WORDS
        
        # APPEND ANY REMAINING WORDS AS THE LAST CHUNK
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        logger.info(f"Text chunked into {len(chunks)} chunks.")
        return chunks


class LangchainTextChunker(BaseChunker):
    """Class to chunk text using Langchain's RecursiveCharacterTextSplitter"""

    def __init__(
            self, 
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
    ) -> None:
        """Initializes the chunker with threshold and overlap"""
        super().__init__(chunk_size, chunk_overlap)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        logger.info(f"LangchainTextChunker initialized with chunk_size: "
                    f"{self.chunk_size}, chunk_overlap: {self.chunk_overlap}")

    def __call__(self, text: str) -> List[str]:
        """Chunks the given text into smaller parts based on the chunk_size and chunk_overlap"""
        logger.info(f"Chunking text having length: {len(text)} ...")
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Text chunked into {len(chunks)} chunks.")
        return chunks


# if __name__ == "__main__":
#     custom_chunker = CustomTextChunker()
#     text = """"""
#     chunks = custom_chunker(text)
#     for chunk in chunks:
#         print(f'\n---- Chunk ----\n{chunk}\n')
    
#     langchain_chunker = LangchainTextChunker()
#     lc_chunks = langchain_chunker(text)
#     for lc_chunk in lc_chunks:
#         print(f'\n---- LC Chunk ----\n{lc_chunk}\n')
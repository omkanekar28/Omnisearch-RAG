import re
from typing import List
from ..utils.logger_setup import setup_logger

logger = setup_logger("text_chunker.py")


class TextChunker:
    """Class to chunk text based on max character threshold and overlap"""

    def __init__(
            self, 
            max_words_threshold: int = 200,
            words_overlap: int = 20,
    ) -> None:
        """Initializes the chunker with threshold and overlap"""
        self.word_splitting_pattern = r'\S+(?:\s+|$)'
        self.max_words_threshold = max_words_threshold
        self.words_overlap = words_overlap
        self.validate_init_params()
    
    def validate_init_params(self) -> None:
        """Validates the initialization parameters"""
        if self.words_overlap >= self.max_words_threshold:
            raise ValueError(f"words_overlap '{self.words_overlap}' must be smaller "
                             f"than max_words_threshold '{self.max_words_threshold}'!")
        if self.words_overlap < 0:
            raise ValueError(f"words_overlap '{self.words_overlap}' must be non-negative!")
        if self.max_words_threshold <= 0:
            raise ValueError(f"max_words_threshold '{self.max_words_threshold}' must be positive!")
        
        logger.info(f"TextChunker initialized with max_words_threshold: "
                    f"{self.max_words_threshold}, words_overlap: {self.words_overlap}")
        if self.words_overlap == 0:
            logger.warning("words_overlap is set to 0. Chunks will be non-overlapping.")

    def __call__(self, text: str) -> List[str]:
        """Chunks the given text into smaller parts based on the max_words_threshold and words_overlap"""
        logger.info(f"Chunking text having length: {len(text)} ...")
        words = re.findall(self.word_splitting_pattern, text)

        if not len(words) > self.max_words_threshold:
            logger.info("Text length is within the threshold. No chunking needed.")
            return [text]
        
        chunks = []
        current_chunk = []

        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= self.max_words_threshold:
                chunks.append(" ".join(current_chunk))
                current_chunk = current_chunk[-self.words_overlap:]    # RETAIN OVERLAP WORDS
        
        # APPEND ANY REMAINING WORDS AS THE LAST CHUNK
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        logger.info(f"Text chunked into {len(chunks)} chunks.")
        return chunks


# if __name__ == "__main__":
#     chunker = TextChunker()
#     text = """"""
#     chunks = chunker(text)
#     for chunk in chunks:
#         print(f'\n---- Chunk ----\n{chunk}\n')
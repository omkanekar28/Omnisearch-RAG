from typing import List
from ..utils.logger_setup import setup_logger

logger = setup_logger(
    logger_name="excel_chunker.py", 
    filename="excel_chunker.log"
)


class ExcelChunker:
    """Class to chunk excel based on max character threshold and overlap"""

    def __init__(
            self, 
            max_rows_threshold: int = 20,
            rows_overlap: int = 5,
    ) -> None:
        """Initializes the chunker with threshold and overlap"""
        self.max_rows_threshold = max_rows_threshold
        self.rows_overlap = rows_overlap
        self.validate_init_params()
    
    def validate_init_params(self) -> None:
        """Validates the initialization parameters"""
        if self.rows_overlap >= self.max_rows_threshold:
            raise ValueError(f"rows_overlap '{self.rows_overlap}' must be smaller "
                             f"than max_rows_threshold '{self.max_rows_threshold}'!")
        if self.rows_overlap < 0:
            raise ValueError(f"rows_overlap '{self.rows_overlap}' must be non-negative!")
        if self.max_rows_threshold <= 0:
            raise ValueError(f"max_rows_threshold '{self.max_rows_threshold}' must be positive!")
        
        logger.info(f"ExcelChunker initialized with max_rows_threshold: "
                    f"{self.max_rows_threshold}, rows_overlap: {self.rows_overlap}")
        if self.rows_overlap == 0:
            logger.warning("rows_overlap is set to 0. Chunks will be non-overlapping.")

    def __call__(self, text: str) -> List[str]:
        """Chunks the given text into smaller parts based on the max_rows_threshold and rows_overlap"""
        rows = text.split('\n')
        header = rows[0] if rows else ""
        logger.info(f"Chunking text having {len(rows)} rows ...")

        if len(rows) <= self.max_rows_threshold:
            logger.info("Text length is within the threshold. No chunking needed.")
            return [text]
        
        chunks = []

        for i in range(0, len(rows), self.max_rows_threshold - self.rows_overlap):
            chunk = rows[i:i + self.max_rows_threshold]
            if i != 0:
                chunk.insert(0, header)  # ADD HEADER TO EACH CHUNK
            chunks.append('\n'.join(chunk))
            if i + self.max_rows_threshold >= len(rows):
                break  # BREAK IF END OF ROWS IS REACHED

        logger.info(f"Text having {len(rows)} rows chunked into {len(chunks)} chunks.")
        return chunks


# if __name__ == "__main__":
#     chunker = ExcelChunker()
#     text = """"""
#     chunks = chunker(text)
#     for chunk in chunks:
#         print(f'\n---- Chunk ----\n{chunk}\n')
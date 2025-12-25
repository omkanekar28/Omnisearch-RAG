import time
import numpy as np
import pandas as pd
from typing import List, Tuple
from ..ingestion.excel_processor import ExcelProcessor
from ..embeddings.embedding_handler import EmbeddingHandler
from ..vectorstore.vector_store import FaissVectorStore
from ..utils.logger_setup import setup_logger

logger = setup_logger(
    logger_name="embedding_indexer.py",
    filename="embedding_indexer.log"
)


class FaissEmbeddingIndexer:
    """class for converting extracted text chunks into vector store."""

    def __init__(
        self, 
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    ) -> None:
        """Initializes the embedding indexer."""
        initialization_start_time = time.time()
        logger.info("Initializing FAISS embedding indexer ...")
        self.excel_processor = ExcelProcessor()    # NEED EXCEL PROCESSOR JUST FOR FILE PATH VALIDATION
        self.embedding_handler = EmbeddingHandler(model_id=embedding_model)
        n_dims = self.embedding_handler.generate_embeddings(["Hello World!"]).shape[1]
        self.vector_store = FaissVectorStore(
            n_dims=n_dims, 
            use_gpu=True
        )
        logger.info("FAISS embedding indexer initialized in "
                    f"{time.time() - initialization_start_time:.2f} seconds")
    
    def get_embeddings_and_metadata(
        self, 
        excel_filepath: str, 
        column_to_extract: str = "Extracted Text"
    ) -> Tuple[np.ndarray, List[dict]]:
        """Generates and returns embeddings and metadata for the given excel file."""
        self.excel_processor.validate_excel_file(excel_filepath)
        excel_df = pd.read_excel(excel_filepath)
        all_texts = excel_df[column_to_extract].dropna().astype(str).tolist()
        embeddings = self.embedding_handler.generate_embeddings(all_texts)
        metadata = excel_df.to_dict(orient="records")
        return embeddings, metadata

    def create_faiss_store(
        self, 
        vectors: np.ndarray,
        metadata: List[dict], 
        store_dir: str = "faiss_store"
    ) -> None:
        """Creates FAISS vector store from the given vectors and metadata."""
        try:
            store_creation_start_time = time.time()
            logger.info("Creating FAISS vector store ...")
            self.vector_store.populate_store(
                vectors=vectors,
                metadata=metadata,
                store_dir=store_dir
            )
            logger.info("FAISS vector store created in "
                        f"{time.time() - store_creation_start_time:.2f} seconds.")
        except Exception as e:
            raise RuntimeError(f"Failed to create FAISS vector store! {e}")

    def process_excel_to_faiss_store(
        self, 
        excel_filepath: str, 
        store_dir: str = "faiss_store",
        column_to_extract: str = "Extracted Text"
    ) -> None:
        """Processes the given excel file and creates FAISS vector store."""
        try:
            processing_start_time = time.time()
            logger.info(f"Processing Excel file '{excel_filepath}' to create FAISS store ...")
            embeddings, metadata = self.get_embeddings_and_metadata(
                excel_filepath=excel_filepath,
                column_to_extract=column_to_extract
            )
            self.create_faiss_store(
                vectors=embeddings,
                metadata=metadata,
                store_dir=store_dir
            )
            logger.info(
                f"Excel file '{excel_filepath}' processed and FAISS store created in "
                f"{time.time() - processing_start_time:.2f} seconds."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to process Excel file to FAISS store! {e}")


# if __name__ == "__main__":
#     indexer = FaissEmbeddingIndexer()
#     excel_file = "results.xlsx"
#     indexer.process_excel_to_faiss_store(
#         excel_filepath=excel_file,
#         store_dir="faiss_store",
#         column_to_extract="Extracted Text"
#     )
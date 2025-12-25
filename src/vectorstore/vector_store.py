import os
import time
import json
import faiss
import numpy as np
from typing import List, Literal
from ..utils.logger_setup import setup_logger

logger = setup_logger(
    logger_name="vector_store.py",
    filename="vector_store.log"
)


class FaissVectorStore:
    def __init__(
        self, 
        n_dims: int = 768,
        faiss_index: Literal["FlatL2", "FlatIP"] = "FlatL2", 
        use_gpu: bool = False
    ) -> None:
        """Initializes the vector store."""
        initialization_start_time = time.time()
        logger.info("Initializing FAISS vector store ...")
        self.n_dims = n_dims
        
        if faiss_index == "FlatL2":
            self.faiss_index = faiss.IndexFlatL2(n_dims)
            if use_gpu:
                logger.info("Using GPU for FAISS index ...")
                try:
                    resource = faiss.StandardGpuResources()
                except AttributeError:
                    raise RuntimeError("FAISS GPU resources not available! "
                                       "Ensure FAISS is installed with GPU support.")
                faiss.index_cpu_to_gpu(resource, 0, self.faiss_index)
        elif faiss_index == "FlatIP":
            self.faiss_index = faiss.IndexFlatIP(n_dims)
            if use_gpu:
                logger.info("Using GPU for FAISS index ...")
                try:
                    resource = faiss.StandardGpuResources()
                except AttributeError:
                    raise RuntimeError("FAISS GPU resources not available! "
                                       "Ensure FAISS is installed with GPU support.")
                faiss.index_cpu_to_gpu(resource, 0, self.faiss_index)
        else:
            raise ValueError("Unsupported FAISS index type!")
        logger.info("FAISS vector store initialized in "
                    f"{time.time() - initialization_start_time:.2f} seconds")

    def are_vectors_compatible(self, vectors: np.ndarray) -> bool:
        """Checks if the provided vectors are compatible with the store's dimensions."""
        for vector in vectors:
            if len(vector) != self.n_dims:
                return False
        return True
    
    def is_metadata_compatible(self, metadata: List[dict]) -> bool:
        """Checks if the provided metadata dictionaries have the required fields."""
        metadata_fields = metadata[0].keys()

        for data in metadata:
            if not all(field in data for field in metadata_fields):
                return False
        return True

    def populate_store(
        self, 
        vectors: np.ndarray,
        metadata: List[dict], 
        store_dir: str
    ) -> None:
        """Adds given chunks to the store"""
        try:
            store_populate_start_time = time.time()
            logger.info(f"Adding {len(vectors)} chunks to the store ...")
            os.makedirs(store_dir, exist_ok=True)
            metadata_fields = metadata[0].keys()

            # VALIDATIONS
            if len(vectors) != len(metadata):
                raise ValueError("Vectors and metadata lists must have the same length!")
            if not self.are_vectors_compatible(vectors):
                raise ValueError(f"All vectors must have {self.n_dims} dimensions!")
            if not self.is_metadata_compatible(metadata):
                raise ValueError(f"All metadata entries must contain the fields: {metadata_fields}!")
            
            # STORING VECTORS
            faiss_index_path = os.path.join(store_dir, "faiss_index.index")
            self.faiss_index.add(np.array(vectors).astype('float32'))
            faiss.write_index(self.faiss_index, faiss_index_path)

            # STORING METADATA (JSON)
            metadata_path = os.path.join(store_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            logger.info("Metadata written successfully.")

            logger.info(f"{len(vectors)} chunks added in "
                        f"{time.time() - store_populate_start_time:.2f} seconds")
        except Exception as e:
            raise RuntimeError(f"Failed to populate the vector store! {e}. Rolling back changes.")


# if __name__ == "__main__":
#     import random
#     input_embeddings = [[random.random() for _ in range(768)] for _ in range(2)]
#     metadata = [
#         {"id": 1, "name": "Alice"},
#         {"id": 2, "name": "Bob"}
#     ]
#     vector_store = FaissVectorStore(use_gpu=True)
#     vector_store.populate_store(input_embeddings, metadata, "test_faiss_dir")
    
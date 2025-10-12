import os
import time
import json
import faiss
import numpy as np
from typing import List, Iterable, Literal
from ..utils.logger_setup import setup_logger

logger = setup_logger("vector_store.py")


class FaissVectorStore:
    def __init__(
            self, 
            store_dir: str,
            metadata_fields: Iterable[str],
            n_dims: int = 768,
            faiss_index: Literal["FlatL2", "FlatIP"] = "FlatL2"
    ) -> None:
        """Initializes the vector store."""
        initialization_start_time = time.time()
        logger.info("Initializing FAISS vector store ...")
        os.makedirs(store_dir, exist_ok=True)
        self.store_dir = store_dir
        
        self.metadata_fields = metadata_fields
        self.n_dims = n_dims
        
        if faiss_index == "FlatL2":
            self.faiss_index = faiss.IndexFlatL2(n_dims)
        elif faiss_index == "FlatIP":
            self.faiss_index = faiss.IndexFlatIP(n_dims)
        else:
            raise ValueError("Unsupported FAISS index type!")
        logger.info("FAISS vector store initialized in "
                    f"{time.time() - initialization_start_time:.2f} seconds")

    def are_vectors_compatible(self, vectors: List[List[float]]) -> bool:
        """Checks if the provided vectors are compatible with the store's dimensions."""
        for vector in vectors:
            if len(vector) != self.n_dims:
                return False
        return True
    
    def is_metadata_compatible(self, metadata: List[dict]) -> bool:
        """Checks if the provided metadata dictionaries have the required fields."""
        for data in metadata:
            if not all(field in data for field in self.metadata_fields):
                return False
        return True

    def populate_store(
        self, 
        vectors: List[List[float]],
        metadata: List[dict]
    ) -> None:
        """Adds given chunks to the store"""
        store_populate_start_time = time.time()
        logger.info(f"Adding {len(vectors)} chunks to the store ...")
        # VALIDATIONS
        if len(vectors) != len(metadata):
            raise ValueError("Vectors and metadata lists must have the same length!")
        if not self.are_vectors_compatible(vectors):
            raise ValueError(f"All vectors must have {self.n_dims} dimensions!")
        if not self.is_metadata_compatible(metadata):
            raise ValueError(f"All metadata entries must contain the fields: {self.metadata_fields}!")
        
        # STORING VECTORS
        self.faiss_index.add(np.array(vectors).astype('float32'))
        faiss.write_index(self.faiss_index, os.path.join(self.store_dir, "faiss_index.index"))

        # STORING METADATA (JSON)
        metadata_json = json.dumps(metadata, indent=4)
        metadata_path = os.path.join(self.store_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            f.write(metadata_json)
        
        logger.info(f"{len(vectors)} chunks added in "
                    f"{time.time() - store_populate_start_time:.2f} seconds")


# if __name__ == "__main__":
#     import random
#     input_embeddings = [[random.random() for _ in range(768)] for _ in range(2)]
#     metadata = [
#         {"id": 1, "name": "Alice"},
#         {"id": 2, "name": "Bob"}
#     ]
#     vector_store = FaissVectorStore("test_faiss_dir", {"id", "name"})
#     vector_store.populate_store(input_embeddings, metadata)
    
import time
import torch
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from ..utils.logger_setup import setup_logger

logger = setup_logger("embedding_handler.py")

device = "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingHandler:
    def __init__(self, model_id: str = "google/embeddinggemma-300M") -> None:
        self.model_id = model_id
        model_loading_start_time = time.time()
        logger.info(f"Loading embedding model '{self.model_id}' ...")
        self.model = SentenceTransformer(self.model_id).to(device=device)
        logger.info(f"Embedding model '{self.model_id}' loaded in "
                    f"{time.time() - model_loading_start_time:.2f} seconds.")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generates embedding for the given text"""
        embeddings = []
        embedding_generation_start = time.time()
        logger.info(f"Generating embeddings using model '{self.model_id}' "
                    f"for {len(texts)} texts ...")
        embeddings = self.model.encode(texts)
        logger.info(f"Embeddings generated in "
                    f"{time.time() - embedding_generation_start:.2f} seconds.")
        return embeddings
    
    def semantic_search(self, text1: str, text2: str) -> float:
        """Computes semantic similarity between two texts"""
        if not text1 or not text2:
            raise ValueError("Input texts should be non-empty")
        embeddings = self.model.encode([text1, text2])
        similarity = self.model.similarity(embeddings[0], embeddings[1])
        return similarity.numpy()[0][0]


# if __name__ == "__main__":
#     handler = EmbeddingHandler()
#     texts = [
#         "I love fruits. Especially Apple.",
#         "I'm currently using Apple as my new phone."
#     ]
#     embeddings = handler.generate_embeddings(texts)
#     print(embeddings)
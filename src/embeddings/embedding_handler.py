import time
import torch
from sentence_transformers import SentenceTransformer
from ..utils.logger_setup import setup_logger

logger = setup_logger("embedding_handler.py")

device = "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingHandler:
    def __init__(self, model_id: str = "google/embeddinggemma-300M"):
        self.model_id = model_id
        model_loading_start_time = time.time()
        logger.info(f"Loading embedding model '{self.model_id}' ...")
        self.model = SentenceTransformer(self.model_id).to(device=device)
        logger.info(f"Embedding model '{self.model_id}' loaded in {time.time() - model_loading_start_time:.2f} seconds.")
    
    def generate_embedding(self, text: str):
        """Generates embedding for the given text"""
        if not text:
            raise ValueError("Input text is empty")
        embedding_generation_start = time.time()
        logger.info(f"Generating embedding using model '{self.model_id}' ...")
        embedding = self.model.encode(text)
        logger.info(f"Embedding generated in {time.time() - embedding_generation_start:.2f} seconds.")
        return embedding
    
    def semantic_search(self, text1: str, text2: str):
        """Computes semantic similarity between two texts"""
        if not text1 or not text2:
            raise ValueError("Input texts should be non-empty")
        embeddings = self.model.encode([text1, text2])
        similarity = self.model.similarity(embeddings[0], embeddings[1])
        return similarity.numpy()[0][0]


# if __name__ == "__main__":
#     handler = EmbeddingHandler()
#     text_a = ""
#     text_b = ""
#     embedding_a = handler.generate_embedding(text_a)
#     embedding_b = handler.generate_embedding(text_b)
#     print(f"Embedding A: {embedding_a}")
#     print(f"Embedding B: {embedding_b}")
#     similarity_score = handler.semantic_search(text_a, text_b)
#     print(f"Semantic Similarity Score: {similarity_score}")
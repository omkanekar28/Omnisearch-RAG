import torch
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingHandler:
    def __init__(self, model_id: str = "google/embeddinggemma-300M"):
        self.model_id = model_id
        self.model = SentenceTransformer(self.model_id).to(device=device)
    
    def generate_embedding(self, text: str):
        """Generates embedding for the given text"""
        if not text:
            raise ValueError("Input text is empty")
        embedding = self.model.encode(text)
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
#     similarity_score = handler.semantic_search(text_a, text_b)
#     print(f"Semantic Similarity Score: {similarity_score}")
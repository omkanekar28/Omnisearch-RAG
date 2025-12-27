import time
import json
import faiss
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from ..embeddings.embedding_handler import EmbeddingHandler
from ..llm.ollama_handler import OllamaHandler
from ..utils.logger_setup import setup_logger

logger = setup_logger(
    logger_name="rag_engine.py",
    filename="rag_engine.log"
)


class RAGEngine:
    """RAG engine for semantic search and context-based query answering."""

    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        ollama_handler: OllamaHandler,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        distance_threshold: float = 1.5,
        top_k: int = 10
    ) -> None:
        """
        Initializes the RAG engine.
        
        Args:
            index_path: Path to the FAISS index file (.index)
            metadata_path: Path to the metadata JSON file (.json)
            embedding_model: Model ID for generating query embeddings
            distance_threshold: Maximum distance for considering a match
            top_k: Number of top similar results to retrieve
        """
        initialization_start_time = time.time()
        logger.info("Initializing RAG engine ...")
        self.ollama_handler = ollama_handler
        self.embedding_handler = EmbeddingHandler(model_id=embedding_model)
        self.distance_threshold = distance_threshold
        self.top_k = top_k
        
        # LOAD FAISS INDEX AND METADATA
        self.index = self.load_index(index_path)
        self.metadata = self.load_metadata(metadata_path)
        
        logger.info(f"RAG engine initialized in "
                    f"{time.time() - initialization_start_time:.2f} seconds")

    def load_index(self, index_path: str) -> faiss.Index:
        """Loads FAISS index from file."""
        try:
            index_file = Path(index_path)
            if not index_file.exists():
                raise FileNotFoundError(f"Index file not found: {index_path}")
            
            logger.info(f"Loading FAISS index from '{index_path}' ...")
            index = faiss.read_index(str(index_file))
            logger.info(f"FAISS index loaded with {index.ntotal} vectors")
            return index
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index: {e}")

    def load_metadata(self, metadata_path: str) -> List[Dict]:
        """Loads metadata from JSON file."""
        try:
            metadata_file = Path(metadata_path)
            if not metadata_file.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
            logger.info(f"Loading metadata from '{metadata_path}' ...")
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info(f"Metadata loaded with {len(metadata)} entries")
            return metadata
        except Exception as e:
            raise RuntimeError(f"Failed to load metadata: {e}")

    def describe_fetched_results(
        self, 
        results: List[Dict], 
        distances: List[float],
        text_column: str = "Extracted Text"
    ) -> None:
        """
        Logs a summary of fetched results.
        
        Args:
            results: List of matching documents
            distances: Distances for each result
            text_column: Column name containing the text content
        """
        if not results:
            logger.warning("No results feteched.")
            return
        
        logger.info("**************************************** FETCHED RESULTS ****************************************")
        for i, (result, distance) in enumerate(zip(results, distances)):
            text_snippet = result.get(text_column, "N/A")[:100].replace('\n', ' ')
            result_to_print = {k: v for k, v in result.items() if k != text_column}
            result_to_print["Text Snippet"] = text_snippet + "..."
            logger.info(f"{i + 1}. "
                        f"\n{json.dumps(result_to_print, indent=4)}"
                        f"\n   Distance: {distance:.4f}\n\n")
        logger.info("*************************************************************************************************")

    def generate_context(
        self, 
        results: List[Dict], 
        distances: List[float],
        text_column: str = "Extracted Text"
    ) -> str:
        """
        Generates context string from search results.
        
        Args:
            results: List of matching documents
            distances: Distances for each result
            text_column: Column name containing the text content
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant context found."
        
        context_parts = []
        for i, (result, distance) in enumerate(zip(results, distances), 1):
            text = result.get(text_column, "N/A")
            context_parts.append(f"[Result {i} - Distance: {distance:.3f}]\n{text}")
        
        return "\n\n".join(context_parts)

    def search(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        distance_threshold: Optional[float] = None
    ) -> Tuple[List[Dict], List[float]]:
        """
        Searches for similar documents based on the query.
        
        Args:
            query: Search query text
            top_k: Number of results to return (overrides default)
            distance_threshold: Maximum distance for considering a match (overrides default)
            
        Returns:
            Tuple of (matching_documents, distances)
        """
        try:
            search_start_time = time.time()
            logger.info(f"Searching for query: '{query}' ...")
            
            # USE PROVIDED VALUES OR DEFAULTS
            k = top_k if top_k is not None else self.top_k
            threshold = distance_threshold if distance_threshold is not None else self.distance_threshold
            
            # GENERATE & NORMALIZE QUERY EMBEDDING
            query_embedding = self.embedding_handler.generate_embeddings([query]).astype('float32')
            
            # SEARCH FAISS INDEX
            distances, indices = self.index.search(query_embedding, k)
            distances = distances[0]
            indices = indices[0]

            # USE BELOW COMMENTED CODE TO LOG ALL RESULTS
            for i, dist in enumerate(distances):
                logger.info(f"Result {i + 1}: Index={indices[i]}, Distance={dist:.4f}, Text Snippet='{self.metadata[indices[i]].get('Extracted Text', '')[:50]}...'")

            # FILTER BY DISTANCE THRESHOLD
            results = []
            results_distances = []
            for idx, distance in zip(indices, distances):
                if idx < len(self.metadata) and distance <= threshold:
                    results.append(self.metadata[idx])
                    results_distances.append(distance)

            self.describe_fetched_results(results, results_distances)
            return results, results_distances
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RuntimeError(f"Failed to execute search: {e}")
    
    def get_qa_prompt(
        self,
        question: str,
        context: str
    ) -> List[Union[
                SystemMessage, 
                HumanMessage, 
                AIMessage
            ]]:
        """
        Constructs a QA prompt using the provided question and context.
        
        Args:
            question: The question to answer
            context: The context to use for answering
        
        Returns:
            The constructed QA prompt
        """
        messages = [
            SystemMessage(content="You are an AI assistant that provides answers based on the given context. Return only the answer without any additional commentary. If answer is not found in the context, respond accordingly."),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
        ]
        return messages
    
    def perform_question_answering(
        self,
        question: str
    ) -> str:
        """
        Performs question answering by searching for context and generating an answer.
        
        Args:
            question: The question to answer
            
        Returns:
            The generated answer
        """
        try:
            logger.info(f"Performing question answering for: '{question}' ...")
            qa_start_time = time.time()
            rag_metadata, rag_distances = self.search(question)
            context = self.generate_context(
                results=rag_metadata,
                distances=rag_distances
            )
            messages = self.get_qa_prompt(
                question=question,
                context=context
            )
            logger.debug(f"Generated QA Prompt Messages:\n{messages}")
            answer = self.ollama_handler.generate_response_chat(messages)
            logger.info(f"Generated Answer: {answer}")
            logger.info(f"Question answered in {time.time() - qa_start_time:.2f} seconds")
            return answer
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            raise RuntimeError(f"Failed to perform question answering: {e}")


# EXAMPLE USAGE
# if __name__ == "__main__":
#     from ..config.config import (
#         DISTANCE_THRESHOLD, TOP_K, 
#         EMBEDDING_MODEL, 
#         RAG_MODEL_CKPT, RAG_TEMPERATURE, RAG_NUM_PREDICT, RAG_OLLAMA_REASONING, 
#         OLLAMA_BASE_URL
#     )
#     ollama_handler = OllamaHandler(
#         model_ckpt=RAG_MODEL_CKPT,
#         base_url=OLLAMA_BASE_URL,
#         temperature=RAG_TEMPERATURE,
#         num_predict=RAG_NUM_PREDICT, 
#         reasoning=RAG_OLLAMA_REASONING
#     )
#     rag = RAGEngine(
#         index_path="faiss_store/faiss_index.index",
#         metadata_path="faiss_store/metadata.json",
#         ollama_handler=ollama_handler,
#         embedding_model=EMBEDDING_MODEL,
#         distance_threshold=DISTANCE_THRESHOLD,
#         top_k=TOP_K
#     )

#     question_answer_mappings = {
#         "Explain data structures in Python.": "",
#         "What are the advantages of C?": "",
#         "What is HTML primarily used for?": "",
#         "When to use MySQL vs MongoDB?": "",
#         "Explain how Artificial Intelligence, Machine Learning, and Deep Learning are related.": ""
#     }

#     for question in question_answer_mappings.keys():
#         answer = rag.perform_question_answering(question)
#         question_answer_mappings[question] = answer
    
#     print("\n\n******************** QUESTION ANSWERING RESULTS ********************")
#     for question, answer in question_answer_mappings.items():
#         print(f"\t- Q: {question}\nA: {answer}\n\n")
#     print("********************************************************************")
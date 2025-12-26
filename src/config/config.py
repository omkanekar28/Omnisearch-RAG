from typing import Literal

###################################################################################
#                                      COMMON                                     #
###################################################################################
EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
OLLAMA_BASE_URL: str = "http://localhost:11434"

###################################################################################
#                                INGESTION PIPELINE                               #
###################################################################################
DOC_PROCESSOR_TYPE: Literal["doc2txt", "unstructured"] = "unstructured"
DOC_PROCESSOR_MODE: Literal["single", "elements", "paged"] = "single"

IMAGE_MODEL_CKPT: str = "qwen3-vl:2b"
IMAGE_TEMPERATURE: float = 0.7
IMAGE_NUM_PREDICT: int = 2048
IMAGE_OLLAMA_REASONING: bool = True

PDF_MODE: Literal["single", "page"] = "page"
PDF_EXTRACT_IMAGES: bool = False

TEXT_CHUNKER_TYPE: Literal["custom", "langchain"] = "langchain"
TEXT_CHUNK_SIZE: int = 1000
TEXT_CHUNK_OVERLAP: int = 200

WEB_SCRAPER_REQUEST_TIMEOUT: int = 10

EXCEL_MAX_ROWS_THRESHOLD: int = 20
EXCEL_ROWS_OVERLAP: int = 5

SAVE_TO_EXCEL: bool = True
SAVE_TO_JSON: bool = False
MIN_CHUNK_CHARS: int = 50

###################################################################################
#                                    RAG ENGINE                                   #
###################################################################################
RAG_MODEL_CKPT: str = "qwen3:1.7b"
RAG_TEMPERATURE: float = 0.7
RAG_NUM_PREDICT: int = 2048
RAG_OLLAMA_REASONING: bool = True
DISTANCE_THRESHOLD: float = 1.0
TOP_K: int = 5

###################################################################################
#                                   VECTOR STORE                                  #
###################################################################################
USE_GPU: bool = True

###################################################################################
#                                   LOGGER SETUP                                  #
###################################################################################
LOG_STORE_DIR: str = "./logs/"
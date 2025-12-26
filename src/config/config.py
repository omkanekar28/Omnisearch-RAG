from typing import Union
from ..ingestion.doc_processor import Doc2TxtDocProcessor, UnstructuredDocProcessor
from ..ingestion.excel_processor import ExcelProcessor
from ..ingestion.image_processor import OllamaImageProcessor
from ..ingestion.pdf_processor import PyMuPDF4LLMPDFProcessor
from ..ingestion.text_processor import TextProcessor
from ..ingestion.web_scraper import WebScraper
from ..chunking.excel_chunker import ExcelChunker
from ..chunking.text_chunker import CustomTextChunker, LangchainTextChunker
from ..llm.ollama_handler import OllamaHandler


###################################################################################
#                                      COMMON                                     #
###################################################################################
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


###################################################################################
#                                INGESTION PIPELINE                               #
###################################################################################
DOC_PROCESSOR: Union[Doc2TxtDocProcessor, UnstructuredDocProcessor] = UnstructuredDocProcessor(
    mode="single"
)
EXCEL_PROCESSOR: ExcelProcessor = ExcelProcessor()
IMAGE_PROCESSOR: OllamaImageProcessor = OllamaImageProcessor(
    ollama_model_handler=OllamaHandler(
        model_ckpt="qwen3-vl:2b",
        reasoning=False,
        temperature=0.7,
        num_predict=1024,
        base_url="http://localhost:11434"
    ), 
    system_prompt="You are a helpful assistant that extracts content from images. "
        "If the image contains text, return the text as-is. "
        "If the image does not contain text, provide a brief description of the image content.",
)
PDF_PROCESSOR: PyMuPDF4LLMPDFProcessor = PyMuPDF4LLMPDFProcessor(
    mode="page",
    extract_images=False,
    images_parser=None
)
TEXT_PROCESSOR: TextProcessor = TextProcessor()
WEB_SCRAPER: WebScraper = WebScraper(
    request_timeout=10
)
TEXT_CHUNKER: Union[CustomTextChunker, LangchainTextChunker] = LangchainTextChunker(
    chunk_size=1000,
    chunk_overlap=200
)
EXCEL_CHUNKER: ExcelChunker = ExcelChunker(
    max_rows_threshold=20, 
    rows_overlap=5
)
SAVE_TO_EXCEL: bool = True
SAVE_TO_JSON: bool = False
MIN_CHUNK_CHARS: int = 50


###################################################################################
#                                    RAG ENGINE                                   #
###################################################################################
DISTANCE_THRESHOLD: float = 1.5
TOP_K: int = 5


###################################################################################
#                                   VECTOR STORE                                  #
###################################################################################
USE_GPU: bool = True


###################################################################################
#                                   LOGGER SETUP                                  #
###################################################################################
LOG_STORE_DIR: str = "./logs/"
import time
import pandas as pd
import warnings
import json
from typing import Union, List, Dict
from ..ingestion.doc_processor import Doc2TxtDocProcessor, UnstructuredDocProcessor
from ..ingestion.excel_processor import ExcelProcessor
from ..ingestion.image_processor import OllamaImageProcessor
from ..ingestion.pdf_processor import PyMuPDF4LLMPDFProcessor
from ..ingestion.text_processor import TextProcessor
from ..ingestion.web_scraper import WebScraper
from ..chunking.excel_chunker import ExcelChunker
from ..chunking.text_chunker import CustomTextChunker, LangchainTextChunker
from ..utils.logger_setup import setup_logger

# TURN OFF SPECIFIC WARNINGS
warnings.filterwarnings(
    "ignore",
    message="Data Validation extension is not supported and will be removed"
)

logger = setup_logger(
    logger_name="main.py", 
    filename="main.log"
)


class IngestionPipeline:
    """Main OmniSearch Ingestion Pipeline class"""

    def __init__(
        self, 
        doc_processor: Union[Doc2TxtDocProcessor, UnstructuredDocProcessor] = UnstructuredDocProcessor(), 
        excel_processor: ExcelProcessor = ExcelProcessor(),
        image_processor: OllamaImageProcessor = OllamaImageProcessor(),
        pdf_processor: PyMuPDF4LLMPDFProcessor = PyMuPDF4LLMPDFProcessor(),
        text_processor: TextProcessor = TextProcessor(),
        web_scraper: WebScraper = WebScraper(), 
        text_chunker: Union[CustomTextChunker, LangchainTextChunker] = LangchainTextChunker(),
        excel_chunker: ExcelChunker = ExcelChunker(),
        save_to_excel: bool = True,
        save_to_json: bool = False,
        min_chunk_chars: int = 50,
    ) -> None:
        """Initializes the OmniSearch Pipeline with various processors and chunkers"""

        # INITIALIZE PROCESSORS
        self.doc_processor = doc_processor
        self.excel_processor = excel_processor
        self.image_processor = image_processor
        self.pdf_processor = pdf_processor
        self.text_processor = text_processor
        self.web_scraper = web_scraper
        self.save_to_excel = save_to_excel
        self.save_to_json = save_to_json

        # INITIALIZE CHUNKERS
        self.text_chunker = text_chunker
        self.excel_chunker = excel_chunker

        self.outputs = {}
        self.min_chunk_chars = min_chunk_chars
    
    def save_results_to_json(
        self, 
        results: Dict[str, List[str]],
        json_filepath: str = "results.json"
    ) -> None:
        """Saves the results to a JSON file"""
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logger.info(f"Results saved to JSON file: '{json_filepath}'")

    def save_results_to_excel(
        self,
        results: Dict[str, List[str]],
        excel_filepath: str = "results.xlsx"
    ) -> None:
        """Saves the results to an Excel file"""
        all_data = []
        for input_source, text in results.items():
            input_filepath = input_source.split(' | ')[0]
            file_type = input_source.split('.')[-1].lower() if not self.web_scraper.is_valid_url(input_source) else "URL"
            
            if file_type == 'URL':
                file_type = "URL"
                page_no = "N/A"
                chunk_no = "N/A"
            else:
                temp = file_type.split(" | ")
                file_type = temp[0] if len(temp) > 0 else "N/A"
                page_no = temp[1] if len(temp) > 1 else "N/A"
                chunk_no = temp[2] if len(temp) > 2 else "N/A"

            all_data.append({
                "Input Source": input_filepath,
                "File Type": file_type,
                "Page No": page_no,
                "Chunk No": chunk_no,
                "Extracted Text": text
            })
        
        df = pd.DataFrame(all_data)
        df.to_excel(excel_filepath, index=False)
        logger.info(f"Results saved to Excel file: '{excel_filepath}'")
    
    def process_single_input(
        self, 
        input_source: str, 
        processor: Union[
            Doc2TxtDocProcessor, 
            UnstructuredDocProcessor,
            ExcelProcessor, 
            OllamaImageProcessor, 
            PyMuPDF4LLMPDFProcessor, 
            TextProcessor, 
            WebScraper
        ],
        chunker: Union[CustomTextChunker, LangchainTextChunker, ExcelChunker]
    ) -> None:
        """
        Processes a single input source and updates output dict with the extracted chunks

        E.g, file.pdf:-

        {
            "file.pdf | Page 1 | Chunk 1": "abc", 
            "file.pdf | Page 1 | Chunk 2": "def", 
            ...
            "file.pdf | Page N | Chunk M": "xyz"
        }
        """
        raw_output = processor.process(input_source)
        output_chunks = {
            f"Page {op_idx + 1}": chunker(op) for op_idx, op in enumerate(raw_output)
        }
        for page, page_chunks in output_chunks.items():
            for chunk_idx, chunk in enumerate(page_chunks):
                if len(chunk) < self.min_chunk_chars:
                    logger.info(
                        f"Skipping chunk {chunk_idx + 1} on {page} of '{input_source}' "
                        f"due to insufficient length ({len(chunk)} chars)"
                    )
                    continue
                
                self.outputs[f"{input_source} | {page} | Chunk {chunk_idx + 1}"] = chunk

    def __call__(self, inputs: List[str]) -> None:
        """Ingests the given input files/URLs and returns the text present"""

        try:
            # CHECK AND REMOVE DUPLICATES
            original_inputs = inputs.copy()
            inputs = list(dict.fromkeys(inputs))

            if len(original_inputs) != len(inputs):
                logger.warning(
                    f"Duplicate inputs found and removed. "
                    f"Original count: {len(original_inputs)}, "
                    f"Unique count: {len(inputs)}"
                )

            # PROCESS EACH INPUT BASED ON TYPE
            processing_start_time = time.time()
            logger.info(f"Starting processing for {len(inputs)} inputs...")
            for ip_idx, ip in enumerate(inputs):
                try:
                    logger.info(f"Processing input: '{ip}' ({ip_idx + 1} / {len(inputs)})...")
                    ip_extension = f".{ip.split('.')[-1].lower()}"

                    if ip_extension in self.doc_processor.valid_extensions:
                        self.process_single_input(
                            input_source=ip,
                            processor=self.doc_processor,
                            chunker=self.text_chunker
                        )
                    elif ip_extension in self.excel_processor.valid_extensions:
                        self.process_single_input(
                            input_source=ip,
                            processor=self.excel_processor,
                            chunker=self.excel_chunker
                        )
                    elif ip_extension in self.image_processor.valid_extensions:
                        self.process_single_input(
                            input_source=ip,
                            processor=self.image_processor,
                            chunker=self.text_chunker
                        )
                    elif ip_extension in self.pdf_processor.valid_extensions:
                        self.process_single_input(
                            input_source=ip,
                            processor=self.pdf_processor,
                            chunker=self.text_chunker
                        )
                    elif ip_extension in self.text_processor.valid_extensions:
                        self.process_single_input(
                            input_source=ip,
                            processor=self.text_processor,
                            chunker=self.text_chunker
                        )
                    elif self.web_scraper.is_valid_url(ip):
                        self.process_single_input(
                            input_source=ip,
                            processor=self.web_scraper,
                            chunker=self.text_chunker
                        )
                    else:
                        raise ValueError(f"Unsupported input type!")
                    logger.info(f"Completed processing input: '{ip}'")
                except Exception as e:
                    logger.warning(f"Error processing input '{ip}': {e}")
                    self.outputs[ip] = f"[ERROR]: {e}"
            
            # SAVE TO JSON IF ENABLED
            if self.save_to_json:
                try:
                    self.save_results_to_json(
                        results=self.outputs
                    )
                except Exception as e:
                    logger.warning(f"Failed to save results to JSON! {e}")
            
            # SAVE TO EXCEL IF ENABLED
            if self.save_to_excel:
                try:
                    self.save_results_to_excel(
                        results=self.outputs
                    )
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    logger.warning(f"Failed to save results to Excel! {e}")
            
            logger.info(
                f"Processing completed for {len(inputs)} inputs in "
                f"{time.time() - processing_start_time:.2f} seconds"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to process inputs! {e}")


# EXAMPLE USAGE
# if __name__ == "__main__":
#     from ..config.config import (
#         DOC_PROCESSOR, EXCEL_PROCESSOR, IMAGE_PROCESSOR, PDF_PROCESSOR, 
#         TEXT_PROCESSOR, WEB_SCRAPER, TEXT_CHUNKER, EXCEL_CHUNKER, 
#         SAVE_TO_EXCEL, SAVE_TO_JSON, MIN_CHUNK_CHARS
#     )
#     pipeline = IngestionPipeline(
#         doc_processor=DOC_PROCESSOR,
#         excel_processor=EXCEL_PROCESSOR,
#         image_processor=IMAGE_PROCESSOR,
#         pdf_processor=PDF_PROCESSOR,
#         text_processor=TEXT_PROCESSOR,
#         web_scraper=WEB_SCRAPER,
#         text_chunker=TEXT_CHUNKER,
#         excel_chunker=EXCEL_CHUNKER,
#         save_to_excel=SAVE_TO_EXCEL,
#         save_to_json=SAVE_TO_JSON,
#         min_chunk_chars=MIN_CHUNK_CHARS
#     )
#     inputs = [
#         "",  # ADD FILE PATHS OR URLS HERE
#     ]
#     results = pipeline(inputs=inputs)
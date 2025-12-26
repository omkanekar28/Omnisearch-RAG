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
    logger_name="input_ingestion.py", 
    filename="input_ingestion.log"
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
            
            # TRACK EACH INPUT STATUS
            input_status = {ip: "PENDING" for ip in inputs}

            # PROCESS EACH INPUT BASED ON TYPE
            processing_start_time = time.time()
            logger.info(f"Starting processing for {len(inputs)} inputs...")
            for ip_idx, ip in enumerate(inputs):
                ip_status = "PROCESSING"
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
                    ip_status = "COMPLETED"
                    logger.info(f"Completed processing input: '{ip}'")
                except Exception as e:
                    logger.warning(f"Error processing input '{ip}': {e}")
                    self.outputs[ip] = f"[ERROR]: {e}"
                    ip_status = "FAILED"
                finally:
                    input_status[ip] = ip_status
            
            # LOG SUMMARY OF INPUT PROCESSING
            success_count = sum(1 for status in input_status.values() if status == "COMPLETED")
            failed_count = sum(1 for status in input_status.values() if status == "FAILED")
            logger.info("******************** INPUT PROCESSING SUMMARY ********************")
            logger.info(f"\t- Total: {len(inputs)}")
            logger.info(f"\t- Successful: {success_count}")
            logger.info(f"\t- Failed: {failed_count}")
            logger.info("******************************************************************")
            
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
#         DOC_PROCESSOR_TYPE, DOC_PROCESSOR_MODE, 
#         IMAGE_MODEL_CKPT, IMAGE_TEMPERATURE, IMAGE_NUM_PREDICT, 
#         OLLAMA_BASE_URL, IMAGE_OLLAMA_REASONING, 
#         PDF_MODE, PDF_EXTRACT_IMAGES, 
#         TEXT_CHUNKER_TYPE, TEXT_CHUNK_SIZE, TEXT_CHUNK_OVERLAP, 
#         WEB_SCRAPER_REQUEST_TIMEOUT, 
#         EXCEL_MAX_ROWS_THRESHOLD, EXCEL_ROWS_OVERLAP, 
#         SAVE_TO_EXCEL, SAVE_TO_JSON, 
#         MIN_CHUNK_CHARS
#     )
#     from ..llm.ollama_handler import OllamaHandler

#     def build_processors():
#         doc_processor = (
#             Doc2TxtDocProcessor()
#             if DOC_PROCESSOR_TYPE == "doc2txt"
#             else UnstructuredDocProcessor(mode=DOC_PROCESSOR_MODE)
#         )

#         image_processor = OllamaImageProcessor(
#             ollama_model_handler=OllamaHandler(
#                 model_ckpt=IMAGE_MODEL_CKPT,
#                 reasoning=IMAGE_OLLAMA_REASONING,
#                 temperature=IMAGE_TEMPERATURE,
#                 num_predict=IMAGE_NUM_PREDICT,
#                 base_url=OLLAMA_BASE_URL,
#             )
#         )

#         return {
#             "doc": doc_processor,
#             "excel": ExcelProcessor(),
#             "image": image_processor,
#             "pdf": PyMuPDF4LLMPDFProcessor(
#                 mode=PDF_MODE,
#                 extract_images=PDF_EXTRACT_IMAGES,
#             ),
#             "text": TextProcessor(),
#             "web": WebScraper(request_timeout=WEB_SCRAPER_REQUEST_TIMEOUT),
#             "text_chunker": LangchainTextChunker(
#                 chunk_size=TEXT_CHUNK_SIZE,
#                 chunk_overlap=TEXT_CHUNK_OVERLAP,
#             ) if TEXT_CHUNKER_TYPE == "langchain" else CustomTextChunker(
#                 chunk_size=TEXT_CHUNK_SIZE,
#                 chunk_overlap=TEXT_CHUNK_OVERLAP,
#             ),
#             "excel_chunker": ExcelChunker(
#                 max_rows_threshold=EXCEL_MAX_ROWS_THRESHOLD,
#                 rows_overlap=EXCEL_ROWS_OVERLAP,
#             ),
#         }
    
#     processors = build_processors()

#     pipeline = IngestionPipeline(
#         doc_processor=processors["doc"],
#         excel_processor=processors["excel"],
#         image_processor=processors["image"],
#         pdf_processor=processors["pdf"],
#         text_processor=processors["text"],
#         web_scraper=processors["web"],
#         text_chunker=processors["text_chunker"],
#         excel_chunker=processors["excel_chunker"],
#         save_to_excel=SAVE_TO_EXCEL,
#         save_to_json=SAVE_TO_JSON,
#         min_chunk_chars=MIN_CHUNK_CHARS
#     )

#     dir_path = "data/testing"
#     import os
#     inputs = [os.path.join(dir_path, fname) for fname in os.listdir(dir_path)]
#     results = pipeline(inputs=inputs)
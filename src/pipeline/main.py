import time
import pandas as pd
from typing import Union, List, Dict
from ..ingestion.doc_processor import Doc2TxtDocProcessor, UnstructuredDocProcessor
from ..ingestion.excel_processor import ExcelProcessor
from ..ingestion.image_processor import OllamaImageProcessor
from ..ingestion.pdf_processor import PyMuPDF4LLMPDFProcessor
from ..ingestion.text_processor import TextProcessor
from ..ingestion.web_scraper import WebScraper
from ..utils.logger_setup import setup_logger

logger = setup_logger("main.py")


class OmniSearchPipeline:
    """Main OmniSearch Pipeline class"""

    def __init__(
        self, 
        doc_processor: Union[Doc2TxtDocProcessor, UnstructuredDocProcessor] = Doc2TxtDocProcessor(), 
        excel_processor: ExcelProcessor = ExcelProcessor(),
        image_processor: OllamaImageProcessor = OllamaImageProcessor(),
        pdf_processor: PyMuPDF4LLMPDFProcessor = PyMuPDF4LLMPDFProcessor(),
        text_processor: TextProcessor = TextProcessor(),
        web_scraper: WebScraper = WebScraper(), 
        save_to_excel: bool = False
    ) -> None:
        """Initializes the various processors"""
        self.doc_processor = doc_processor
        self.excel_processor = excel_processor
        self.image_processor = image_processor
        self.pdf_processor = pdf_processor
        self.text_processor = text_processor
        self.web_scraper = web_scraper
        self.save_to_excel = save_to_excel
        
    def save_results_to_excel(
        self,
        results: Dict[str, List[str]],
        excel_filepath: str = "results.xlsx"
    ) -> None:
        """Saves the results to an Excel file"""
        all_data = []
        for input_source, texts in results.items():
            file_type = input_source.split('.')[-1].lower() if not self.web_scraper.is_valid_url(input_source) else "URL"

            for page_num, text in enumerate(texts, start=1):
                all_data.append({
                    "Input Source": input_source,
                    "File Type": file_type,
                    "Page/Segment": page_num,
                    "Extracted Text": text
                })
        
        df = pd.DataFrame(all_data)
        df.to_excel(excel_filepath, index=False)
        logger.info(f"Results saved to Excel file: '{excel_filepath}'")

    def __call__(self, inputs: List[str]) -> Dict[str, str]:
        """Ingests the given input files/URLs and returns the text present"""
        outputs = {}

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
                    outputs[ip] = self.doc_processor.process(ip)
                elif ip_extension in self.excel_processor.valid_extensions:
                    outputs[ip] = self.excel_processor.process(ip)
                elif ip_extension in self.image_processor.valid_extensions:
                    outputs[ip] = self.image_processor.process(ip)
                elif ip_extension in self.pdf_processor.valid_extensions:
                    outputs[ip] = self.pdf_processor.process(ip)
                elif ip_extension in self.text_processor.valid_extensions:
                    outputs[ip] = self.text_processor.process(ip)
                elif self.web_scraper.is_valid_url(ip):
                    outputs[ip] = self.web_scraper(ip)
                else:
                    raise ValueError(f"Unsupported input type!")
                logger.info(f"Completed processing input: '{ip}'")
            except Exception as e:
                logger.warning(f"Error processing input '{ip}': {e}")
                outputs[ip] = [f"[ERROR]: {e}"]
        
        # SAVE TO EXCEL IF ENABLED
        if self.save_to_excel:
            try:
                self.save_results_to_excel(
                    results=outputs
                )
            except Exception as e:
                logger.warning(f"Failed to save results to Excel! {e}")
        
        logger.info(
            f"Processing completed for {len(inputs)} inputs in "
            f"{time.time() - processing_start_time:.2f} seconds"
        )
        
        return outputs


# EXAMPLE USAGE
# if __name__ == "__main__":
#     pipeline = OmniSearchPipeline(save_to_excel=True)
#     inputs = [
#         "",  # ADD FILE PATHS OR URLS HERE
#     ]
#     results = pipeline(inputs=inputs)
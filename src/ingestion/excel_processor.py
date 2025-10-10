import time
import pandas as pd
from typing import List
from ..utils.logger_setup import setup_logger
from ..utils.common import validate_file_location, validate_file_type

logger = setup_logger("excel_processor.py")


class ExcelProcessor():
    """Class for Excel processing using Pandas"""

    def __init__(self) -> None:
        """Initializes the file path"""
        self.valid_extensions = {".xl", ".xlsx", ".xlsm", ".xlsb", ".xls"}

    def validate_excel_file(self, filepath: str) -> None:
        """Validates if the given file exists and is an Excel file"""
        validate_file_location(filepath=filepath)
        validate_file_type(
            filepath=filepath,
            valid_extensions=self.valid_extensions
        )
    
    def get_dataframe_from_file(self, filepath: str) -> List[pd.DataFrame]:
        """Uses pandas library to return Dataframe object for the provided file"""
        if filepath.endswith(".csv"):
            return [pd.read_csv(filepath)]
        elif filepath.endswith((".xls", ".xlsx", ".xlsm", ".xlsb")):
            dataframes = []
            xls = pd.ExcelFile(filepath)
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                dataframes.append(df)
            return dataframes
        else:
            raise ValueError(f"Unsupported file format for '{filepath}'!")

    def process(self, filepath: str) -> List[str]:
        """Processes the Excel file and returns text found"""
        try:
            self.validate_excel_file(filepath=filepath)
            
            processing_start_time = time.time()
            logger.info(f"Processing file: '{filepath}' ...")
            dataframes = self.get_dataframe_from_file(filepath=filepath)
            text_data = []
            for idx, df in enumerate(dataframes):
                text_data.append(f"--- Sheet {idx + 1} ---")
                text_data.append(df.to_string(index=False))            
            
            logger.info(
                f"File: '{filepath}' processed in "
                f"{time.time() - processing_start_time:.2f} seconds"
            )
            return text_data
        except Exception as e:
            raise RuntimeError(f"Failed to process '{filepath}' Excel file! {e}")


# EXAMPLE USAGE
# if __name__ == "__main__":
#     processor = ExcelProcessor()
#     results = processor.process(
#         filepath="",
#     )
#     for sheet in results:
#         print(sheet)

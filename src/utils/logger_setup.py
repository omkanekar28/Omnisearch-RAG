import logging
from logging import Logger

def setup_logger(
    logger_name: str, 
    filename: str = "omnisearch_rag.log"
) -> Logger:
    """
    Sets up a logger that:
      - Prints all logs (including DEBUG) to console
      - Writes logs (INFO and above) to a file
    Args:
        logger_name (str): Name of the logger.
        filename (str): Path to the log file.
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)    # Capture all levels internally

    # Prevent duplicate logs if function is called multiple times
    if logger.handlers:
        return logger

    # --- Formatter ---
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # --- Console Handler ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # --- File Handler ---
    file_handler = logging.FileHandler(filename, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)  # Skip DEBUG logs
    file_handler.setFormatter(formatter)

    # --- Add Handlers ---
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

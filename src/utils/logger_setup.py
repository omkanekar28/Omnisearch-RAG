import os
import logging
from logging import Logger
from config.config import LOG_STORE_DIR

def setup_logger(
    logger_name: str, 
    filename: str, 
    store_dir: str = LOG_STORE_DIR
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
    # CREATE FILE IF NOT EXISTS
    filepath = os.path.join(store_dir, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(filepath, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)  # Skip DEBUG logs
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
